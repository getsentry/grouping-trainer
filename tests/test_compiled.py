import contextlib

import pytest
import torch

import grouping_trainer as gt

COMPILED_TOKEN_BUCKETS = (64, 128, 256)
BUCKET_EXCEEDING_MAX_SEQ_LENGTH = 999_999


@pytest.fixture(scope="module")
def sentence_transformer(model_name: str) -> gt.compiled.SentenceTransformer:
    """
    Load a compiled SentenceTransformer on CPU without calling compile_and_warm_up.
    """
    return gt.compiled.SentenceTransformer(
        model_name,
        compiled_batch_size=1,
        compiled_token_buckets=(*COMPILED_TOKEN_BUCKETS, BUCKET_EXCEEDING_MAX_SEQ_LENGTH),
    )


class TestInit:
    def test_filters_buckets_exceeding_max_seq_length(
        self, sentence_transformer: gt.compiled.SentenceTransformer
    ) -> None:
        assert all(bucket in sentence_transformer._compiled_token_buckets for bucket in COMPILED_TOKEN_BUCKETS)
        assert BUCKET_EXCEEDING_MAX_SEQ_LENGTH not in sentence_transformer._compiled_token_buckets


class TestTokenize:
    @pytest.mark.parametrize(
        ("text", "expected_bucket"),
        [
            ("hello world", 64),
            ("word " * 50, 64),
            ("word " * 100, 128),
            ("word " * 200, 256),
        ],
    )
    def test_pads_to_nearest_bucket(
        self, sentence_transformer: gt.compiled.SentenceTransformer, text: str, expected_bucket: int
    ) -> None:
        result = sentence_transformer.tokenize([text])
        num_tokens = result["input_ids"].shape[1]
        assert num_tokens == expected_bucket

    def test_no_padding_when_exceeding_all_buckets(self, sentence_transformer: gt.compiled.SentenceTransformer) -> None:
        long_text = "word " * 1000
        result = sentence_transformer.tokenize([long_text])
        num_tokens = result["input_ids"].shape[1]
        assert num_tokens > max(COMPILED_TOKEN_BUCKETS)
        assert num_tokens not in sentence_transformer._compiled_token_buckets

    def test_batch_size_mismatch_skips_padding(self, sentence_transformer: gt.compiled.SentenceTransformer) -> None:
        # compiled_batch_size is 1, passing 2 texts triggers early return without padding
        result = sentence_transformer.tokenize(["hello", "world"])
        num_tokens = result["input_ids"].shape[1]
        assert num_tokens < min(COMPILED_TOKEN_BUCKETS)


@contextlib.contextmanager
def _training_mode(model: gt.compiled.SentenceTransformer, training: bool):
    was_training = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(was_training)


@pytest.fixture
def baseline_matmul_precision():
    """
    Set the global precision to something different from _COMPILED_MATMUL_PRECISION so tests
    can verify the compiled paths flip it and restore it on exit.
    """
    baseline: gt.compiled.MatmulPrecision = (
        "highest" if gt.compiled._COMPILED_MATMUL_PRECISION != "highest" else "medium"
    )
    with gt.compiled._set_float32_matmul_precision(baseline):
        yield baseline


class TestCompileAndWarmUp:
    def test_uses_compiled_precision_and_restores_global(
        self,
        sentence_transformer: gt.compiled.SentenceTransformer,
        monkeypatch: pytest.MonkeyPatch,
        baseline_matmul_precision: str,
    ) -> None:
        observed_precision: list[str] = []

        def encode_with_precision_check(*args, **kwargs):
            observed_precision.append(torch.get_float32_matmul_precision())

        monkeypatch.setattr(sentence_transformer, "encode", encode_with_precision_check)
        monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: None)
        # compile_and_warm_up mutates _compiled_forward
        monkeypatch.setattr(sentence_transformer, "_compiled_forward", sentence_transformer._compiled_forward)

        sentence_transformer.compile_and_warm_up()

        assert all(p == gt.compiled._COMPILED_MATMUL_PRECISION for p in observed_precision)
        assert len(observed_precision) > 0
        assert torch.get_float32_matmul_precision() == baseline_matmul_precision


class TestForward:
    def test_raises_in_training_mode(self, sentence_transformer: gt.compiled.SentenceTransformer) -> None:
        with _training_mode(sentence_transformer, training=True):
            dummy_input = {"input_ids": torch.zeros(1, 64, dtype=torch.long)}
            with pytest.raises(ValueError, match="training"):
                sentence_transformer.forward(dummy_input)

    def test_raises_when_compile_and_warm_up_not_called(
        self, sentence_transformer: gt.compiled.SentenceTransformer
    ) -> None:
        with _training_mode(sentence_transformer, training=False):
            dummy_input = {
                "input_ids": torch.zeros(1, 64, dtype=torch.long),
                "attention_mask": torch.ones(1, 64, dtype=torch.long),
            }
            with pytest.raises(ValueError, match="compile_and_warm_up"):
                sentence_transformer.forward(dummy_input)

    def test_encode_uses_compiled_precision_and_restores_global(
        self,
        sentence_transformer: gt.compiled.SentenceTransformer,
        monkeypatch: pytest.MonkeyPatch,
        baseline_matmul_precision: str,
    ) -> None:
        observed_precision: list[str] = []

        def forward_with_precision_check(input: dict[str, torch.Tensor], **kwargs):
            observed_precision.append(torch.get_float32_matmul_precision())
            return gt.utils.SentenceTransformer.forward(sentence_transformer, input, **kwargs)

        monkeypatch.setattr(sentence_transformer, "_compiled_forward", forward_with_precision_check)

        sentence_transformer.encode("hello", show_progress_bar=False)

        assert observed_precision == [gt.compiled._COMPILED_MATMUL_PRECISION]
        assert torch.get_float32_matmul_precision() == baseline_matmul_precision


_MODEL_NAMES_FOR_TOKENIZE_TESTS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # WordPiece, max_seq 256
    "BAAI/bge-small-en-v1.5",  # WordPiece + token_type_ids, max_seq 512
    "Alibaba-NLP/gte-modernbert-base",  # BPE, max_seq 8192
]


@pytest.fixture(scope="module")
def encoder_indirect(request: pytest.FixtureRequest) -> gt.utils.SentenceTransformer:
    return gt.utils.SentenceTransformer(request.param)


class TestCreateTextWithNumTokens:
    @pytest.mark.parametrize("encoder_indirect", _MODEL_NAMES_FOR_TOKENIZE_TESTS, indirect=True)
    @pytest.mark.parametrize("target_num_tokens", [8, 13, 47, 64, 199, 256, 777, 1024, 4001])
    def test_lands_exactly_on_target(
        self, encoder_indirect: gt.utils.SentenceTransformer, target_num_tokens: int
    ) -> None:
        if target_num_tokens > encoder_indirect.max_seq_length:
            pytest.skip(f"target {target_num_tokens} > max_seq {encoder_indirect.max_seq_length}")
        text = gt.compiled._create_text_with_num_tokens(target_num_tokens, encoder_indirect.tokenize)
        actual_num_tokens = encoder_indirect.tokenize([text])["input_ids"].shape[1]
        assert actual_num_tokens == target_num_tokens

    @pytest.mark.parametrize("encoder_indirect", _MODEL_NAMES_FOR_TOKENIZE_TESTS, indirect=True)
    @pytest.mark.parametrize("offset", [-1, 0])
    def test_lands_at_max_seq_boundary(self, encoder_indirect: gt.utils.SentenceTransformer, offset: int) -> None:
        target_num_tokens = encoder_indirect.max_seq_length + offset
        text = gt.compiled._create_text_with_num_tokens(target_num_tokens, encoder_indirect.tokenize)
        actual_num_tokens = encoder_indirect.tokenize([text])["input_ids"].shape[1]
        assert actual_num_tokens == target_num_tokens


class TestEncode:
    @pytest.mark.parametrize(
        ("encode_kwargs", "expected_tokenize_batch_sizes"),
        [
            # No batch_size -> defaults to _compiled_batch_size=1, so 3 texts tokenize individually.
            ({}, [1, 1, 1]),
            # User-provided batch_size is respected, so all 3 texts tokenize together.
            ({"batch_size": 3}, [3]),
        ],
    )
    def test_batch_size_kwarg(
        self,
        sentence_transformer: gt.compiled.SentenceTransformer,
        monkeypatch: pytest.MonkeyPatch,
        encode_kwargs: dict,
        expected_tokenize_batch_sizes: list[int],
    ) -> None:
        tokenize_batch_sizes: list[int] = []
        original_tokenize = sentence_transformer.tokenize

        def tokenize_with_capture(texts, **kwargs):
            tokenize_batch_sizes.append(len(texts))
            return original_tokenize(texts, **kwargs)

        monkeypatch.setattr(sentence_transformer, "tokenize", tokenize_with_capture)
        monkeypatch.setattr(
            sentence_transformer,
            "_compiled_forward",
            lambda input, **kwargs: gt.utils.SentenceTransformer.forward(sentence_transformer, input, **kwargs),
        )

        sentence_transformer.encode(["hello", "world", "foo"], show_progress_bar=False, **encode_kwargs)

        assert tokenize_batch_sizes == expected_tokenize_batch_sizes
