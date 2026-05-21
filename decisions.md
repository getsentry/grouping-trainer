# Decisions

> Why does loss.py use a pairwise loss from 2005?

TLDR: We invested in and trust the labels.

The data were sampled from the previous (v1) embedding model's decision boundary. These pairs are more interesting and
are inherently the ones where a successor stands to gain. Easy stacktrace pairs are likely easy for every model. These
pairs typically have completely different errors and frames. Sampling weights decreased as the distance from v1's
boundary increased. Pairs were formed via a self-join per Sentry project. There are no other constraints around the
data's structure besides a limit on the number of stacktraces fetched from table B for each stacktrace in table A of the
self-join, and a distance range which was determined via eye-test: no pairs have a v1 cosine distance outside the range
[0.001, 0.50]. A stacktrace can be similar or dissimilar to any number of other stacktraces. This dataset has:

  - Some stacktraces which are similar to 20 others
  - Some which are dissimilar to 20
  - Some which are similar to 0
  - Some which are dissimlar to 0.

Labels were made by prompting Claude Sonnet 4.5 w/ thinking. The prompt was carefully iterated to follow a set of basic
Sentry guidelines around grouping, and to align w/ human-labeled pairs sampled from internal production Sentry projects.
The human labelers were the Sentry grouping czars—employees who have mulled over stacktrace similarity for years. We
generally err on the side of avoiding overgrouping—wrongly grouping two errors into one issue. Hiding new errors is
costly; Sentry's job is to tell you that something in prod broke.

There are many ways to train an embeddings model from here. You could, e.g., collate in-batch negatives and train using
symmetric MNRL—the "standard" loss today. The immediate problem is that in-batch negatives can easily be false negatives
in this dataset. Across pairs and w/in a project, there are plenty of similar stacktraces that aren't explicitly labeled
as similar. MNRL is especially sensitive to the false negative rate. This sensitivity is good when you're training for
the globally sparse similarity structures encountered in RAG, but bad when you're training for fine-grained stacktrace
similarity w/in a Sentry project. GISTEmbed aims to address this kind of issue, but there aren't good guide models for
this niche task. `gemini-embedding-2`, e.g.,
[flops](https://github.com/getsentry/grouping-trainer/tree/main/eval/comparisons/test_full3/gemini-cluster_dim3072_vs_large-no-prefix_dim64).

You could instead have the batch sampler pair up stacktraces from different Sentry projects to avoid false negatives.
These examples are too easy; different Sentry projects have different code -> very different stacktrace frames. At
test-time, the DB query doesn't compare stacktraces across projects. A more basic problem w/ MNRL is that the batching
procedure needs to handle stacktraces that aren't similar to anything in the dataset or are similar to multiple
stacktraces by padding and masking the loss, or adding a pairwise loss term over lone pairs.

Another idea is to use a triplet loss. Triplet sampling is well-defined when an input is classified. It's ambiguous for
our dataset which has variable numbers of similar and dissimilar stacktraces. I'm not sure there's a principled way to
select from the many combinations of triplets without dropping explicitly labeled stacktrace relationships.

In general, non-pairwise losses coerce a jagged but rich similarity structure into a rectangular one. The data
intentionally contains hard positives and hard negatives. Pairwise losses put their faith in the data and accomodate the
jagged structure by melting it into a rectangular one.

Pairwise losses do have downsides. A statistical downside is that we don't have many negatives per positive, which is
risky since we need to limit overgrouping. To softly mitigate this, the sampling procedure allocated more weight to
pairs on the dissimilar side of the decision boundary, and synthetic.py mines more negatives offline to avoid
regressions wrt v1. A computational downside is that a random scan over pairs results in lots of re-computations of the
same stacktrace across training. (It's generally good to see the same input across training, but empirically it [didn't
matter much
here](https://github.com/getsentry/grouping-trainer/tree/main/eval/comparisons/test_full3/grouped-scan_dim768_vs_random-scan_dim768).)
This is mitigated by pre-sorting each Sentry project by stacktrace, having the batch-sampler pick a single project per
GPU, and having `ModelForTraining.encode` gather deduplicated embeddings in the forwards pass and scatter them in the
backwards pass. This dance, courtesy of torch's autograd, cuts training time by ~2.8x.

The 2005 `ContrastiveLoss` outperformed `SigmoidLoss`. I haven't studied this result much yet, e.g., confirming if this
is caused by `ContrastiveLoss` being more forgiving by comparing it to a label-smoothed `SigmoidLoss`.
