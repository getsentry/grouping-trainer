# NCCL_NET=Socket LD_LIBRARY_PATH="" torchrun --nproc_per_node=2 bin/nccl_test.py
# Without Socket override (test if native networking works):
# LD_LIBRARY_PATH="" torchrun --nproc_per_node=2 bin/nccl_test.py

import torch
import torch.distributed as dist
import os

rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl")
print(f"Rank {rank}: initialized", flush=True)

t = torch.ones(4, device=f"cuda:{rank}")
dist.all_reduce(t)
print(f"Rank {rank}: all_reduce result = {t}", flush=True)

dist.destroy_process_group()
print(f"Rank {rank}: done", flush=True)
