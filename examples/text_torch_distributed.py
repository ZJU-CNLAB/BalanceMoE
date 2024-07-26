import torch
import os

from fmoe import DistributedGroupedDataParallel as fmoeDDP
from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import SwitchGate

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
# torchrun --nnodes 1 --nproc_per_node 2 --rdzv_id 0 --rdzv_backend c10d text_torch_distributed.py

class DummyMoEModel(torch.nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.non_moe = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8))
        self.moe = FMoETransformerMLP(
            num_expert=2,
            world_size=world_size,
            d_model=8,
            d_hidden=16,
            top_k = 1,
        )

    def forward(self, inp):
        torch.cuda.nvtx.range_push("Non-MoE")
        out = self.non_moe(inp)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("FMoETransformerMLP")
        out = self.moe(out)
        torch.cuda.nvtx.range_pop()
        return torch.sum(out)

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    model = DummyMoEModel(world_size).to(f"cuda:{local_rank}")
    model = fmoeDDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.0001)

    for i in range(5):
        inp = torch.randn(16, 8).to(f"cuda:{local_rank}")
        opt.zero_grad()
        # if i == 10:
        #     torch.cuda.cudart().cudaProfilerStart()
        loss = model(inp)
        # if i == 15:
        #     torch.cuda.cudart().cudaProfilerStop()
        print('loss:', loss.float().item())
        loss.backward()