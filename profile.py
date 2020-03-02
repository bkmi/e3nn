import torch

from e3nn.kernel import Kernel, KernelAutoBackward
from e3nn.radial import ConstantRadialModel


torch.set_default_dtype(torch.float64)
device = torch.device('cuda')

Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]
batch = 100
atoms = 40

k = Kernel(Rs_in, Rs_out, RadialModel=ConstantRadialModel).to(device)
kab = KernelAutoBackward(Rs_in, Rs_out, RadialModel=ConstantRadialModel).to(device)


def forward_backward(kernel):
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    a = torch.cuda.max_memory_allocated()
    c = torch.cuda.max_memory_cached()
    geometry = torch.rand(batch, atoms, 3).to(device)
    features = kernel(geometry)
    target = torch.rand_like(features).to(device)
    loss = torch.norm(features - target)
    loss.backward()
    print("allocated", (a - torch.cuda.max_memory_allocated()) / 1e6)
    print("cached", (c - torch.cuda.max_memory_cached()) / 1e6)
