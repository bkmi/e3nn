from functools import partial

import torch

from e3nn.kernel import KernelAutoBackward
from e3nn.point.operations import Convolution
from e3nn.point.kernelconv import KernelConv
from e3nn.radial import ConstantRadialModel
from e3nn.rs import dim

torch.set_default_dtype(torch.float64)
device = torch.device('cuda')

Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]
batch = 25
atoms = 40

features = torch.rand(batch, atoms, dim(Rs_in), device=device)
geometry = torch.rand(batch, atoms, 3, device=device)
mask = torch.ones(batch, atoms, device=device)
diff_geom = geometry.unsqueeze(1) - geometry.unsqueeze(2)

k = partial(KernelAutoBackward, RadialModel=ConstantRadialModel)
c = Convolution(k, Rs_in, Rs_out).to(device)

kc = KernelConv(Rs_in, Rs_out, RadialModel=ConstantRadialModel).to(device)


def forward_backward_conv():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    allocated = torch.cuda.max_memory_allocated()
    cached = torch.cuda.max_memory_cached()

    n_features = c(features, geometry)
    target = torch.rand_like(n_features, device=device)
    loss = torch.norm(n_features - target)
    loss.backward()

    print("allocated", (allocated - torch.cuda.max_memory_allocated()) / 1e6)
    print("cached", (cached - torch.cuda.max_memory_cached()) / 1e6)


def forward_backward_kc():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    allocated = torch.cuda.max_memory_allocated()
    cached = torch.cuda.max_memory_cached()

    n_features = kc(features, diff_geom, mask)
    target = torch.rand_like(n_features, device=device)
    loss = torch.norm(n_features - target)
    loss.backward()

    print("allocated", (allocated - torch.cuda.max_memory_allocated()) / 1e6)
    print("cached", (cached - torch.cuda.max_memory_cached()) / 1e6)
