import torch

from e3nn.point.kernelconv import KernelConv
from e3nn.radial import ConstantRadialModel
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid
from e3nn.rs import dim


torch.set_default_dtype(torch.float64)
torch.backends.cudnn.deterministic = True

Rs_in = [(1, 0), (1, 1), (2, 0), (1, 2)]
Rs_out = [(2, 0), (1, 1), (1, 2), (3, 0)]

batch = 100
atoms = 40
geometry = torch.rand(batch, atoms, 3)
rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
r = rb - ra
features = torch.rand(batch, atoms, dim(Rs_in), requires_grad=True)
mask = torch.ones(batch, atoms)

act = GatedBlock(Rs_out, sigmoid, sigmoid)
conv = KernelConv(Rs_in, act.Rs_in, ConstantRadialModel)

a = conv(features, r, mask)
b = act(a)

print('done')
