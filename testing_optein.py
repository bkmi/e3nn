from time import perf_counter_ns

import torch

from e3nn.tensor_product import TensorProduct, ElementwiseTensorProduct
from e3nn.SO3 import clebsch_gordan


def check_memory():
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    clock = perf_counter_ns()

    def printit():
        print((torch.cuda.memory_allocated() - before_allocated) / 1e6,
              (torch.cuda.memory_reserved() - before_reserved) / 1e6,
              perf_counter_ns() - clock)
    return printit


stuff = torch.load('/home/mi/milleb92/sci/equivariant-benchmark/kernel.pkl')

set_of_l_filters = stuff['set_of_l_filters']
l_filters = stuff['l_filters']
Y = stuff['Y']
l_out = stuff['l_out']
l_in = stuff['l_in']
kernel = stuff['kernel']
sub_norm_coef = stuff['sub_norm_coef']
sub_R = stuff['sub_R']

#############################
# Can we split up the einsum and get something for lower memory which is also faster?
# That is the goal of this first section!
#############################

# just get things setup. Really nothing else going on here.
for k, l_filter in enumerate(l_filters):
    tmp = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
    sub_Y = Y[tmp: tmp + 2 * l_filter + 1]  # [m, batch]
    C = clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel)  # [m_out, m_in, m]

# This is an example of things "as they are"
p = check_memory()
out_true = torch.einsum("ijk,kz,zuv,z->zuivj", (C, sub_Y, sub_R[..., k], sub_norm_coef))
p()

# This is a split, just from intuition. Using opt-einsum we might be able to find a better "einsum path"
p = check_memory()
midstate = torch.einsum("ijk,kz,z->zij", C, sub_Y, sub_norm_coef)
out_attempt = torch.einsum('zij,zuv->zuivj', midstate, sub_R[..., k])
p()

# It's bad!! (not even working right now, but when it was tensordot and permute was slower!
# p = check_memory()
# midstate1 = torch.einsum("ijk,kz->zij", C, sub_Y)
# midstate2 = torch.tensordot(C, sub_Y, dims=1).permute([2, 0, 1]) * sub_norm_coef
# out_other = torch.einsum(
#     'zij,zuv->zuivj',
#     midstate,
#     sub_R[..., k]
# )
# p()

assert torch.allclose(out_true, out_attempt)

#######################
# In the following section I will attempt to take the sub_R out of the loop. After all, it looks like the sub_R
# multiplication is really slowing down the heart of the kernel and increasing the memory profile, as seen above.
#######################
K = 0
Ks = []
for k, l_filter in enumerate(l_filters):
    tmp = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
    sub_Y = Y[tmp: tmp + 2 * l_filter + 1]  # [m, batch]

    C = clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel)  # [m_out, m_in, m]

    # note: The multiplication with `sub_R` could also be done outside of the for loop
    K_mid = torch.einsum("ijk,kz,z->zij", C, sub_Y, sub_norm_coef)  # [batch, m_out, m_in]
    Ks.append(K_mid)
K = torch.einsum('zij,zuvx->zuivjx', K_mid, sub_R)  # [batch, mul_out, m_out, mul_in, m_in]

# If the following for loop does not yield equality then I don't think it's possible to take out the sub_R
for i, k in enumerate(Ks):
    print(torch.allclose(K[..., i], torch.einsum('zij,zuv->zuivj', k, sub_R[..., i])))
# Since each element isn't equal, I don't think we can remove the R from the loop. I tried this signature which is
# essentially a elementwise multiplication followed by a sum
# K = torch.einsum('zij,zuvx->zuivj', K_mid, sub_R) # <- wrong answer!

# This is the kernel as is.
p = check_memory()
G = 0
for k, l_filter in enumerate(l_filters):
    tmp = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
    sub_Y = Y[tmp: tmp + 2 * l_filter + 1]  # [m, batch]

    C = clebsch_gordan(l_out, l_in, l_filter, cached=True, like=kernel)  # [m_out, m_in, m]

    # note: The multiplication with `sub_R` could also be done outside of the for loop
    G_mid = torch.einsum("ijk,kz,z->zij", C, sub_Y, sub_norm_coef)  # [batch, m_out, m_in]
    G += torch.einsum('zij,zuv->zuivj', G_mid, sub_R[..., k])  # [batch, mul_out, m_out, mul_in, m_in]
p()

assert torch.allclose(K.sum(dim=-1), G)


# Nothing special here... I was going to experiment with making this block diagonal matrix as we discussed.
geo = torch.rand(16, 40, 3, dtype=torch.float32)
features = torch.rand(16, 40, 2, dtype=torch.float32)
filter = torch.rand(16, 40, 3)
Rs_feature = [(2, 0)]
Rs_filter = [(1, 1)]

etp = TensorProduct(Rs_feature, Rs_filter)
