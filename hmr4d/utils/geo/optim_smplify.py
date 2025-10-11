import torch
from hmr4d.model.smplify.losses import SimpleSMPLifyLoss
from hmr4d.utils.smplx_utils import make_smplx


def run_smplify(param_dict, helper_dict, loss_fn):
    loss_fn.to_cuda()  # if not already on cuda
    loss_fn.set_init_params(param_dict)

    B, L = param_dict["body_pose"].shape[:2]
    lr = 1e-2
    num_steps = 2

    # # Stage 1. Optimize transl only
    # param_dict = {k: v.detach().clone().requires_grad_(True) for k, v in param_dict.items()}
    # # optimizer = torch.optim.LBFGS([param_dict["transl"]], lr=lr, max_iter=num_iters, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adam([param_dict["transl"]], lr=lr)
    # closure = loss_fn.create_closure(optimizer, param_dict, helper_dict)
    # for j in range(num_steps):
    #     optimizer.zero_grad()
    #     loss = optimizer.step(closure)
    #     print(f"Loss: {loss.item():.1f}")

    # Stage 2. Optimize all
    param_dict = {k: v.detach().clone().requires_grad_(True) for k, v in param_dict.items()}
    scale_factor = B * L
    optimizer = torch.optim.LBFGS(param_dict.values(), lr=lr * scale_factor, max_iter=5, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adam(param_dict.values(), lr=lr * B)
    closure = loss_fn.create_closure(optimizer, param_dict, helper_dict)

    for j in range(num_steps):
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        print(f"Loss: {loss.item():.1f}")

    param_dict = {k: v.detach() for k, v in param_dict.items()}
    return param_dict
