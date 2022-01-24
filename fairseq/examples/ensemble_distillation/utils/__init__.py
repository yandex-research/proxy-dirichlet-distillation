import torch


def get_dirichlet_parameters(
    net_output, add_to_alphas=0, dtype=torch.float
):
    logits, extra = net_output
    max_val = torch.finfo(dtype).max / logits.size(1) - 1

    alphas = torch.clamp(
        torch.exp(logits.to(dtype=dtype)) + add_to_alphas, max=max_val
    )
    precision = torch.sum(alphas, dim=-1, dtype=dtype)
    return alphas, precision
