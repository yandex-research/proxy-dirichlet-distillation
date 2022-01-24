import torch

EPS = torch.finfo(torch.float32).eps


def entropy(probs, dim: int = -1):
    return -(probs * (probs + EPS).log()).sum(dim=dim)


def compute_dirichlet_uncertainties(dirichlet_params, precisions, expected_dirichlet):
    """
    Function which computes measures of uncertainty for Dirichlet model.

    :param dirichlet_params:  Tensor of size [batch_size, n_classes] of Dirichlet concentration parameters.
    :param precisions: Tensor of size [batch_size, 1] of Dirichlet Precisions
    :param expected_dirichlet: Tensor of size [batch_size, n_classes] of probablities of expected categorical under Dirichlet.
    :return: Tensors of token level uncertainties of size [batch_size]
    """
    batch_size, n_classes = dirichlet_params.size()

    entropy_of_expected = entropy(expected_dirichlet)

    expected_entropy = (
        -expected_dirichlet * (torch.digamma(dirichlet_params + 1) - torch.digamma(precisions + 1))).sum(dim=-1)

    mutual_information = -((expected_dirichlet + EPS) * (
        torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + 1 + EPS) + torch.digamma(
        precisions + 1 + EPS))).sum(dim=-1)
    # assert torch.allclose(mutual_information, entropy_of_expected - expected_entropy, atol=1e-4, rtol=0)

    epkl = (n_classes - 1) / precisions.squeeze(-1)

    mkl = (expected_dirichlet * (
        torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + EPS) + torch.digamma(
        precisions + EPS))).sum(dim=-1)

    return entropy_of_expected.clamp(min=0), \
           expected_entropy.clamp(min=0), \
           mutual_information.clamp(min=0), \
           epkl.clamp(min=0), \
           mkl.clamp(min=0)


def forward_kl_mean_loss(logits, ensemble_stats, **kwargs):
    log_probs = torch.log_softmax(logits, dim=-1)

    stats = {}

    avg_teacher_probs = ensemble_stats['mean_probs']
    loss = torch.nn.functional.kl_div(log_probs, avg_teacher_probs, reduction='none').sum(-1)

    return torch.mean(loss), stats


def dirichlet_likelihood_loss(logits, ensemble_stats, model_offset, **kwargs):
    alphas, precision = get_dirichlet_parameters(logits, add_to_alphas=model_offset)

    unsqueezed_precision = precision.unsqueeze(1)
    normalized_probs = alphas / unsqueezed_precision
    entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_dirichlet_uncertainties(alphas,
                                                                                                           unsqueezed_precision,
                                                                                                           normalized_probs)

    stats = {
        'alpha_min': alphas.min(),
        'alpha_mean': alphas.mean(),
        'precision': precision,
        'entropy_of_expected': entropy_of_expected,
        'mutual_info': mutual_information,
        'mkl': mkl,
    }

    teacher_probs = ensemble_stats['probs']

    log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + EPS), dim=-2)

    # Define the cost in two parts (dependent on targets and independent of targets)
    target_independent_term = (torch.sum(torch.lgamma(alphas + EPS), dim=-1) - torch.lgamma(precision + EPS))
    target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=-1)
    cost = target_dependent_term + target_independent_term

    return torch.mean(cost), stats


def proxy_loss(logits, ensemble_stats, model_offset, target_offset):
    alphas, precision = get_dirichlet_parameters(logits, model_offset)

    unsqueezed_precision = precision.unsqueeze(1)
    normalized_probs = alphas / unsqueezed_precision
    entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_dirichlet_uncertainties(alphas,
                                                                                                           unsqueezed_precision,
                                                                                                           normalized_probs)

    stats = {
        'alpha_min': alphas.min(),
        'alpha_mean': alphas.mean(),
        'precision': precision,
        'entropy_of_expected': entropy_of_expected,
        'mutual_info': mutual_information,
        'mkl': mkl,
    }

    num_classes = alphas.size(-1)

    ensemble_precision = ensemble_stats['precision']

    ensemble_params = ensemble_stats['mean_probs'] * ensemble_precision + target_offset
    ensemble_precision += target_offset * num_classes

    target_independent_term = (
        torch.lgamma(ensemble_precision.squeeze(1)) - torch.sum(torch.lgamma(ensemble_params), dim=-1) +
        torch.sum(torch.lgamma(alphas), dim=-1) - torch.lgamma(precision)
    )

    target_dependent_term = torch.sum(
        (ensemble_params - alphas) *
        (torch.digamma(ensemble_params) - torch.digamma(ensemble_precision)),
        dim=-1)

    cost = target_dependent_term + target_independent_term
    return torch.mean(cost), stats


def rkl_proxy_loss(logits, ensemble_stats, model_offset, target_offset):
    alphas, precision = get_dirichlet_parameters(logits, model_offset)

    unsqueezed_precision = precision.unsqueeze(1)
    normalized_probs = alphas / unsqueezed_precision

    entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_dirichlet_uncertainties(alphas,
                                                                                                           unsqueezed_precision,
                                                                                                           normalized_probs)

    stats = {
        'alpha_min': alphas.min(),
        'alpha_mean': alphas.mean(),
        'precision': precision,
        'entropy_of_expected': entropy_of_expected,
        'mutual_info': mutual_information,
        'mkl': mkl,
    }

    num_classes = alphas.size(-1)

    ensemble_precision = ensemble_stats['precision']

    ensemble_precision += target_offset * num_classes
    ensemble_probs = ensemble_stats['mean_probs']

    expected_KL_term = -1.0 * torch.sum(ensemble_probs * (torch.digamma(alphas + EPS)
                                                          - torch.digamma(precision.unsqueeze(-1) + EPS)), dim=-1)
    assert torch.isfinite(expected_KL_term).all(), (torch.max(alphas), torch.max(precision), alphas.dtype)

    differential_negentropy_term = torch.sum(torch.lgamma(alphas + EPS), dim=-1) - torch.lgamma(precision + EPS) \
                                   - torch.sum(
        (alphas - 1) * (torch.digamma(alphas + EPS) - torch.digamma(precision.unsqueeze(-1) + EPS)), dim=-1)
    assert torch.isfinite(differential_negentropy_term).all()

    cost = expected_KL_term - differential_negentropy_term / ensemble_precision.squeeze(-1)

    assert torch.isfinite(cost).all()
    return torch.mean(cost), stats


losses_dict = {
    'forward_kl': forward_kl_mean_loss,
    'dirichlet_likelihood': dirichlet_likelihood_loss,
    'proxy': proxy_loss,
    'rkl_proxy': rkl_proxy_loss,
}


def get_dirichlet_parameters(logits, add_to_alphas=0, dtype=torch.double):
    max_val = torch.finfo(dtype).max / logits.size(-1) - 1
    alphas = torch.clip(torch.exp(logits.to(dtype=dtype)) + add_to_alphas, max=max_val)
    precision = torch.sum(alphas, dim=-1, dtype=dtype)
    return alphas, precision
