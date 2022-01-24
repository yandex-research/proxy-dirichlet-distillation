import numpy as np
import torch
from tqdm import tqdm

from model import accuracy
from uncertainty import get_model_uncertainty_values, get_ensemble_uncertainty_values, \
    get_ensemble_proxy_uncertainty_values, get_calibration_errors


@torch.cuda.amp.autocast()
def single_model_predictions(model, inputs, label_mask, offset=0):
    logits = model(inputs).float()
    uncertainties = get_model_uncertainty_values(logits.cpu().numpy(), offset=offset)

    probs = torch.softmax(logits[..., label_mask], dim=1)
    confidences, preds = probs.max(dim=1)

    return preds, probs, uncertainties, confidences


@torch.cuda.amp.autocast()
def ensemble_predictions(ensemble, inputs, label_mask):
    member_logits = [model(inputs) for model in ensemble]
    logits = torch.stack(member_logits, dim=1).float()
    uncertainties = get_ensemble_uncertainty_values(logits.cpu().numpy())

    probs = torch.softmax(logits[..., label_mask], dim=2).mean(dim=1)
    confidences, preds = probs.max(dim=1)

    return preds, probs, uncertainties, confidences


@torch.cuda.amp.autocast()
def proxy_predictions(ensemble, inputs, label_mask):
    member_logits = [model(inputs) for model in ensemble]
    logits = torch.stack(member_logits, dim=1).float()
    uncertainties = get_ensemble_proxy_uncertainty_values(logits.cpu().numpy())

    probs = torch.softmax(logits[..., label_mask], dim=2).mean(dim=1)
    confidences, preds = probs.max(dim=1)

    return preds, probs, uncertainties, confidences


@torch.cuda.amp.autocast()
def mimo_predictions(model, inputs, label_mask):
    inputs = torch.tile(torch.unsqueeze(inputs, dim=1), (1, model.ensemble_size, 1, 1, 1))

    logits = model(inputs).mean(1).float()

    uncertainties = get_model_uncertainty_values(logits.cpu().numpy())

    probs = torch.softmax(logits[..., label_mask], dim=1)
    confidences, preds = probs.max(dim=1)

    return preds, probs, uncertainties, confidences


@torch.no_grad()
def get_domain_uncertainties(get_predictions, domain_name, domain_dataloader, label_mask):
    domain_uncertainties = []
    domain_targets = []
    domain_accuracies_top1 = []
    domain_accuracies_top5 = []
    domain_preds = []
    domain_probs = []

    pbar = tqdm(
        domain_dataloader,
        desc=domain_name,
        leave=True,
        dynamic_ncols=True,
        smoothing=0,
    )

    for batch_inputs, batch_targets in pbar:
        preds, probs, uncertainties, confidences = get_predictions(inputs=batch_inputs.cuda(), label_mask=label_mask)
        domain_uncertainties.append(uncertainties)

        acc1, acc5 = accuracy(probs, batch_targets.cuda(), topk=(1, 5), reduce=False)

        domain_accuracies_top1.append(acc1.cpu().numpy())
        domain_accuracies_top5.append(acc5.cpu().numpy())

        domain_targets.append(batch_targets.cpu().numpy())

        domain_preds.append(preds.cpu().numpy())
        domain_probs.append(probs.cpu().numpy())

    uncertainties = domain_uncertainties[0].keys()

    domain_targets = np.concatenate(domain_targets)
    top1_accuracy = np.concatenate(domain_accuracies_top1).sum().item() / len(domain_targets)
    top5_accuracy = np.concatenate(domain_accuracies_top5).sum().item() / len(domain_targets)

    domain_preds = np.concatenate(domain_preds)
    domain_hits = (domain_preds == domain_targets)

    calibration_errors = get_calibration_errors(np.concatenate(domain_probs), domain_hits, domain_targets, num_bins=10)

    domain_uncertainties_concat = {
        uncertainty_measure: np.concatenate([batch[uncertainty_measure] for batch in domain_uncertainties])
        for uncertainty_measure in uncertainties
    }
    return domain_uncertainties_concat, domain_targets, {'acc1': top1_accuracy, 'acc5': top5_accuracy,
                                                         **calibration_errors}, domain_preds
