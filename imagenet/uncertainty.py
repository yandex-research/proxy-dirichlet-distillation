from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, softmax
from sklearn.metrics import roc_auc_score, auc

EPS = 1e-10

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


def kl_divergence(probs1, probs2):
    return np.sum(probs1 * (np.log(probs1 + EPS) - np.log(probs2 + EPS)), axis=1)


def expected_pairwise_kl_divergence(probs):
    kl = np.zeros((probs.shape[0],), dtype=np.float32)
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            kl += kl_divergence(probs[:, i, :], probs[:, j, :])

    return kl


def entropy_of_expected(probs):
    mean_probs = np.mean(probs, axis=1)
    log_probs = -np.log(mean_probs + EPS)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy(probs):
    log_probs = -np.log(probs + EPS)
    return np.mean(np.sum(probs * log_probs, axis=2), axis=1)


def get_model_uncertainty_values(logits, offset=0):
    # logits [batch_size, num_classes]
    alphas = np.exp(logits) + offset
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0

    conf = np.max(probs, axis=1)

    entropy_of_exp = -np.sum(probs * np.log(probs + EPS), axis=1)
    expected_entropy = -np.sum(probs * (digamma(alphas + 1) - digamma(alpha0 + 1)), axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = np.squeeze((alphas.shape[1] - 1) / alpha0)
    mkl = epkl - mutual_info

    uncertainty = {'confidence': -conf,
                   'entropy_of_expected': entropy_of_exp,
                   'expected_entropy': expected_entropy,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'MKL': mkl,
                   }

    return uncertainty


def get_ensemble_uncertainty_values(ensemble_logits):
    # ensemble_logits [batch_size, ensemble_size, num_classes]
    probs = softmax(ensemble_logits, axis=2)
    mean_probs = np.mean(probs, axis=1)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected(probs)
    exe = expected_entropy(probs)
    mutual_info = eoe - exe

    epkl = expected_pairwise_kl_divergence(probs)
    mkl = epkl - mutual_info

    uncertainty = {'confidence': -conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'MKL': mkl}

    return uncertainty


def get_ensemble_proxy_uncertainty_values(ensemble_logits):
    # ensemble_logits [batch_size, ensemble_size, num_classes]
    probs = softmax(ensemble_logits, axis=2)
    mean_probs = np.mean(probs, axis=1)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected(probs)
    exe = expected_entropy(probs)
    mutual_info = eoe - exe

    epkl = expected_pairwise_kl_divergence(probs)
    mkl = epkl - mutual_info

    num_classes = probs.shape[1]
    alpha0 = (num_classes - 1) / (2 * mkl[:, None] + EPS)
    alphas = mean_probs * alpha0

    proxy_eoe = -np.sum(mean_probs * np.log(mean_probs + EPS), axis=1)
    proxy_exe = -np.sum((alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1)),
                        axis=1)
    proxy_mutual_info = proxy_eoe - proxy_exe

    proxy_epkl = np.squeeze((alphas.shape[1] - 1) / alpha0)
    proxy_mkl = epkl - proxy_mutual_info

    uncertainty = {'confidence': -conf,
                   'entropy_of_expected': proxy_eoe,
                   'expected_entropy': proxy_exe,
                   'mutual_information': proxy_mutual_info,
                   'EPKL': proxy_epkl,
                   'MKL': proxy_mkl}

    return uncertainty


def compute_ood_auc(in_domain_predictions, out_of_domain_predictions):
    domain_labels = np.concatenate(
        (np.zeros((in_domain_predictions.shape[0])), np.ones(out_of_domain_predictions.shape[0]))
    )
    domain_scores = np.concatenate((in_domain_predictions, out_of_domain_predictions))

    roc_auc = roc_auc_score(domain_labels, domain_scores)
    return roc_auc


def combine_ood_metrics(imagenet_uncertainties, imagenet_targets, ood_uncertainties):
    ood_metrics = dict()

    for domain_name, (
        (domain_uncertainties, domain_targets, domain_accuracy, domain_preds),
        domain_mask) in ood_uncertainties.items():
        metrics_for_domain_masked = {**domain_accuracy}

        id_mask = domain_mask[imagenet_targets]

        for measure_name, ood_measure_values in domain_uncertainties.items():
            id_measure_values = imagenet_uncertainties[measure_name][id_mask]
            metrics_for_domain_masked[measure_name] = compute_ood_auc(id_measure_values, ood_measure_values)

        ood_metrics[domain_name + '_masked'] = metrics_for_domain_masked

        metrics_for_domain = {**domain_accuracy}

        for measure_name, ood_measure_values in domain_uncertainties.items():
            id_measure_values = imagenet_uncertainties[measure_name]
            metrics_for_domain[measure_name] = compute_ood_auc(id_measure_values, ood_measure_values)

        ood_metrics[domain_name] = metrics_for_domain

    return ood_metrics


def get_calibration_errors(full_probs, accuracies, targets, num_bins):
    probs = full_probs.max(axis=1)
    assert len(probs) == len(accuracies)
    calibration_error = 0
    max_calibration_error = 0
    bins = np.linspace(0, 1, num_bins + 1, endpoint=True)
    total_samples = len(probs)

    bin_indices = np.digitize(probs, bins)

    unique_bin_indices = np.unique(bin_indices)

    for bin_ind in unique_bin_indices:
        mask_for_bin, = np.where(bin_indices == bin_ind)
        samples_in_bin = len(mask_for_bin)

        mean_probs = probs[mask_for_bin].mean()
        mean_accuracies = accuracies[mask_for_bin].mean()

        error_for_bin = np.abs(mean_probs - mean_accuracies).item()
        calibration_error += (samples_in_bin / total_samples) * error_for_bin
        max_calibration_error = max(max_calibration_error, error_for_bin)

    onehots = np.zeros_like(full_probs)
    onehots[np.arange(len(probs)), targets] = 1

    brier = np.square(full_probs - onehots).sum(1).mean().item()

    nll = -np.log(full_probs[np.arange(len(probs)), targets]).mean().item()

    return {"ECE": calibration_error, "MCE": max_calibration_error, "brier": brier, "NLL": nll}


def reject_class(labels, preds, measure, measure_name, dataset_name, image_dir):
    inds = np.argsort(measure)

    total_data = preds.shape[0]

    errors = np.cumsum(labels[inds] != preds[inds], dtype=np.float32) * 100 / total_data
    percentages = np.linspace(100 / total_data, 100, num=total_data, endpoint=True, dtype=np.float32)

    base_error = errors[-1]
    n_items = errors.shape[0]
    auc_uns = 1 - auc(percentages / 100, errors[::-1] / 100)

    random_rejection = base_error * np.linspace(1, 1 - (n_items - 1) / n_items, num=n_items, endpoint=True,
                                                dtype=np.float32)
    auc_rnd = 1 - auc(percentages / 100, random_rejection / 100)

    orc_rejection = base_error * np.linspace(1,
                                             1 - (int(base_error / 100 * n_items) - 1) / (base_error / 100 * n_items),
                                             num=int(base_error / 100 * n_items), endpoint=True, dtype=np.float32)
    orc = np.zeros_like(errors)
    orc[0:orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1 - auc(percentages / 100, orc / 100)

    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd)

    random_rejection = np.squeeze(random_rejection)
    orc = np.squeeze(orc)
    errors = np.squeeze(errors)
    if image_dir is not None:
        plot_dir = image_dir / dataset_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.plot(percentages, orc, lw=2)
        plt.fill_between(percentages, orc, random_rejection, alpha=0.5)
        plt.plot(percentages, errors[::-1], lw=2)
        plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.0)
        plt.plot(percentages, random_rejection, 'k--', lw=2)
        plt.legend(['Oracle', 'Uncertainty', 'Random'])
        plt.xlabel('Percentage of predictions rejected to oracle')
        plt.ylabel('Classification Error (%)')
        plt.title(f'{dataset_name}-{measure_name}')
        plt.savefig(image_dir / dataset_name / f'rej-{dataset_name}-{measure_name}-oracle.pdf', format='pdf',
                    bbox_inches='tight', dpi=300)
        plt.close()

        plt.plot(percentages, orc, lw=2)
        plt.fill_between(percentages, orc, random_rejection, alpha=0.0)
        plt.plot(percentages, errors[::-1], lw=2)
        plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.5)
        plt.plot(percentages, random_rejection, 'k--', lw=2)
        plt.legend(['Oracle', 'Uncertainty', 'Random'])
        plt.xlabel('Percentage of predictions rejected to oracle')
        plt.ylabel('Classification Error (%)')
        plt.title(f'{dataset_name}-{measure_name}')
        plt.savefig(image_dir / dataset_name / f'rej-{dataset_name}-{measure_name}-uncertainty.pdf', format='pdf',
                    bbox_inches='tight', dpi=300)
        plt.close()

    return rejection_ratio, auc_uns


def combine_ood_rejection_metrics(imagenet_uncertainties, imagenet_targets, imagenet_preds, ood_uncertainties,
                                  image_dir):
    error_metrics = defaultdict(lambda: dict())

    # figures are saved inside reject_class
    for measure_name, ood_measure_values in imagenet_uncertainties.items():
        rejection_ratio, auc_uns = reject_class(imagenet_targets, imagenet_preds, ood_measure_values, measure_name,
                                                'imagenet_val', image_dir)

        error_metrics['imagenet_val'].update({
            f'{measure_name}_PRR': rejection_ratio,
            f'{measure_name}_PR-AUC': auc_uns,
        })

    for domain_name, (
        (domain_uncertainties, domain_targets, domain_accuracy, domain_preds),
        domain_mask) in ood_uncertainties.items():

        for measure_name, ood_measure_values in domain_uncertainties.items():
            rejection_ratio, auc_uns = reject_class(domain_targets, domain_preds, ood_measure_values, measure_name,
                                                    domain_name, image_dir)
            error_metrics[domain_name].update({
                f'{measure_name}_PRR': rejection_ratio,
                f'{measure_name}_PR-AUC': auc_uns,
            })

            id_measure_values = imagenet_uncertainties[measure_name]
            concat_targets = np.concatenate((imagenet_targets, domain_targets))
            concat_preds = np.concatenate((imagenet_preds, domain_preds))
            concat_measure_values = np.concatenate((id_measure_values, ood_measure_values))
            joint_domain_name = f'{domain_name}_with_val'

            rejection_ratio, auc_uns = reject_class(concat_targets, concat_preds, concat_measure_values, measure_name,
                                                    joint_domain_name, image_dir)

            error_metrics[joint_domain_name].update({
                f'{measure_name}_PRR': rejection_ratio,
                f'{measure_name}_PR-AUC': auc_uns,
            })

    return error_metrics
