import math
from dataclasses import dataclass, field

import torch
from omegaconf import MISSING, II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass, ChoiceEnum

EPS = torch.finfo(torch.float32).eps


def compute_mean_forward_kl(
    model, sample, ensemble_stats, net_output, ignore_index, reduce
):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    target = model.get_targets(sample, net_output)

    teacher_probs = ensemble_stats["logprobs"]

    # average loss over all teacher distributions
    lprobs = lprobs.unsqueeze(2).expand_as(teacher_probs)
    loss = (
        torch.nn.functional.kl_div(lprobs, teacher_probs, reduction="none")
            .mean(2)
            .sum(-1)
    )

    # mask loss for padding tokens
    pad_mask = target.eq(ignore_index)
    loss.masked_fill_(pad_mask, 0.0)

    if reduce:
        return torch.sum(loss)
    return loss


@torch.no_grad()
def compute_epkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mlog_probs = torch.mean(ensemble_logprobs, dim=2)
    eoe_upper_bound = -torch.sum(ensemble_mean_probs * mlog_probs, dim=-1)

    exe = -torch.mean(torch.sum(ensemble_probs * ensemble_logprobs, dim=-1), dim=2)
    epkl = eoe_upper_bound - exe
    return epkl


@torch.no_grad()
def compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mkl = (
        torch.nn.functional.kl_div(
            ensemble_logprobs,
            ensemble_mean_probs.unsqueeze(2).expand_as(ensemble_probs),
            reduction="none",
        )
            .sum(3)
            .mean(2)
    )
    return mkl


@torch.no_grad()
def compute_mutual_information(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    exe = -torch.mean(torch.sum(ensemble_probs * ensemble_logprobs, dim=-1), dim=2)
    log_mprobs = torch.log(ensemble_mean_probs)
    eoe = -torch.sum(ensemble_mean_probs * log_mprobs, dim=-1)
    mutual_info = eoe - exe
    return mutual_info


@torch.no_grad()
def compute_ensemble_stats(sample, target_concentration):
    """
    Return a dictionary with ensemble predictions' statistics and uncertainty measures.

    Arguments:
        sample: a dictionary with training batch source/target data as keys, see fairseq.data.language_pair_dataset
        target_concentration: whether to estimate precision from ensemble EPKL ('epkl') or MKL ('mkl')
    """
    ensemble_logits = sample["ensemble_logits"]
    ensemble_probs = utils.softmax(ensemble_logits, dim=-1)
    ensemble_mean_probs = ensemble_probs.mean(dim=2)
    ensemble_logprobs = utils.log_softmax(ensemble_logits, dim=-1)

    epkl = compute_epkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)
    mkl = compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)
    mutual_info = compute_mutual_information(
        ensemble_probs, ensemble_mean_probs, ensemble_logprobs
    )

    num_classes = ensemble_logits.size(-1)

    if target_concentration == "mkl":
        ensemble_precision = (num_classes - 1) / (2 * mkl.unsqueeze(2) + EPS)
    elif target_concentration == "epkl":
        ensemble_precision = (num_classes - 1) / (epkl.unsqueeze(2) + EPS)
    else:
        raise ValueError

    stats = {
        "probs": ensemble_probs,
        "mean_probs": ensemble_mean_probs,
        "logprobs": ensemble_logprobs,
        "epkl": epkl,
        "mkl": mkl,
        "mutual_info": mutual_info,
        "precision": ensemble_precision,
    }
    return stats


def compute_nll(model, sample, net_output, reduce, ignore_index):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    target = model.get_targets(sample, net_output)
    smoothed_loss, nll_loss = label_smoothed_nll_loss(
        lprobs.view(-1, lprobs.size(-1)),
        target.view(-1, 1),
        epsilon=0,
        ignore_index=ignore_index,
        reduce=reduce,
    )
    return nll_loss


TARGET_CONCENTRATION_CHOICES = ChoiceEnum(["mkl", "epkl"])


@dataclass
class DistillationCriterionBaseConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    target_concentration: TARGET_CONCENTRATION_CHOICES = field(default=MISSING)
    sentence_avg: bool = II("optimization.sentence_avg")


class _DistillationCriterionBase(FairseqCriterion):
    """
    An abstract interface which must be subclassed by all criteria for distillation.
    """

    def __init__(
        self, task, label_smoothing, target_concentration, sentence_avg
    ):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.target_concentration = target_concentration
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        # batch x len x n_tokens
        net_output = model(**sample["net_input"])

        ensemble_stats = compute_ensemble_stats(
            sample, self.target_concentration
        )

        loss, stats = self.compute_loss(
            model, net_output, ensemble_stats, sample, reduce=reduce
        )

        nll_loss = compute_nll(model, sample, net_output, reduce, self.padding_idx)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "ensemble_mkl": utils.item(ensemble_stats["mkl"].sum()),
            "ensemble_epkl": utils.item(ensemble_stats["epkl"].sum()),
            "ensemble_precision": utils.item(ensemble_stats["precision"].sum()),
            **stats,
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss",
            sum(log.get("loss", 0) for log in logging_outputs) / sample_size,
            sample_size,
            round=3,
        )

        metrics.log_scalar(
            "nll_loss",
            sum(log.get("nll_loss", 0) for log in logging_outputs)
            / ntokens
            / math.log(2),
            ntokens,
            round=3,
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ensemble_mkl",
            sum(log.get("ensemble_mkl", 0) for log in logging_outputs) / sample_size,
            sample_size,
            round=3,
        )

        metrics.log_scalar(
            "ensemble_epkl",
            sum(log.get("ensemble_epkl", 0) for log in logging_outputs) / sample_size,
            sample_size,
            round=3,
        )

        metrics.log_scalar(
            "ensemble_precision",
            sum(log.get("ensemble_precision", 0) for log in logging_outputs)
            / sample_size,
            sample_size,
            round=3,
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


@register_criterion(
    "forward_kl_mean_distillation", dataclass=DistillationCriterionBaseConfig
)
class ForwardKLMeanCritertion(_DistillationCriterionBase):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        avg_teacher_probs = ensemble_stats["mean_probs"]

        loss = torch.nn.functional.kl_div(
            log_probs, avg_teacher_probs, reduction="none"
        ).sum(-1)

        # mask loss for padding tokens
        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            return torch.sum(loss), dict()
        return loss, dict()


@register_criterion(
    "sequence_distribution_distillation",
    dataclass=DistillationCriterionBaseConfig,
)
class SequenceDistributionDistillationCritertion(_DistillationCriterionBase):
    def __init__(
        self,
        task,
        label_smoothing,
        target_concentration,
        sentence_avg,
    ):
        super().__init__(
            task, label_smoothing, target_concentration, sentence_avg
        )
        self.model_offset = task.model_offset

    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        from examples.ensemble_distillation.utils import get_dirichlet_parameters

        alphas, precision = get_dirichlet_parameters(
            net_output,
            add_to_alphas=self.model_offset,
        )

        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        precision_sum = precision.masked_fill(pad_mask, 0).sum()

        stats = {
            "precision": precision_sum,
        }

        teacher_probs = ensemble_stats["probs"]
        mean_teacher_probs = ensemble_stats["mean_probs"]

        log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs + EPS), dim=-2)

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = torch.sum(
            torch.lgamma(alphas + EPS), dim=-1
        ) - torch.lgamma(precision + EPS)
        target_dependent_term = -torch.sum(
            (alphas - 1.0) * log_teacher_probs_geo_mean, dim=-1
        )
        cost = target_dependent_term + target_independent_term

        # mask loss for padding tokens
        cost.masked_fill_(pad_mask, 0.0)

        if reduce:
            return torch.sum(cost), stats
        return cost, stats

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        _DistillationCriterionBase.reduce_metrics(logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        number_of_outputs = sum(
            1 if log.get("precision") is not None else 0 for log in logging_outputs
        )

        metrics.log_scalar(
            "precision",
            sum(log.get("precision", 0) for log in logging_outputs) / ntokens,
            weight=ntokens,
            round=3
        )

        for metric in ("entropy", "epkl", "mkl", "mutual_info", "precision"):
            metrics.log_scalar(
                f"{metric}_spearman",
                sum(log.get(f"{metric}_spearman", 0) for log in logging_outputs)
                / number_of_outputs,
                number_of_outputs,
                round=3
            )
            metrics.log_scalar(
                f"seq_{metric}_spearman",
                sum(log.get(f"seq_{metric}_spearman", 0) for log in logging_outputs)
                / nsentences,
                nsentences,
                round=3
            )


@dataclass
class DirichletProxyDistillationCriterionConfig(
    DistillationCriterionBaseConfig
):
    target_offset: float = field(default=0)


@register_criterion(
    "dirichlet_proxy_distillation",
    dataclass=DirichletProxyDistillationCriterionConfig,
)
class ProxyDirichletDistillationCriterion(
    SequenceDistributionDistillationCritertion
):
    def __init__(
        self,
        task,
        label_smoothing,
        target_concentration,
        sentence_avg,
        target_offset,
    ):
        super().__init__(
            task, label_smoothing, target_concentration, sentence_avg
        )
        self.target_offset = target_offset

    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        from examples.ensemble_distillation.utils import get_dirichlet_parameters

        alphas, precision = get_dirichlet_parameters(
            net_output, self.model_offset
        )

        num_classes = alphas.size(-1)

        ensemble_precision = ensemble_stats["precision"]

        ensemble_params = (
            ensemble_stats["mean_probs"] * ensemble_precision + self.target_offset
        )
        ensemble_precision += self.target_offset * num_classes

        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        precision_sum = precision.masked_fill(pad_mask, 0).sum()

        stats = {
            "precision": precision_sum,
        }

        target_independent_term = (
            torch.lgamma(ensemble_precision.squeeze(2))
            - torch.sum(torch.lgamma(ensemble_params), dim=-1)
            + torch.sum(torch.lgamma(alphas), dim=-1)
            - torch.lgamma(precision)
        )

        target_dependent_term = torch.sum(
            (ensemble_params - alphas)
            * (torch.digamma(ensemble_params) - torch.digamma(ensemble_precision)),
            dim=-1,
        )

        cost = target_dependent_term + target_independent_term
        # mask loss for padding tokens
        cost.masked_fill_(pad_mask, 0.0)

        if reduce:
            return torch.sum(cost), stats
        return cost, stats


@register_criterion(
    "rkl_dirichlet_proxy_distillation",
    dataclass=DirichletProxyDistillationCriterionConfig,
)
class RKLProxyDirichletDistillationCriterion(ProxyDirichletDistillationCriterion):
    def compute_loss(self, model, net_output, ensemble_stats, sample, reduce):
        from examples.ensemble_distillation.utils import get_dirichlet_parameters

        assert torch.isfinite(net_output[0]).all(), net_output
        alphas, precision = get_dirichlet_parameters(
            net_output,
            self.model_offset,
            dtype=torch.double,
        )
        assert torch.isfinite(alphas).all(), alphas
        assert torch.isfinite(precision).all(), precision

        num_classes = alphas.size(-1)

        ensemble_precision = ensemble_stats["precision"]

        ensemble_precision += self.target_offset * num_classes

        pad_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        precision_sum = precision.masked_fill(pad_mask, 0).sum()

        stats = {
            "precision": precision_sum,
        }

        ensemble_probs = ensemble_stats["mean_probs"]

        expected_KL_term = -1.0 * torch.sum(
            ensemble_probs
            * (
                torch.digamma(alphas + EPS)
                - torch.digamma(precision.unsqueeze(-1) + EPS)
            ),
            dim=-1,
        )
        assert torch.isfinite(expected_KL_term).all(), expected_KL_term

        differential_negentropy_term = (
            torch.sum(torch.lgamma(alphas + EPS), dim=-1)
            - torch.lgamma(precision + EPS)
            - torch.sum(
            (alphas - 1)
            * (
                torch.digamma(alphas + EPS)
                - torch.digamma(precision.unsqueeze(-1) + EPS)
            ),
            dim=-1,
        )
        )
        assert torch.isfinite(
            differential_negentropy_term
        ).all(), differential_negentropy_term

        cost = expected_KL_term - differential_negentropy_term * (
            1.0 / (ensemble_precision.squeeze(-1) + EPS) + 1e-11
        )
        assert torch.isfinite(cost).all(), cost

        # mask loss for padding tokens
        cost.masked_fill_(pad_mask, 0.0)

        if reduce:
            sum_cost = torch.sum(cost)
            assert torch.isfinite(sum_cost).all(), sum_cost
            return sum_cost, stats
        return cost, stats
