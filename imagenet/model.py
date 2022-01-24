import re

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.models import resnet50

from losses import EPS, losses_dict
from mimo_model import mimo_resnet50


@torch.no_grad()
def accuracy(output, target, topk=(1,), reduce=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)

    if target.dim() == 2:
        batch_size, mimo_ensemble_size = target.size()

        _, pred = output.topk(maxk, dim=2, largest=True, sorted=True)

        pred = pred.permute(2, 0, 1)
        correct = pred.eq(target.view(1, batch_size, mimo_ensemble_size).expand_as(pred))

        bsz_for_reduction = batch_size * mimo_ensemble_size
    else:
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        bsz_for_reduction = batch_size

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        if reduce:
            res.append(correct_k.mul_(100.0 / bsz_for_reduction))
        else:
            res.append(correct_k)
    return res


class ImagenetClassifier(LightningModule):
    def __init__(self, learning_rate, weight_decay, epochs):
        super().__init__()
        self.model = resnet50(pretrained=False, zero_init_residual=True)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        bn_regex = re.compile('layer\d.\d.bn\d.(bias|weight)')

        parameter_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if not bn_regex.match(n)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if bn_regex.match(n)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.SGD(parameter_groups,
                                    lr=self.learning_rate,
                                    momentum=0.9, nesterov=True)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(30 * self.epochs) // 90,
                                                                                (60 * self.epochs) // 90,
                                                                                (80 * self.epochs) // 90])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        loss = F.cross_entropy(output, target)
        acc1, = accuracy(output, target, topk=(1,))

        self.log_dict({
            'train_loss': loss,
            'train_acc1': acc1,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        val_loss = F.cross_entropy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log_dict({
            'val_loss': val_loss,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return val_loss, acc1

    def test_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        val_loss = F.cross_entropy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log_dict({
            'val_loss': val_loss,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return val_loss, acc1


@torch.no_grad()
def compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mkl = torch.nn.functional.kl_div(ensemble_logprobs, ensemble_mean_probs.unsqueeze(1).expand_as(ensemble_probs),
                                     reduction='none').sum(-1).mean(1)
    return mkl


@torch.no_grad()
def compute_ensemble_stats(ensemble_logits):
    ensemble_probs = torch.softmax(ensemble_logits, dim=-1)
    ensemble_mean_probs = ensemble_probs.mean(dim=1)
    ensemble_logprobs = torch.log_softmax(ensemble_logits, dim=-1)

    entropy_of_expected = torch.distributions.Categorical(probs=ensemble_mean_probs).entropy()
    expected_entropy = torch.distributions.Categorical(probs=ensemble_probs).entropy().mean(dim=1)
    mutual_info = entropy_of_expected - expected_entropy

    mkl = compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)

    num_classes = ensemble_logits.size(-1)

    ensemble_precision = (num_classes - 1) / (2 * mkl.unsqueeze(1) + EPS)

    stats = {
        'probs': ensemble_probs,
        'mean_probs': ensemble_mean_probs,
        'logprobs': ensemble_logprobs,
        'mkl': mkl,
        'precision': ensemble_precision,
        'entropy_of_expected': entropy_of_expected,
        'mutual_info': mutual_info
    }
    return stats


def compute_correlations(stats, prefix=''):
    for metric in 'mkl', 'mutual_info', 'precision', 'entropy_of_expected':
        metric_for_ensemble = stats.pop(f'{prefix}ensemble_{metric}')
        metric_for_model = stats.pop(f'{prefix}{metric}')
        stats[f'{prefix}ensemble_{metric}_mean'] = metric_for_ensemble.mean()
        stats[f'{prefix}{metric}_mean'] = metric_for_model.mean()


class ImagenetDistilledClassifier(ImagenetClassifier):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--loss', choices=losses_dict.keys(), required=True)
        parser.add_argument('--model_offset', type=float, default=0)
        parser.add_argument('--target_offset', type=float, default=0)
        parser.add_argument('--optimizer', default='sgd')

        return parser

    def __init__(self, ensemble_modules, learning_rate, epochs, weight_decay, loss_type, model_offset, target_offset,
                 optimizer):
        super().__init__(learning_rate, weight_decay, epochs)
        self.model = resnet50(pretrained=False, zero_init_residual=True)
        self.ensemble_modules = ensemble_modules
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_offset = model_offset
        self.target_offset = target_offset
        self.loss_type = loss_type
        self.loss = losses_dict[loss_type]
        self.optimizer = optimizer
        self.save_hyperparameters('learning_rate', 'weight_decay', 'loss_type', 'model_offset', 'target_offset',
                                  'optimizer')

    @torch.no_grad()
    def get_ensemble_preds(self, x):
        ensemble_predictions = [model.to(self.device).eval()(x) for model in self.ensemble_modules]
        ensemble_logits = torch.stack(ensemble_predictions, dim=1).float()

        return x, compute_ensemble_stats(ensemble_logits)

    def training_step(self, batch, batch_idx):
        x, target = batch

        x, ensemble_stats = self.get_ensemble_preds(x)

        output = self(x)

        loss, stats = self.loss(output, ensemble_stats, model_offset=self.model_offset,
                                target_offset=self.target_offset)
        acc1, = accuracy(output, target, topk=(1,))

        if self.loss_type != 'forward_kl':
            stats = {
                'train_loss': loss,
                'train_acc1': acc1,
                'ensemble_precision': ensemble_stats['precision'],
                'ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'ensemble_entropy_of_expected': ensemble_stats['entropy_of_expected'],
                'ensemble_mutual_info': ensemble_stats['mutual_info'],
                'ensemble_mkl': ensemble_stats['mkl'],
                **stats
            }
            compute_correlations(stats)
        else:
            stats = {
                'train_loss': loss,
                'train_acc1': acc1,
                'ensemble_precision_mean': ensemble_stats['precision'].mean(),
                'ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'ensemble_entropy_of_expected_mean': ensemble_stats['entropy_of_expected'].mean(),
                'ensemble_mutual_info_mean': ensemble_stats['mutual_info'].mean(),
                'ensemble_mkl_mean': ensemble_stats['mkl'].mean(),
                **stats
            }

        self.log_dict(stats, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        x, ensemble_stats = self.get_ensemble_preds(x)

        output = self(x)

        loss, stats = self.loss(output, ensemble_stats, model_offset=self.model_offset,
                                target_offset=self.target_offset)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if self.loss_type != 'forward_kl':
            stats = {
                'val_loss': loss,
                'val_acc1': acc1,
                'val_ensemble_precision': ensemble_stats['precision'],
                'val_ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'val_ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'val_ensemble_entropy_of_expected': ensemble_stats['entropy_of_expected'],
                'val_ensemble_mutual_info': ensemble_stats['mutual_info'],
                'val_ensemble_mkl': ensemble_stats['mkl'],
                **{f'val_{key}': value for key, value in stats.items()}
            }
            compute_correlations(stats, prefix='val_')
        else:
            stats = {
                'val_loss': loss,
                'val_acc1': acc1,
                'val_ensemble_precision_mean': ensemble_stats['precision'].mean(),
                'val_ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'val_ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'val_ensemble_entropy_of_expected_mean': ensemble_stats['entropy_of_expected'].mean(),
                'val_ensemble_mutual_info_mean': ensemble_stats['mutual_info'].mean(),
                'val_ensemble_mkl_mean': ensemble_stats['mkl'].mean(),
                **{f'val_{key}': value for key, value in stats.items()}
            }

        self.log_dict(stats)
        return loss, acc1

    def test_step(self, batch, batch_idx):
        x, target = batch

        x, ensemble_stats = self.get_ensemble_preds(x)

        output = self(x)

        loss, stats = self.loss(output, ensemble_stats, model_offset=self.model_offset,
                                target_offset=self.target_offset)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if self.loss_type != 'forward_kl':
            stats = {
                'test_loss': loss,
                'test_acc1': acc1,
                'ensemble_precision': ensemble_stats['precision'],
                'ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'ensemble_entropy_of_expected': ensemble_stats['entropy_of_expected'],
                'ensemble_mutual_info': ensemble_stats['mutual_info'],
                'ensemble_mkl': ensemble_stats['mkl'],
                **stats
            }
            compute_correlations(stats)
        else:
            stats = {
                'test_loss': loss,
                'test_acc1': acc1,
                'ensemble_precision_mean': ensemble_stats['precision'].mean(),
                'ensemble_prob_max': ensemble_stats['mean_probs'].max(),
                'ensemble_prob_min': ensemble_stats['mean_probs'].min(),
                'ensemble_entropy_of_expected_mean': ensemble_stats['entropy_of_expected'].mean(),
                'ensemble_mutual_info_mean': ensemble_stats['mutual_info'].mean(),
                'ensemble_mkl_mean': ensemble_stats['mkl'].mean(),
                **stats
            }

        self.log_dict(stats)
        return loss, acc1

    def configure_optimizers(self):
        bn_regex = re.compile('layer\d.\d.bn\d.(bias|weight)')

        parameter_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if not bn_regex.match(n)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if bn_regex.match(n)],
                'weight_decay': 0.0,
            },
        ]

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameter_groups,
                                        lr=self.learning_rate,
                                        momentum=0.9, nesterov=True)

            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(30 * self.epochs) // 90,
                                                                                    (60 * self.epochs) // 90,
                                                                                    (80 * self.epochs) // 90])
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        else:
            optimizer = torch.optim.AdamW(parameter_groups, lr=self.learning_rate, eps=1e-6)
            return optimizer


class ImagenetMIMOClassifier(ImagenetClassifier):
    def __init__(self, learning_rate, weight_decay, epochs, ensemble_size, repetition_probability, batch_repetitions):
        super().__init__(learning_rate, weight_decay, epochs)
        self.model = mimo_resnet50(ensemble_size=ensemble_size, zero_init_residual=True)
        self.ensemble_size = ensemble_size
        self.repetition_probability = repetition_probability
        self.batch_repetitions = batch_repetitions

        self.save_hyperparameters()

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--ensemble_size', type=int, default=2)
        parser.add_argument('--repetition_probability', type=float, default=0.6)
        parser.add_argument('--batch_repetitions', type=int, default=2)

        return parser

    def training_step(self, batch, batch_idx):
        x, target = batch

        main_shuffle = torch.tile(
            torch.arange(x.size(0), device=x.device),
            (self.batch_repetitions,)
        )[torch.randperm(x.size(0) * self.batch_repetitions, device=x.device)]

        to_shuffle = int(main_shuffle.size(0) * (1 - self.repetition_probability))

        shuffle_indices = [
            torch.cat([main_shuffle[:to_shuffle][torch.randperm(to_shuffle, device=x.device)],
                       main_shuffle[to_shuffle:]], axis=0)
            for _ in range(self.ensemble_size)]

        x = torch.stack([x[indices] for indices in shuffle_indices], dim=1)

        target = torch.stack([target[indices] for indices in shuffle_indices], dim=1)

        output = self(x)

        loss = F.cross_entropy(output.permute(0, 2, 1), target)
        acc1, = accuracy(output, target, topk=(1,))

        stats = {
            'train_loss': loss,
            'train_acc1': acc1,
        }

        self.log_dict(stats, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        x = torch.tile(torch.unsqueeze(x, dim=1), (1, self.ensemble_size, 1, 1, 1))

        output = self(x).mean(1)

        loss = F.cross_entropy(output, target)
        acc1, = accuracy(output, target, topk=(1,))

        stats = {
            'val_loss': loss,
            'val_acc1': acc1,
        }

        self.log_dict(stats)
        return loss, acc1

    def test_step(self, batch, batch_idx):
        x, target = batch

        x = torch.tile(torch.unsqueeze(x, dim=1), (1, self.ensemble_size, 1, 1, 1))

        output = self(x).mean(1)

        loss = F.cross_entropy(output, target)
        acc1, = accuracy(output, target, topk=(1,))

        stats = {
            'test_loss': loss,
            'test_acc1': acc1,
        }

        self.log_dict(stats)
        return loss, acc1
