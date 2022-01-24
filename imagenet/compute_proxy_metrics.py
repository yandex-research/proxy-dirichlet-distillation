import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data_utils import LocalImagenetTrainDataModule, get_ood_loaders, imagenet_val_mask
from evaluation import get_domain_uncertainties, proxy_predictions
from model import ImagenetClassifier
from uncertainty import combine_ood_metrics, combine_ood_rejection_metrics

ch = ModelCheckpoint()


def main(args):
    assert os.path.exists(args.ood_data_path)

    ensemble = []
    for path in args.ensemble_paths:
        model = ImagenetClassifier.load_from_checkpoint(path, learning_rate=0.1, weight_decay=1e-4,
                                                        epochs=90).to_torchscript()
        model.cuda()
        model.eval()
        ensemble.append(model)

    imagenet_dm = LocalImagenetTrainDataModule(args.data_dir, batch_size=args.batch_size)

    prediction_func = partial(proxy_predictions, ensemble=ensemble)

    print('Getting per-domain uncertainties:')
    imagenet_uncertainties, imagenet_targets, imagenet_accuracy, imagenet_preds = get_domain_uncertainties(
        prediction_func,
        'imagenet',
        imagenet_dm.test_dataloader(),
        imagenet_val_mask)

    out_of_domain_loaders = get_ood_loaders(args.ood_data_path, args.batch_size)

    ood_uncertainties = {
        domain_name: (get_domain_uncertainties(prediction_func, domain_name, loader, domain_mask), domain_mask)
        for domain_name, (loader, domain_mask) in out_of_domain_loaders.items()}

    print('Aggregating error metrics:')
    error_detection_metrics = combine_ood_rejection_metrics(imagenet_uncertainties, imagenet_targets, imagenet_preds,
                                                            ood_uncertainties, args.image_output)

    print('Aggregating OOD metrics:')
    ood_detection_metrics = combine_ood_metrics(imagenet_uncertainties, imagenet_targets, ood_uncertainties)

    print('Saving results:')
    metrics = defaultdict(lambda: dict())

    for dataset, metrics_for_dataset in error_detection_metrics.items():
        metrics[dataset].update(metrics_for_dataset)

    for dataset, metrics_for_dataset in ood_detection_metrics.items():
        metrics[dataset].update(metrics_for_dataset)

    metrics['imagenet_val'].update(imagenet_accuracy)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.json_output is not None:
        with open(args.json_output, 'w+') as output_f:
            json.dump(metrics, output_f, indent=2, sort_keys=True)
            output_f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--ensemble_paths', nargs='+', required=True)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--ood_data_path', '-p', type=Path, required=True)
    parser.add_argument('--json_output', '-j', type=Path)
    parser.add_argument('--image_output', '-i', type=Path)
    parser = Trainer.add_argparse_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
