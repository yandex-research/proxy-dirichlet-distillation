import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data_utils import LocalImagenetTrainDataModule, get_ood_loaders, imagenet_val_mask
from evaluation import get_domain_uncertainties, single_model_predictions
from model import ImagenetClassifier
from uncertainty import combine_ood_metrics, combine_ood_rejection_metrics

ch = ModelCheckpoint()


def main(args):
    assert os.path.exists(args.resume_from_checkpoint)
    assert os.path.exists(args.ood_data_path)
    model = ImagenetClassifier.load_from_checkpoint(args.resume_from_checkpoint, learning_rate=0.1, weight_decay=1e-4,
                                                    epochs=90, strict=False).to_torchscript()

    if hasattr(model, 'model_offset'):
        print(model.model_offset)

    model.cuda()
    model.eval()

    imagenet_dm = LocalImagenetTrainDataModule(args.data_dir, batch_size=args.batch_size)

    prediction_func = partial(single_model_predictions, model=model, offset=args.model_offset)

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
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--ood_data_path', '-p', type=Path, required=True)
    parser.add_argument('--model-offset', type=float, default=0)
    parser.add_argument('--json_output', '-j', type=Path)
    parser.add_argument('--image_output', '-i', type=Path)
    parser = Trainer.add_argparse_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
