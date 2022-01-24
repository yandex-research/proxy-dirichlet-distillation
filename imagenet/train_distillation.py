import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_info

from data_utils import LocalImagenetTrainDataModule
from model import ImagenetClassifier, ImagenetDistilledClassifier


def main(args):
    seed_everything(args.seed)

    ensemble = []
    for path in args.ensemble_paths:
        model = ImagenetClassifier.load_from_checkpoint(path, learning_rate=0.1, weight_decay=1e-4,
                                                        epochs=args.max_epochs).to_torchscript()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        ensemble.append(model)

    actual_learning_rate = args.learning_rate * args.batch_size * args.gpus / 256

    pl_module = ImagenetDistilledClassifier(ensemble, actual_learning_rate, args.max_epochs, args.weight_decay,
                                            args.loss, args.model_offset, args.target_offset, args.optimizer)
    dm = LocalImagenetTrainDataModule(args.data_dir, batch_size=args.batch_size)

    logger = TensorBoardLogger(save_dir=args.logdir, version=args.version, name='')
    checkpoint_callback = ModelCheckpoint(dirpath=args.weights_save_path, monitor='val_acc1', mode='max',
                                          filename='{epoch:02d}', save_last=True)
    lr_monitor = LearningRateMonitor()

    if args.resume_from_checkpoint is None or not os.path.exists(args.resume_from_checkpoint):
        rank_zero_info('Checkpoint not found')
        args.resume_from_checkpoint = None
    else:
        rank_zero_info('Resuming from checkpoint')

    plugins = [DDPPlugin(find_unused_parameters=False)] if args.accelerator == 'ddp' else []

    trainer = Trainer.from_argparse_args(args, logger=logger, resume_from_checkpoint=args.resume_from_checkpoint,
                                         callbacks=[checkpoint_callback, lr_monitor], plugins=plugins)
    trainer.fit(pl_module, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--learning_rate', '--lr', type=float, required=True)
    parser.add_argument('--weight_decay', '--wd', type=float, required=True)
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--ensemble_paths', nargs='+', required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--use_randaugment', action='store_true')
    parser.add_argument('--randaugment_magnitude', '-m', type=float)
    parser = Trainer.add_argparse_args(parser)
    parser = ImagenetDistilledClassifier.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
