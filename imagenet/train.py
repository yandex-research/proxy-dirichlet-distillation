import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils import LocalImagenetTrainDataModule
from model import ImagenetClassifier


def main(args):
    seed_everything(args.seed)

    actual_learning_rate = args.learning_rate * args.batch_size * args.gpus / 256

    pl_module = ImagenetClassifier(actual_learning_rate, args.weight_decay, args.max_epochs)
    dm = LocalImagenetTrainDataModule(args.data_dir, batch_size=args.batch_size)

    logger = TensorBoardLogger(save_dir=args.logdir, version=args.version, name='')
    checkpoint_callback = ModelCheckpoint(dirpath=args.weights_save_path, monitor='val_acc1', mode='max',
                                          filename='{epoch:02d}', save_last=True)
    lr_monitor = LearningRateMonitor()

    if not os.path.exists(args.resume_from_checkpoint):
        print('Checkpoint not found')
        args.resume_from_checkpoint = None
    else:
        print('Resuming from checkpoint')

    trainer = Trainer.from_argparse_args(args, logger=logger, resume_from_checkpoint=args.resume_from_checkpoint,
                                         callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(pl_module, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--learning_rate', '--lr', type=float, required=True)
    parser.add_argument('--weight_decay', '--wd', type=float, required=True)
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
