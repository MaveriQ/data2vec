from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from torch.optim import Adam

from transformers import BertModel, BertConfig

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.models.self_supervised.byol.models import SiameseArm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Data2Vec(LightningModule):
    """PyTorch Lightning implementation of Data2Vec by Meta AI.
    
    Codebase extended from BYOL implementaion of `Annika Brundyn <https://github.com/annikabrundyn>` \
    found at https://github.com/PyTorchLightning/lightning-bolts/tree/master/pl_bolts/models/self_supervised/byol

    Model implemented by:
        - `Haris Jabbar <https://github.com/maveriq>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - Implement data augmentation pipeline
        - Verify Implementation
        - Implement selectable top K layers instead of all (current)

    Example::

        model = Data2Vec()

        dm = ///tobeimplemented...

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

    CLI command::


    .. _data2vec: https://arxiv.org/abs/2202.03555
    """

    def __init__(
        self,
        num_classes,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        # base_encoder: Union[str, torch.nn.Module] = "resnet50",
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
            base_encoder: the base encoder module or resnet name
            encoder_out_dim: output dimension of base_encoder
            projector_hidden_size: hidden layer size of projector MLP
            projector_out_dim: output size of projector MLP
        """
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")
        
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states=True

        self.teacher_network = BertModel(config,)
        self.student_network = deepcopy(self.teacher_network)
        self.weight_callback = BYOLMAWeightUpdate()
        self.loss_fn = torch.nn.MSELoss()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        output = self.online_network(x)
        return output

    def shared_step(self, batch, batch_idx):
        raw_input, masked_input = batch

        # Image 1 to image 2 loss
        output_student = self.student_network(masked_input)
        with torch.no_grad():
            output_teacher = self.teacher_network(raw_input)

        embed_student = torch.cat(output_student.hidden_states,0).mean(0)
        embed_teacher = torch.cat(output_teacher.hidden_states,0).mean(0)
        # Final loss
        loss = self.loss_fn(embed_student, embed_teacher)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log({"valid_loss": loss})

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--online_ft", action="store_true", help="run online finetuner")
        parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet2012", "stl10"])

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--num_workers", default=8, type=int)

        # optim
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1.5e-6)
        parser.add_argument("--warmup_epochs", type=float, default=10)

        # Model
        parser.add_argument("--meta_dir", default=".", type=str, help="path to meta.bin for imagenet")

        return parser