import satlaspretrain_models
from torch import Tensor

from torchgeo.trainers import SemanticSegmentationTask


class SatlasSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, model_identifier: str, fpn: bool = True, pretrained: bool = True, *args, **kwargs):
        self.weights_manager = satlaspretrain_models.Weights()
        self.model_identifier = model_identifier
        self.pretrained = pretrained
        self.fpn = fpn
        super().__init__(*args, **kwargs)

    def configure_models(self):
        self.model = self.weights_manager.get_pretrained_model(
            model_identifier=self.model_identifier,
            fpn=self.fpn,
            head=satlaspretrain_models.Head.SEGMENT,
            num_categories=self.hparams["num_classes"]
        )

        if self.hparams["freeze_backbone"]:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat, loss = self(x)  # don't really use loss computed by satlaspretrain package
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss, batch_size=batch_size)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat, loss = self(x)  # don't really use loss computed by satlaspretrain package
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=batch_size)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)
