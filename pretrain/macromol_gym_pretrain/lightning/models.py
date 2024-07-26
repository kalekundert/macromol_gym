import lightning as L

from ..torch import NeighborLocationPredictor, ViewPairEncoder
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

class PredictorModule(L.LightningModule):

    def __init__(self, model, opt_factory, num_classes=6):
        super().__init__()
        self.save_hyperparameters(ignore='model')

        self.model = model
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.optimizer = opt_factory(model.parameters())

    @classmethod
    def from_encoder(cls, view_encoder, pair_classifier, opt_factory):
        pair_encoder = ViewPairEncoder(view_encoder)
        model = NeighborLocationPredictor(
                pair_encoder=pair_encoder,
                pair_classifier=pair_classifier,
        )
        return cls(model, opt_factory)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/loss": 0})

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return y_hat, loss, acc

    def training_step(self, batch, _):
        _, loss, acc = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        _, loss, acc = self.forward(batch)
        self.log('val/loss', loss)
        self.log('val/accuracy', acc)
        return loss

    def test_step(self, batch, _):
        _, loss, acc = self.forward(batch)
        self.log('test/loss', loss)
        self.log('test/accuracy', acc)
        return loss

