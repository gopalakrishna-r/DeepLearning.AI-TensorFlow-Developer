from common.HypeParams import HyperParameters
from

class Trainer(HyperParameters):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self, visualizer, max_epochs=100, num_gpus=0):
        self.save_hyperparameters()
        self.visualizer = visualizer
        assert num_gpus == 0, "No GPU support yet"

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model):
        model.trainer = self
        self.model = model
        self.model.compile(
            optimizer=model.configure_optimizers(),
            loss=model.configure_loss(),
            metrics=["accuracy"],
        )

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.history = self.model.fit(
            x=self.train_dataloader,
            validation_data=self.val_dataloader,
            epochs=self.max_epochs,
            callbacks=[
                self.model.configure_callbacks(
                    self.visualizer,
                    self.max_epochs,
                    self.num_train_batches,
                    self.num_val_batches,
                ),
                WandbCallback(monitor='val_loss', mode='min')
            ],
            verbose=1,
            workers=16,
            use_multiprocessing=True,
        )
