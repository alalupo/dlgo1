import tensorflow as tf

keras = tf.keras


class DebugCallback(keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_inputs = model.inputs
        self.model_outputs = model.outputs

    def on_train_batch_end(self, batch, logs=None):
        model = self.model
        if model is None:
            return ValueError(f'Model empty')
        print(f'')
        print(f"batch={batch}")
        print(f'logs={logs}')
        print(f'log keys: {logs.keys()}')
        print(f'DebugCallback: model inputs: {self.model.inputs}')
        print(f'')
