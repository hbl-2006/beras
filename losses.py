import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean(np.mean((y_true - y_pred) ** 2, axis=-1))

    def get_input_gradients(self) -> list[Tensor]:
        batch_size = self.inputs[0].shape[0]
        num_features = self.inputs[0].shape[1]
        grad_y_pred = 2 * (self.inputs[0] - self.inputs[1]) / (batch_size * num_features)
        grad_y_true = np.zeros_like(self.inputs[1])
        return [grad_y_pred, grad_y_true]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=-1), axis=0)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        batch_size = self.inputs[0].shape[0]
        grad_y_pred = (-self.inputs[1] / np.clip(self.inputs[0], 1e-7, 1 - 1e-7)) / batch_size
        grad_y_true = np.zeros_like(self.inputs[1])
        return [grad_y_pred, grad_y_true]
