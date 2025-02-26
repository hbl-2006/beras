import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        total_numbers = len(unique_labels)
        one_hot_encoded = np.eye(total_numbers)

        self.label_to_one_hot = {label: one_hot_encoded[i] for i, label in enumerate(unique_labels)}
        self.one_hot_to_label = {tuple(one_hot_encoded[i]): label for i, label in enumerate(unique_labels)}

    def forward(self, data):
        return np.array([self.label_to_one_hot[label] for label in data])

    def inverse(self, data):
        return np.array([self.one_hot_to_label[tuple(one_hot)] for one_hot in data])
