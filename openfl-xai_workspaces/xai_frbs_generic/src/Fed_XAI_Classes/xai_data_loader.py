# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""XAIDataLoader module. (Extends the DataLoader base class)"""

import numpy as np
from openfl.federated.data.loader import DataLoader


class XAIDataLoader(DataLoader):
    """
    Data Loader for in memory Numpy data.

    """

    def __init__(self, data_path, num_classes=None):
        """
        Initialize the training data from two numpy files named:
            - X_train.npy
            - y_train.npy

        Args:
            data_path: path to the numpy files folder.

            **kwargs: Additional arguments to pass to the function
        """
        super().__init__
        print("-------------------------------------------------------------------------------------------------------")
        X_train = np.load(data_path + '/X_train.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                          encoding='ASCII')
        y_train = np.load(data_path + '/y_train.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                          encoding='ASCII')

        print(f'Training data loaded. Shape is {X_train.shape}')
        print(
            '--------------------------------------------------------------------------------------------------------')

        self.X_train = X_train
        self.y_train = y_train

        if num_classes is None:
            num_classes = np.unique(self.y_train).shape[0]
            print(f'Inferred {num_classes} classes from the provided labels...')

        self.num_classes = num_classes

    def get_train_data(self):
        """
        Get training data.

        Returns
        -------
            numpy.ndarray: training set
        """
        return self.X_train

    def get_train_labels(self):
        """
        Get training data labels.

        Returns
        -------
            numpy.ndarray: training set labels 
        """
        return self.y_train

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return len(self.X_train)
