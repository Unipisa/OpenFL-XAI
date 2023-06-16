# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""XAITaskRunned module."""

from logging import getLogger
from importlib import import_module


class XAITaskRunner(object):
    """Federated Learning Task Runner Class for XAI Models."""

    def __init__(self, data_loader,
                 dict_parameters: dict,
                 dict_model: dict,
                 tensor_dict_split_fn_kwargs: dict = None,
                 **kwargs):
        """
        Intialize.

        Args:
            data_loader:  XAIDataLoader object
            dict_parameters: hyperparameters needed by the XAI Model
            dict_model: path and class name of the XAI Model
            tensor_dict_split_fn_kwargs: (Default=None)
            **kwargs: Additional parameters to pass to the function

        """
        self.data_loader = data_loader

        if tensor_dict_split_fn_kwargs is None:
            tensor_dict_split_fn_kwargs = {}
        self.tensor_dict_split_fn_kwargs = tensor_dict_split_fn_kwargs
        self.set_logger()

        self.model = None
        self.trained_model = None

        self.required_tensorkeys_for_function = {}
        self.dict_prova = dict_parameters

        self.model_class_name = dict_model["model_class_name"]
        self.model_module_path = import_module(dict_model["model_module_path"])
        self.model = getattr(self.model_module_path, self.model_class_name)(dict_parameters)

    def train(self, col_name, round_num, **kwargs):
        '''
            define training logic of the model

            It needs to produce antecedents, consequents and weights, then terminate with snippet:
                
                Rules = namedtuple('Rules', ['name', 'value'])
                rules = []
                rules.append(Rules(name="antecedents", value=antecedents))
                # rules.append(Rules(name="num_conflicting_antecedents", value=num_conflicting_antecedents))
                rules.append(Rules(name="consequents", value=consequents))
                # rules.append(Rules(name="num_training_samples", value=num_training_samples))
                rules.append(Rules(name="rule_weights", value=rule_weights))

                save_xai_model(self, antecedents, consequents, rule_weights)

                origin = col_name
                output_rules_dict = {
                    TensorKey(
                        rule_set_name, origin, round_num, True, ('rule',)
                    ): rule_set_value
                    for (rule_set_name, rule_set_value) in rules
                }
                
                return output_rules_dict
                
            
        '''
        """
        Generate a fuzzy-system
        :param variable_names: the name of the variables used in the antecedents
        :param num_rules: the number of rules of the system. Used by the fuzzy c-means
        :param order_consequent: the order of the consequent ['zero', 'first']
        Returns
        -------
        dict
            'TensorKey: nparray'
        """

        output_rules_dict = self.model.train(col_name, round_num, self.data_loader)
        print("-----------------------------------------------------------------------------")
        print(f'Local Training done. Output rules dict: \n{output_rules_dict}')
        print("-----------------------------------------------------------------------------")

        return output_rules_dict

    def set_logger(self):
        """Set up the log object."""
        self.logger = getLogger(__name__)

    def get_data_loader(self):
        """
        Get the data_loader object.

        Serves up batches and provides info regarding data_loader.

        Returns:
            data_loader object
        """
        return self.data_loader

    def set_data_loader(self, data_loader):
        """Set data_loader object.

        Args:
            data_loader: data_loader object to set
        Returns:
            None
        """

        self.data_loader = data_loader

    def get_train_data_size(self):
        """
        Get the number of training examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of training examples.
        """
        return self.data_loader.get_train_data_size()
