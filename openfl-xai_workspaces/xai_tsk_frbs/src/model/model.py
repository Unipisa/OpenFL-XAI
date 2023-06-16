# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def __init__(self, parameters_dict: dict, **kwars):
        """
            parameters_dict ( dict ) :dictionary defined in the plan.yaml file.
                Inside are specified the initialization parameters for the model.
        """
        pass

    @abstractmethod
    def train(self, col_name, round_num, data_loader, **kwargs):
        """
            col_name : collaborator name
            round_num : current round
            data_loader : XAIDataLoader, class that extends openfl.federated.data.loader. 
                Used to load the data.
             
            IF want to save the LOCAL model.
                must invoke the save_xai_model before returning
            
            Output: dict
                The col_name and round_num information must be included into the TensorKeys.
                An example from a use case:
                    Rules = namedtuple('Rules', ['name', 'value'])
                    rules = []
                    rules.append(Rules(name="antecedents", value=antecedents))
                    rules.append(Rules(name="consequents", value=consequents))
                    rules.append(Rules(name="rule_weights", value=rule_weights))

                    origin = col_name
                    output_rules_dict = {
                        TensorKey(
                            rule_set_name, origin, round_num, True, ('rule',)
                        ): rule_set_value
                        for (rule_set_name, rule_set_value) in rules
                    }

        """
        pass

    @abstractmethod
    def save_xai_model(self):
        """
            defines save LOCAL model logic ( e.g. store on local file system, send through Restful api, ...)
            
            Can be left empty if there is no need to save local models on Collaborators

        """
        pass
