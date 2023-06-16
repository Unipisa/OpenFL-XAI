# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""XAI rules aggregation logic module."""

import numpy as np
from openfl.component.aggregation_functions.interface import AggregationFunction


class AggregationFunctionXAI(AggregationFunction):
    """XAI rules aggregation logic."""

    def call(self, local_tensors, db_iterator, *_):
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'rule' indicates that tensor is a set of xai rules of a collaborator

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: tensor of xai rules antecedents 
            np.ndarray: tensor of xai rules consequents 
            np.ndarray: tensor of xai rules weights 
        """

        # build collaborators rules dictionary
        collaborators_rules_dict = {}
        collaborators_list = []
        for i in db_iterator:
            if 'rule' in i.tags and not i.tensor_name == 'aggregated_rules':
                collaborators_list.append(i.tags[1])
                if i.tensor_name == 'antecedents':
                    if i.tags[1] not in collaborators_rules_dict.keys():
                        collaborators_rules_dict[i.tags[1]] = {}
                    collaborators_rules_dict[i.tags[1]]['antecedents'] = i.nparray
                if i.tensor_name == 'consequents':
                    if i.tags[1] not in collaborators_rules_dict.keys():
                        collaborators_rules_dict[i.tags[1]] = {}
                    collaborators_rules_dict[i.tags[1]]['consequents'] = i.nparray
                if i.tensor_name == 'rule_weights':
                    if i.tags[1] not in collaborators_rules_dict.keys():
                        collaborators_rules_dict[i.tags[1]] = {}
                    collaborators_rules_dict[i.tags[1]]['rule_weights'] = i.nparray

        # build aggregated TSK Rules set
        collaborators_list = set(collaborators_list)

        col = collaborators_list.pop()
        antecedents = collaborators_rules_dict[col]['antecedents']
        consequents = collaborators_rules_dict[col]['consequents']
        rules_weight = collaborators_rules_dict[col]['rule_weights']

        for col in collaborators_list:
            antecedent = collaborators_rules_dict[col]['antecedents']
            consequent = collaborators_rules_dict[col]['consequents']
            rule_weight = collaborators_rules_dict[col]['rule_weights']

            antecedents = np.concatenate((antecedents, antecedent), axis=0)
            consequents = np.concatenate((consequents, consequent), axis=0)
            rules_weight = np.concatenate((rules_weight, rule_weight))

        unq, unique_indexes, count = np.unique(ar=antecedents, axis=0, return_counts=True,
                                               return_index=True)

        # Contain the antecedents which cause duplications
        repeated_groups = unq[count > 1]

        for repeated_group in repeated_groups:
            # Contain the indexes where are present the duplicated antecedent
            repeated_idx = np.argwhere(np.all(antecedents == repeated_group, axis=1)).ravel()

            index_to_update = list(set(repeated_idx).intersection(unique_indexes))[0]

            # Extract the consequents
            consequents_same_antecedent = consequents[repeated_idx]

            # Extract the rule weights
            rules_weight_same_antecedents = rules_weight[repeated_idx]

            average_consequents = np.average(consequents_same_antecedent, axis=0, weights=rules_weight_same_antecedents)
            average_rule_weights = np.average(rules_weight_same_antecedents, axis=0)

            consequents[index_to_update] = average_consequents
            rules_weight[index_to_update] = average_rule_weights

        # ******** RETRIEVE THE UNIQUE RULES **********#

        unique_antecedents = antecedents[unique_indexes]
        unique_consequents = consequents[unique_indexes]
        unique_rule_weights = rules_weight[unique_indexes]

        return unique_antecedents, unique_consequents, unique_rule_weights
