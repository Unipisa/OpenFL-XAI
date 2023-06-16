# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from typing import Tuple

import numpy as np
from openfl.utilities import TensorKey
from pyfume import ConsequentEstimator, Clusterer
from simpful import FuzzySet, Triangular_MF
from src.model.fuzzySystem import FuzzySystem
from src.model.model import Model


class TSK(Model):

    def __init__(self, parameters_dict, **kwars):

        self.num_rules = parameters_dict["num_rules"]
        self.order_consequent = parameters_dict["order_consequent"]

    def train(self, col_name, round_num, data_loader, **kwargs):
        # training_x: is the training set and must be in the format (number of sample, number_of_input_features)
        training_x = data_loader.get_train_data()
        training_y = data_loader.get_train_labels()
        num_rules = self.num_rules

        np.random.seed(0)
        cl = Clusterer(x_train=training_x, y_train=training_y, nr_clus=num_rules)
        centroids, _, _ = cl.cluster(method='fcm', m=2)

        antecedents, num_conflicting_antecedents = self.generate_antecedents(centroids=centroids)

        consequents, num_training_samples = self.generate_consequents(antecedents=antecedents,
                                                                      training_x=training_x, training_y=training_y,
                                                                      order_consequent=self.order_consequent)

        rule_weights = self.compute_weight_rules(antecedents=antecedents, consequents=consequents,
                                                 training_x=training_x, training_y=training_y)

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

        self.save_xai_model(antecedents, consequents, rule_weights)

        return output_rules_dict

    def generate_antecedents(self, centroids: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Utility function to generate the antecedents
        :param centroids: the centroid with shape (num_centroid, num_input_features + 1)
        :return: Tuple
                - antecedents with shape (number_antecedent, num_input_features)
                - number conflicting antecedents
        """

        centroid_dim = centroids.shape[1]

        results = list()

        for centroid in centroids:
            input_elements = centroid[: centroid_dim - 1].tolist()

            antecedents = list()
            for elem in input_elements:
                fuzzy_set = FuzzySystem.find_fuzzy_set(value=elem)
                antecedents.append(fuzzy_set)

            results.append(antecedents)

        antecedents_with_conflict = np.array(results)

        antecedents, num_conflicting_antecedents = np.unique(ar=antecedents_with_conflict, axis=0, return_counts=True)

        num_conflicting_antecedents = num_conflicting_antecedents - 1

        num_conflicting_antecedents = sum(num_conflicting_antecedents)

        return antecedents, num_conflicting_antecedents

    def generate_consequents(self, antecedents: np.ndarray, training_x: np.ndarray, training_y: np.ndarray,
                             order_consequent: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility function used to compute the firing strength of each rules for set of input points
        :param antecedents: the antecedents with shape (num_rules, num_input_features)
        :param training_x: the input values in the training-set in the form (number_of_samples, num_input_features)
        :param training_y: the value to predict in the form (number_of_samples,)
        :param order_consequent: the order of the consequent
        :return:
                Tuple:
                    first element
                        for estimation of first order:
                            - the consequent with shape (num_rules, num_input_features + 1)
                                w0 is in the first position
                        for estimation of zero order:
                            the consequent with shape (num_rules, )
                    second element
                        - the number of training samples used to estimates the rules shape(num_rules, )

        """

        if order_consequent != 'zero' and order_consequent != 'first':
            raise ValueError('order consequent can be \'first\' or \'zero\'')

        results = list()
        for element in training_x:
            rules_activation = FuzzySystem.compute_firing_strengths(antecedents=antecedents, input_vector=element)
            results.append(rules_activation)

        # (number_of_training_samples, num_of_rules)
        firing_strengths = np.array(results)

        num_samples_used_for_estimation = np.count_nonzero(firing_strengths, axis=0)

        ce = ConsequentEstimator(x_train=training_x, y_train=training_y,
                                 firing_strengths=firing_strengths)

        if order_consequent == 'first':
            # (number_of_rules, num_input_features + 1)
            consequent_parameters = ce.suglms()

            w_0 = consequent_parameters[:, -1]
            consequent_parameters = np.insert(consequent_parameters, 0, w_0, axis=1)
            consequents = np.delete(consequent_parameters, -1, axis=1)
        else:
            consequents = ce.zero_order()

        return consequents, num_samples_used_for_estimation

    def compute_weight_rules(self, antecedents: np.ndarray, consequents: np.ndarray,
                             training_x: np.ndarray, training_y: np.ndarray) -> np.ndarray:
        """
        Utility function to compute the support for each rule
        :param antecedents: the antecedents with shape (num_rules, num_input_features)
        :param consequents: the consequents with:
                            - shape(num_rules, num_input_features + 1) 'first order'
                            - shape(num_rules,) 'zero order'
        :param training_x: the input values in the training-set in the form (number_of_samples, number_of_input_features)
        :param training_y: the expected values in the training-set in the form (number_of_samples,)
        :return: rules weights with shape (num_rules,)
        """

        num_of_training_samples = len(training_x)
        num_of_rules = len(antecedents)
        is_zero_order = len(consequents.shape) == 1

        results = list()
        for element in training_x:
            rules_activation = FuzzySystem.compute_firing_strengths(antecedents=antecedents, input_vector=element)
            results.append(rules_activation)

        # shape (number_of_samples, number_of_rules)
        firing_strengths = np.array(results)

        prediction_matrix = np.zeros((num_of_training_samples, num_of_rules))

        for consequent, index_consequent in zip(consequents, range(num_of_rules)):
            if not is_zero_order:
                w0 = consequent[0]
                consequent = consequent[1:]
                for sample, index_sample in zip(training_x, range(num_of_training_samples)):
                    result = (consequent * sample).sum() + w0
                    prediction_matrix[index_sample, index_consequent] = result
            else:
                for sample, index_sample in zip(training_x, range(num_of_training_samples)):
                    result = consequent
                    prediction_matrix[index_sample, index_consequent] = result

        absolute_error_matrix = np.zeros((num_of_training_samples, num_of_rules))

        for index_rule in range(num_of_rules):
            for true_value, index_sample in zip(training_y, range(num_of_training_samples)):
                predicted_value = prediction_matrix[index_sample, index_rule]
                absolute_error = abs(predicted_value - true_value)
                absolute_error_matrix[index_sample, index_rule] = absolute_error

        max_ae = np.amax(absolute_error_matrix)
        min_ae = np.amin(absolute_error_matrix)
        delta_max_min_ae = max_ae - min_ae

        normalized_absolute_error_matrix = np.zeros((num_of_training_samples, num_of_rules))

        for i in range(num_of_training_samples):
            for j in range(num_of_rules):
                normalized_value = (absolute_error_matrix[i, j] - min_ae) / delta_max_min_ae
                normalized_absolute_error_matrix[i, j] = normalized_value

        FS_quality_of_prediction = FuzzySet(function=Triangular_MF(a=-1, b=0, c=1), term='QUALITY OF PREDICTION')

        membership_values_matrix = np.zeros((num_of_training_samples, num_of_rules))

        for i in range(num_of_training_samples):
            for j in range(num_of_rules):
                normalized_value = normalized_absolute_error_matrix[i, j]
                membership_values_matrix[i, j] = FS_quality_of_prediction.get_value(normalized_value)

        fuzzy_confidence = np.zeros(num_of_rules)
        fuzzy_support = np.zeros(num_of_rules)
        rule_weight = np.zeros(num_of_rules)

        for j in range(num_of_rules):
            sum_weighted_membership_value = 0
            sum_firing_strength = 0
            for i in range(num_of_training_samples):
                firing_strength = firing_strengths[i, j]
                res = firing_strength * membership_values_matrix[i, j]

                sum_firing_strength += firing_strength
                sum_weighted_membership_value += res

            fuzzy_confidence[j] = sum_weighted_membership_value / sum_firing_strength
            fuzzy_support[j] = sum_weighted_membership_value / num_of_training_samples
            rule_weight[j] = (2 * fuzzy_support[j] * fuzzy_confidence[j]) / (fuzzy_support[j] + fuzzy_confidence[j])

        return rule_weight

    def save_xai_model(self, rules_antec, rules_conseq, rules_weights):
        """
            Utility to save the Aggregated XAI model on aggregator's local file system. 
            The aggregated XAI TSK model consists of 3 numpy arrays: 
                - aggregated rule antecedents 
                - aggregated rule consequents 
                - aggregated rule weights 
        """

        np.save("./TSK_global_model_rules_antec.npy", rules_antec)
        np.save("./TSK_global_model_rules_conseq.npy", rules_conseq)
        np.save("./TSK_global_model_weights.npy", rules_weights)

        # self.logger.info(f'Shape of aggregated rules antecedents in federated model: {rules_antec.shape}')
        # self.logger.info(f'Shape of aggregated rules consequents in federated model: {rules_conseq.shape}')
