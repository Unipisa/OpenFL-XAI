# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
from simpful import FuzzySet, Triangular_MF


class FuzzySystem:
    FS_low = FuzzySet(function=Triangular_MF(a=-0.5, b=0, c=0.5), term='LOW')
    FS_medium = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term='MEDIUM')
    FS_high = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1.5), term='HIGH')

    FS_mapping = [FS_low, FS_medium, FS_high]

    # ******** MAPPING ******** #
    # 0 -> FS_low
    # 1 -> FS_medium
    # 2 -> FS_HIGH
    # ******** MAPPING ******** #

    def __init__(self, variable_names: List[str], antecedents: np.ndarray,
                 consequents: np.ndarray, rule_weights: np.ndarray):
        """
        :param variable_names: the name of the variable to build the FIS
        :param antecedents: the antecedents with shape(num_rules, num_input_features)
        :param consequents: shape (num_rules, num_input_features + 1)
        :param rule_weights: shape (num_rules,)
        """

        self._variable_names = variable_names

        self._antecedents = np.array([[int(item) if not isinstance(item, int) else item for item in rule] for rule in antecedents])
        self._consequents = consequents
        self._rule_weights = rule_weights

        self._num_rules = len(self._antecedents)

        self._counter_activation = np.zeros(self._num_rules)

    def clear_counter_activation(self) -> None:
        """
        Function to reset the counter for the rules activation
        :return: None
        """

        self._counter_activation = np.zeros(self._num_rules)

    def get_counter_activation(self) -> np.ndarray:
        """
        Return the counter for the rule activation
        :return: the counter of the rule activation
        """

        return self._counter_activation

    def get_variable_name(self) -> List[str]:
        """
        Return the variable names
        :return: a list of string with the variable names
        """

        return self._variable_names

    def _get_antecedent_as_string(self, antecedent: np.ndarray) -> str:
        """
        Utility function to get a string representation of the antecedent
        :param antecedent: a list of fuzzy set (expressed using the mapping) in the antecedent
        :return: a string representing the antecedent
        """
        result = 'IF '

        for elem, variable_name in zip(antecedent[: len(antecedent) - 1],
                                       self._variable_names[: len(self._variable_names) - 1]):
            result += '(' + variable_name + ' IS ' + FuzzySystem.FS_mapping[elem].get_term() + ') AND '

        result += '(' + self._variable_names[-1] + ' IS ' + FuzzySystem.FS_mapping[antecedent[-1]].get_term() + ')'

        return result

    def _get_consequent_as_string(self, consequent: np.ndarray) -> str:
        """
        Utility function to get a string representation of the consequent
        :param consequent: the weights of the consequent
        :return: a string representation of the consequent
        """

        is_first_order = not isinstance(consequent, float)

        if is_first_order:
            result = "{:.2e}".format(consequent[0])

            for weight, variable_name in zip(consequent[1:], self._variable_names):
                if weight > 0:
                    result += ' + '
                else:
                    result += ' - '
                result += "({:.2e}".format(abs(weight)) + ' * ' + variable_name + ')'
        else:
            result = '{:.2e}'.format(consequent)

        return result

    def __str__(self):
        result = ''
        rule_index = 1
        for antecedent, consequent, rule_weight in zip(self._antecedents.tolist(), self._consequents.tolist(),
                                                       self._rule_weights.tolist()):
            result += 'R' + str(rule_index) + ':\t' + self._get_antecedent_as_string(antecedent=antecedent) + '\n' + \
                      '\t\tTHEN: ' + self._get_consequent_as_string(consequent=consequent) + '\n' + \
                      '\t\t(Rule Weight: ' + '{:.2e}'.format(rule_weight) + ')\n\n'

            rule_index += 1

        return result

    def get_fuzzy_rules(self) -> Tuple[np.ndarray]:
        """
        Return the fuzzy rules
        :return: tuple:
            - antecedent with shape (number_of_rules, number_input_features)
            - consequent with shape (number_of_rules, number_input_features + 1)
            - rule_weights with shape (number_of_rule, )
        """

        return self._antecedents, self._consequents, self._rule_weights

    def save_to_file(self, path: str, name: str) -> None:
        """
        Function to save in a file the fuzzy system
        :param path: the path where save the fuzzy system. It must be terminated with /
        :param name: the name of the file
        :return: None
        """

        result = str(self)

        f = open(path + name + '.txt', 'w')
        f.write(result)
        f.close()

    @staticmethod
    def find_fuzzy_set(value: float) -> int:
        """
        Find the name of the fuzzy-set which is most activated
        :param value: the value to test
        :return: the fuzzy-set mainly activated (using the mapping between a fuzzy set and an integer)
        """

        membership_value_low = FuzzySystem.FS_low.get_value(value)
        membership_value_medium = FuzzySystem.FS_medium.get_value(value)
        membership_value_high = FuzzySystem.FS_high.get_value(value)

        values = [membership_value_low, membership_value_medium, membership_value_high]
        max_value = max(values)
        max_index = values.index(max_value)

        return max_index

    @staticmethod
    def compute_firing_strengths(antecedents, input_vector: np.ndarray, t_norm: str = 'product') -> np.ndarray:
        """
        Compute the firing strength of each rule given a certain input

        :param antecedents: the rules antecedents with shape(num_rules, num_input_features)
        :param input_vector: vector in the form (num_input_features,)
        :param t_norm: the t-norm to be used in the computation of the firing strength of the antecedent
        :return: array with shape (num_rules,)
        """

        if t_norm != 'min' and t_norm != 'product':
            raise ValueError('invalid t-norm')

        list_firing_strengths = list()

        for rule in antecedents:
            activations_values = list()
            for elem, value in zip(rule, input_vector):
                fuzzy_set = FuzzySystem.FS_mapping[int(elem)]
                membership_value = fuzzy_set.get_value(value)
                activations_values.append(membership_value)

            rule_firing_strength = 0
            if t_norm == 'min':
                rule_firing_strength = min(activations_values)
            if t_norm == 'product':
                rule_firing_strength = np.prod(activations_values)

            list_firing_strengths.append(rule_firing_strength)
        return np.array(list_firing_strengths)

    def get_firing_strengths(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Compute the firing strength of each rule given a certain input
        :param input_vector: vector in the form (number_of_input_features,)
        :return: array with shape (num_rules,)
        """

        firing_strengths = FuzzySystem.compute_firing_strengths(antecedents=self._antecedents,
                                                                input_vector=input_vector)

        return firing_strengths

    def get_number_of_rules(self) -> int:
        """
        Return the number of rules in the system
        :return: the number of rules
        """

        return self._num_rules

    def get_rule_by_index(self, index: int) -> str:
        """
        Get the rule corresponding to the given index
        :param index: the index of the rule
        :return: a string representation of the rule
        """

        if index < 1 or index > self._num_rules:
            raise ValueError('invalid index')

        index = index - 1

        antecedent = self._antecedents[index]
        consequent = self._consequents[index]

        res = 'R' + str(index) + ':\t' + self._get_antecedent_as_string(antecedent=antecedent) + '\n' + \
              '\tTHEN: ' + self._get_consequent_as_string(consequent=consequent)

        return res

    def __get_rule_maximum_weight(self) -> Tuple[np.ndarray, int]:
        """
        Utility function used to retrieve the rule with the maximum weight
        :return: a tuple with:
                    - antecedent
                    - consequent
                    - index
        """

        index_rule_weight = self._rule_weights.argmax()

        return self._antecedents[index_rule_weight, :], self._consequents[index_rule_weight, :], index_rule_weight

    def predict(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            the predicted values.
        """
        return np.array(list(map(lambda input_vector: self.__internal_predict(input_vector)[0], X)))

    def predict_and_get_rule(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [2] or [n_samples, 2]
            the predicted values and activated rule.
        """
        return list(map(lambda input_vector: self.__internal_predict(input_vector), X))

    def __internal_predict(self, input_vector: np.ndarray) -> Tuple[float, int]:
        """
        Predict the output
        :param input_vector: the input features in the form (n_features,)
        :return: the predicted value, the index of the rule
        """

        firing_strengths = self.get_firing_strengths(input_vector=input_vector)

        is_all_zero = np.all((firing_strengths == 0))
        if is_all_zero:
            _, consequent_weights, index_max_rules = self.__get_rule_maximum_weight()
        else:
            max_values_firing = firing_strengths.max()
            index_same_firing_strengths = np.where(firing_strengths == max_values_firing)[0]

            # Retrieve the relative weights
            rule_weight = self._rule_weights[index_same_firing_strengths]

            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]

            consequent_weights = self._consequents[index_max_rules]

        self._counter_activation[index_max_rules] += 1

        is_zero_order = isinstance(consequent_weights, float)

        if not is_zero_order:
            w0 = consequent_weights[0]

            consequent_weights = consequent_weights[1:]

            result = (consequent_weights * input_vector).sum() + w0
        else:
            result = consequent_weights

        return result, int(index_max_rules + 1)

    def predict_consider_all_activated_rules(self, input_vector: np.ndarray) -> float:
        """
        predict the result considering the weighted output of the all the rules activated. The weights are firing
        strengths
        :param input_vector: the input features in the form (num_input_features,)
        :return: the predicted value
        """
        firing_strengths = self.get_firing_strengths(input_vector=input_vector)

        sum_firing_strengths = np.sum(firing_strengths)
        weighted_result = 0

        if sum_firing_strengths == 0:
            _, consequent_weights, index_max_rules = self.__get_rule_maximum_weight()
            self._counter_activation[index_max_rules] += 1
            sum_firing_strengths = 1
            w0 = consequent_weights[0]
            consequent_weights = consequent_weights[1:]
            weighted_result = (consequent_weights * input_vector).sum() + w0
        else:
            index_rule = 0
            for firing_strength, consequent_weights in zip(firing_strengths, self._consequents):
                if firing_strength > 0:
                    w0 = consequent_weights[0]
                    consequent_weights = consequent_weights[1:]
                    result = (consequent_weights * input_vector).sum() + w0
                    weighted_result += (result * firing_strength)

                    self._counter_activation[index_rule] += 1

                    index_rule += 1

        return weighted_result / sum_firing_strengths
