# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""XAI FRBS Aggregator module."""
import queue
from importlib import import_module
from logging import getLogger

import numpy as np
from openfl.component import Aggregator
from openfl.pipelines import NoCompressionPipeline
from openfl.pipelines import TensorCodec
from openfl.utilities import TaskResultKey
from openfl.utilities import TensorKey
from openfl.utilities.logs import write_metric
from src.Fed_XAI_Classes.tensor_db import TensorDBXAI


class AggregatorXAI(Aggregator):
    r"""An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (str): Aggregation ID.
        federation_uuid (str): Federation ID.
        authorized_cols (list of str): The list of IDs of enrolled collaborators.


        aggregation_function_path_template (dict): path to import AggregationFunction
        model_infos (dict): name of the model, can be extended with additional parameters

    Note:
        \* - plan setting.
    """

    def __init__(self,

                 aggregator_uuid,
                 federation_uuid,
                 authorized_cols,
                 aggregation_function_path_template,
                 model_infos,
                 assigner,
                 rounds_to_train=256,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,

                 **kwargs):
        """Initialize."""
        self.round_number = 0
        self.single_col_cert_common_name = single_col_cert_common_name

        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            self.single_col_cert_common_name = ''

        self.rounds_to_train = rounds_to_train

        self.authorized_cols = authorized_cols
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.assigner = assigner
        self.quit_job_sent_to = []

        self.model_infos = model_infos
        self.aggregation_function_path = {
            "path": aggregation_function_path_template["aggregation_function_module_path"],
            "class_name": aggregation_function_path_template["aggregation_function_class_name"]}

        # create new instance of custom class TensorDBXAI, extended from openfl.databases import TensorDB
        self.tensor_db = TensorDBXAI()

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.logger = getLogger(__name__)

        self.best_tensor_dict: dict = {}
        self.last_tensor_dict: dict = {}

        self.metric_queue = queue.Queue()
        self.best_model_score = None

        self.log_dir = f'logs/{self.uuid}_{self.federation_uuid}'

        self.collaborator_tensor_results = {}  # {TensorKey: nparray}}

        self.collaborator_tasks_results = {}  # {TaskResultKey: list of TensorKeys}

        self.collaborator_task_weight = {}  # {TaskResultKey: data_size}

        self.log_metric = write_metric

    def _compute_validation_related_task_metrics(self, task_name):
        """
        The FL Round is finished: all collaborators sent their local rule
        set to the Aggregator. This function retrieves the data sizes of the 
        Collaborators datasets to use them as weights in the aggregation 
        self.metric_queue.put(metric_dict) procedure (and nomalizes these). 
        Retrieves the aggregated_tensor_key 
        and calls the get_aggregated_tensor function which performs the 
        local models aggregation in the Global model and returns it.  

        Args:
            task_name : str
                The task name to compute
        
        """

        # This handles getting the subset of collaborators that may be
        # part of the task
        collaborators_for_task = self.assigner.get_collaborators_for_task(
            task_name, self.round_number)

        # The collaborator data sizes for that task
        collaborator_weights_unnormalized = {
            c: self.collaborator_task_weight[TaskResultKey(task_name, c, self.round_number)]
            for c in collaborators_for_task}
        weight_total = sum(collaborator_weights_unnormalized.values())
        collaborator_weight_dict = {
            k: v / weight_total
            for k, v in collaborator_weights_unnormalized.items()
        }

        # The validation task should have just a couple tensors (i.e.
        # metrics) associated with it. Because each collaborator should
        # have sent the same tensor list, we can use the first
        # collaborator in our subset, and apply the correct
        # transformations to the tensorkey to resolve the aggregated
        # tensor for that round
        task_agg_function = self.assigner.get_aggregation_type_for_task(task_name)
        task_key = TaskResultKey(task_name, collaborators_for_task[0], self.round_number)

        tensor_key = self.collaborator_tasks_results[task_key][0]
        tensor_name, origin, round_number, report, tags = tensor_key

        assert (tags[-1] == collaborators_for_task[0]), (
            f'Tensor {tensor_key} in task {task_name} has not been processed correctly'
        )

        # Strip the collaborator label, and lookup aggregated tensor
        new_tags = tuple(tags[:-1])
        agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        agg_tensor_name, agg_origin, agg_round_number, agg_report, agg_tags = agg_tensor_key

        agg_fun_module_path = import_module(self.aggregation_function_path["path"])
        agg_function = getattr(agg_fun_module_path, self.aggregation_function_path["class_name"])() if 'rule' in tags else task_agg_function

        agg_rules_antecedents_results, agg_rules_consequents_results, agg_rules_weights_results = self.tensor_db.get_aggregated_tensor(
            agg_tensor_key, collaborator_weight_dict, aggregation_function=agg_function)

        agg_rules_results = agg_rules_antecedents_results
        agg_rules_results = np.concatenate((agg_rules_results, agg_rules_consequents_results), axis=1)

        self.save_xai_model(agg_rules_antecedents_results, agg_rules_consequents_results, agg_rules_weights_results)

        if report:
            # Print the aggregated metric
            metric_dict = {
                'metric_origin': 'Aggregator',
                'task_name': task_name,
                'metric_name': tensor_key.tensor_name,
                'metric_value': agg_rules_results,
                'round': round_number}

            if agg_rules_antecedents_results is None or agg_rules_consequents_results is None:
                self.logger.warning(
                    f'Aggregated metric {agg_tensor_name} could not be collected '
                    f'for round {self.round_number}. Skipping reporting for this round')

            if agg_function:
                if agg_tensor_name in ['antecedents', 'consequents']:
                    agg_tensor_name = 'aggregated_rules'

            self.logger.metric(f'Round {round_number}, aggregator: {task_name} '
                               f'{agg_function} \n\n{agg_tensor_name} antecedents:\t\n\n{agg_rules_antecedents_results:}')

            self.logger.metric(f'Round {round_number}, aggregator: {task_name} '
                               f'{agg_function} \n\n{agg_tensor_name} consequents:\t\n\n{agg_rules_consequents_results:}')

            self.logger.metric(f'\naggregated_rule_weights: \n\n{agg_rules_weights_results}')

            self.metric_queue.put(metric_dict)

    def _end_of_round_check(self):
        """
        Check if the federated learning round terminated.

        If round terminated, then execute the following operations:
            - _compute_validation_related_task_metrics to compute aggregated model
            - increment round number
            - check if _time_to_quit()
        
        Args:
            None

        Returns:
            None
        """
        if not self._is_round_done():
            return

        # Compute all validation related metrics
        all_tasks = self.assigner.get_all_tasks_for_round(self.round_number)
        for task_name in all_tasks:
            self._compute_validation_related_task_metrics(task_name)

        # Once all the task results have been processed
        # Increment the round number
        self.round_number += 1

        if self._time_to_quit():
            self.logger.info('Experiment Completed. Cleaning up...')

    def save_xai_model(self, rules_antec, rules_conseq, rules_weights):
        """
            Utility to save the Aggregated XAI model on aggregator's local file system. 
            The aggregated XAI FRBS model consists of 3 numpy arrays: 
                - aggregated rule antecedents 
                - aggregated rule consequents 
                - aggregated rule weiights 

        """
        model = self.model_infos["model_name"]
        np.save("./" + model + "_global_model_rules_antec.npy", rules_antec)
        np.save("./" + model + "_global_model_rules_conseq.npy", rules_conseq)
        np.save("./" + model + "_global_model_weights.npy", rules_weights)

        self.logger.info(f'Shape of aggregated rules antecedents in federated model: {rules_antec.shape}')
        self.logger.info(f'Shape of aggregated rules consequents in federated model: {rules_conseq.shape}')
