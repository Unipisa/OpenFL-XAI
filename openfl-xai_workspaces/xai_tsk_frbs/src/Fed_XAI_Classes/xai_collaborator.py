# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

""" CollaboratorXAI module. (Extends Collaborator base class)"""

from logging import getLogger
from time import sleep

from openfl.component.collaborator import Collaborator
from openfl.pipelines import NoCompressionPipeline
from openfl.pipelines import TensorCodec
from src.Fed_XAI_Classes.tensor_db import TensorDBXAI


class CollaboratorXAI(Collaborator):
    r"""The Collaborator object class extension for XAI model.

    Args:
        collaborator_name (string): The common name for the collaborator
        aggregator_uuid: The unique id for the client
        federation_uuid: The unique id for the federation
        compression_pipeline: The compression pipeline (Defaults to None)
        task_runner: TaskRunnerXAI object
    """

    def __init__(self,
                 collaborator_name,
                 aggregator_uuid,
                 federation_uuid,
                 client,
                 task_runner,
                 task_config,
                 compression_pipeline=None,
                 **kwargs):
        """Initialize."""
        self.single_col_cert_common_name = None

        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = ''  # for protobuf compatibility

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)

        # create new instance of custom class TensorDBXAI, extended from openfl.databases import TensorDB
        self.tensor_db = TensorDBXAI()

        self.task_runner = task_runner

        self.client = client

        self.task_config = task_config

        self.logger = getLogger(__name__)

    def do_task(self, task, round_number):
        """Do the specified task."""

        func_name = self.task_config[task]['function']
        kwargs = self.task_config[task]['kwargs']

        input_tensor_dict = {}

        if hasattr(self.task_runner, 'TASK_REGISTRY'):
            # New interactive python API
            # New `Core` TaskRunner contains registry of tasks
            func = self.task_runner.TASK_REGISTRY[func_name]
            self.logger.info('Using Interactive Python API')
        else:
            # TaskRunner subclassing API
            # Tasks are defined as methods of TaskRunner
            func = getattr(self.task_runner, func_name)
            self.logger.info('Using TaskRunner subclassing API')

        global_output_tensor_dict = func(
            col_name=self.collaborator_name,
            round_num=round_number,
            input_tensor_dict=input_tensor_dict,
            **kwargs)

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)

        # send the results for these tasks; delta and compression will occur in
        # this function
        self.send_task_results(global_output_tensor_dict, round_number, task)

    def send_task_results(self, tensor_dict, round_number, task_name):
        """Send task results to the aggregator."""
        named_tensors = [
            self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()
        ]

        data_size = -1

        if 'train' in task_name:
            data_size = self.task_runner.get_train_data_size()

        if 'valid' in task_name:
            data_size = self.task_runner.get_valid_data_size()

        self.logger.debug(f'{task_name} data size = {data_size}')

        for tensor in tensor_dict:
            tensor_name, origin, fl_round, report, tags = tensor

            if report:
                self.logger.metric(
                    f'Round {round_number}, collaborator {self.collaborator_name} '
                    f'is sending metric for task {task_name}:'
                    f' {tensor_name}\t{tensor_dict[tensor]}')

        self.client.send_local_task_results(
            self.collaborator_name, round_number, task_name, data_size, named_tensors)

    def run(self):
        """Run the collaborator."""
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                break
            elif sleep_time > 0:
                sleep(sleep_time)
            else:
                self.logger.info(f'Received the following tasks: {tasks}')
                for task in tasks:
                    self.do_task(task, round_number)

        self.logger.info('End of Federation reached. Exiting...')
