# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""TensorDBXAI Module. (TensorDB extension)"""

import numpy as np
from openfl.databases import TensorDB
from openfl.utilities import LocalTensor
from openfl.utilities import TensorKey


class TensorDBXAI(TensorDB):
    """
    The TensorDB stores a tensor key and the data that it corresponds to.

    It is built on top of a pandas dataframe
    for it's easy insertion, retreival and aggregation capabilities. Each
    collaborator and aggregator has its own TensorDB.
    """

    def get_aggregated_tensor(self, tensor_key, collaborator_weight_dict,
                              aggregation_function):
        """
        Args:
            tensor_key: The tensor key to be resolved.
            collaborator_weight_dict: List of collaborator names in federation
                                      and their respective weights
            aggregation_function: Call the underlying aggregation
                                   function. 
        Returns:
            Aggregated model in antecedents, consequents, weights. Each in a numpy array.
            None if not all values are present

        """
        if len(collaborator_weight_dict) != 0:
            assert np.abs(1.0 - sum(collaborator_weight_dict.values())) < 0.01, (
                f'Collaborator weights do not sum to 1.0: {collaborator_weight_dict}'
            )

        collaborator_names = collaborator_weight_dict.keys()
        agg_tensor_dict = {}

        # Check if the aggregated tensor is already present in TensorDB
        tensor_name, origin, fl_round, report, tags = tensor_key

        raw_df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_name)
                                & (self.tensor_db['origin'] == origin)
                                & (self.tensor_db['round'] == fl_round)
                                & (self.tensor_db['report'] == report)
                                & (self.tensor_db['tags'] == tags)]['nparray']
        if len(raw_df) > 0:
            return np.array(raw_df.iloc[0]), {}

        for col in collaborator_names:
            if type(tags) == str:
                new_tags = tuple([tags] + [col])
            else:
                new_tags = tuple(list(tags) + [col])
            raw_df = self.tensor_db[
                (self.tensor_db['tensor_name'] == tensor_name)
                & (self.tensor_db['origin'] == origin)
                & (self.tensor_db['round'] == fl_round)
                & (self.tensor_db['report'] == report)
                & (self.tensor_db['tags'] == new_tags)]['nparray']
            if len(raw_df) == 0:
                tk = TensorKey(tensor_name, origin, report, fl_round, new_tags)
                print(f'No results for collaborator {col}, TensorKey={tk}')
                return None
            else:
                agg_tensor_dict[col] = raw_df.iloc[0]

        local_tensors = [LocalTensor(col_name=col_name,
                                     tensor=agg_tensor_dict[col_name],
                                     weight=collaborator_weight_dict[col_name])
                         for col_name in collaborator_names]

        db_iterator = self._iterate()
        agg_nparray_rules_antecedents, agg_nparray_rules_consequents, agg_nparray_rules_weights = aggregation_function(
            local_tensors,
            db_iterator,
            tensor_name,
            fl_round,
            tags)

        return np.array(agg_nparray_rules_antecedents), np.array(agg_nparray_rules_consequents), np.array(
            agg_nparray_rules_weights)
