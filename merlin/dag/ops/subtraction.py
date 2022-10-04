#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import copy

from merlin.core.dispatch import DataFrameType
from merlin.dag.base_operator import BaseOperator
from merlin.schema import Schema


class SubtractionOp(BaseOperator):
    """
    This operator class provides an implementation for the `-` operator used in constructing graphs.
    """

    def __init__(self, other_nodes):
        self.output_schema
        self.other_nodes = other_nodes
        super().__init__()

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        schema = copy.copy(input_schema)
        for node in self.other_nodes:
            schema -= node.output_schema
        self.output_schema = schema
        return schema

    def transform(self, input_schema: Schema, df: DataFrameType) -> DataFrameType:
        return df[self.output_schema.column_names]

    transform.__doc__ = BaseOperator.__doc__
