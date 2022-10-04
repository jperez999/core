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

from enum import Flag, auto
from typing import Any, List, Union

import merlin.dag
from merlin.core.dispatch import DataFrameType
from merlin.schema import Schema


class Supports(Flag):
    """Indicates what type of data representation this operator supports for transformations"""

    # cudf dataframe
    CPU_DATAFRAME = auto()
    # pandas dataframe
    GPU_DATAFRAME = auto()
    # dict of column name to numpy array
    CPU_DICT_ARRAY = auto()
    # dict of column name to cupy array
    GPU_DICT_ARRAY = auto()


class BaseOperator:
    """
    Base class for all operator classes.
    """

    def transform(self, input_schema: Schema, df: DataFrameType):
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        input_schema: Schema
            The schema describing the input columns of df
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on
        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        # should be the same as df[input_schema.column_names]
        return df

    def compute_output_schema(self, input_schema: Schema) -> Schema:
        """Given a set of schemas and a column selector for the input columns,
        returns a set of schemas for the transformed columns this operator will produce
        Parameters
        -----------
        input_schema: Schema
            The schemas of the columns to apply this operator to
        Returns
        -------
        Schema
            The schemas of the columns produced by this operator
        """
        return input_schema

    @property
    def dynamic_dtypes(self):
        # TODO: is this still necessary ?
        return False

    @property
    def dependencies(self) -> List[Union[str, Any]]:
        """Defines an optional list of column dependencies for this operator. This lets you consume columns
        that aren't part of the main transformation workflow.
        Returns
        -------
        str, list of str or ColumnSelector, optional
            Extra dependencies of this operator. Defaults to None
        """
        return []

    def __rrshift__(self, other):
        return merlin.dag.Node(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__

    @property
    def supports(self) -> Supports:
        """Returns what kind of data representation this operator supports"""
        return Supports.CPU_DATAFRAME | Supports.GPU_DATAFRAME

    def column_mapping(self, input_schema):
        """Maps input columns to output columns.

        Used for figuring out which output columns should be removed when
        a given input column is removed
        """
        column_mapping = {}
        for col_name in input_schema.column_names:
            column_mapping[col_name] = [col_name]
        return column_mapping
