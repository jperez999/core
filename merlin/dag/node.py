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
import collections.abc
from typing import List, Union

from merlin.dag.base_operator import BaseOperator
from merlin.dag.ops import ConcatColumns, SubtractionOp
from merlin.dag.ops.selector import ColumnSelector
from merlin.schema import Schema


class Node:
    """A Node is a group of columns that you want to apply the same transformations to.
    Node's can be transformed by shifting operators on to them, which returns a new
    Node with the transformations applied. This lets you define a graph of operations
    that makes up your workflow

    Parameters
    ----------
    op: Operator
        Defines which columns to select from the input Dataset using column names and tags.
    """

    def __init__(self, op=None):
        self.parents = []
        self.children = []

        dependencies = op.dependencies()
        if dependencies:
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

        for dependency in dependencies:
            self.add_parent(dependency)

        self.op = op

        self.input_schema = None  # TODO: do we need an input_schema still ?
        self.output_schema = None

    def add_parent(
        self, parent: Union[str, ColumnSelector, "Node", List[Union[str, "Node", ColumnSelector]]]
    ):
        """
        Adding a parent node to this node

        Parameters
        ----------
        parent : Union[str, ColumnSelector, Node, List[Union[str, Node, ColumnSelector]]]
            Parent to be added
        """
        parent_nodes = Node.construct_from(parent)

        if not isinstance(parent_nodes, list):
            parent_nodes = [parent_nodes]

        for parent_node in parent_nodes:
            parent_node.children.append(self)

        self.parents.extend(parent_nodes)

    def add_child(
        self, child: Union[str, ColumnSelector, "Node", List[Union[str, "Node", ColumnSelector]]]
    ):
        """
        Adding a child node to this node

        Parameters
        ----------
        child : Union[str, ColumnSelector, Node, List[Union[str, Node, ColumnSelector]]]
            Child to be added
        """
        child_nodes = Node.construct_from(child)

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        for child_node in child_nodes:
            child_node.parents.append(self)

        self.children.extend(child_nodes)

    def remove_child(
        self, child: Union[str, ColumnSelector, "Node", List[Union[str, "Node", ColumnSelector]]]
    ):
        """
        Removing a child node from this node

        Parameters
        ----------
        child : Union[str, ColumnSelector, Node, List[Union[str, Node, ColumnSelector]]]
            Child to be removed
        """
        child_nodes = Node.construct_from(child)

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        for child_node in child_nodes:
            if self in child_node.parents:
                child_node.parents.remove(self)
            if child_node in self.children:
                self.children.remove(child_node)

    def compute_schemas(self, root_schema: Schema, preserve_dtypes: bool = False):
        """
        Defines the input and output schema

        Parameters
        ----------
        root_schema : Schema
            Schema of the input dataset
        preserve_dtypes : bool, optional
            `True` if we don't want to override dtypes in the current schema, by default False
        """
        if self.parents:
            self.input_schema = _combine_schemas(self.parents)
        else:
            self.input_schema = root_schema

        self.output_schema = self.op.compute_output_schema(self.input_schema)

    def __rshift__(self, operator):
        """Transforms this Node by applying an BaseOperator

        Parameters
        -----------
        operators: BaseOperator or callable

        Returns
        -------
        Node
        """
        if isinstance(operator, type) and issubclass(operator, BaseOperator):
            # handle case where an operator class is passed
            operator = operator()

        if not isinstance(operator, BaseOperator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")

        child = type(self)(operator)
        child.add_parent(self)

        return child

    def __add__(self, other):
        """Adds columns from this Node with another to return a new Node

        Parameters
        -----------
        other: Node or str or list of str

        Returns
        -------
        Node
        """
        if isinstance(self.op, ConcatColumns):
            child = self
        else:
            # Create a child node
            child = type(self)(ConcatColumns())
            child.add_parent(self)

        # The right operand becomes a dependency
        other_nodes = Node.construct_from(other)
        other_nodes = [other_nodes]

        for other_node in other_nodes:
            # If the other node is a `+` node, we want to collapse it into this `+` node to
            # avoid creating a cascade of repeated `+`s that we'd need to optimize out by
            # re-combining them later in order to clean up the graph
            if not isinstance(other_node, list) and isinstance(other_node.op, ConcatColumns):
                child.parents.extend(other_node.parents)
            else:
                child.add_parent(other_node)

        return child

    # handle the "column_name" + Node case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this Node with another to return a new Node

        Parameters
        -----------
        other: Node or str or list of str
            Columns to remove

        Returns
        -------
        Node
        """
        other_nodes = Node.construct_from(other)

        if not isinstance(other_nodes, list):
            other_nodes = [other_nodes]

        child = type(self)(SubtractionOp(other_nodes))
        child.add_parent(self)

        return child

    def __getitem__(self, columns):
        """Selects certain columns from this Node, and returns a new Columngroup with only
        those columns

        Parameters
        -----------
        columns: str or list of str
            Columns to select

        Returns
        -------
        Node
        """
        col_selector = ColumnSelector(columns)
        child = type(self)(col_selector)
        columns = [columns] if not isinstance(columns, list) else columns
        child.add_parent(self)
        return child

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<Node {self.label}{output}>"

    def remove_inputs(self, input_cols):
        # TODO: this probably won't work ?? (self.column_mapping logic)
        removed_outputs = _derived_output_cols(input_cols, self.column_mapping)

        self.input_schema = self.input_schema.without(input_cols)
        self.output_schema = self.output_schema.without(removed_outputs)

        return removed_outputs

    @property
    def exportable(self):
        return hasattr(self.op, "export")

    @property
    def output_columns(self):
        if self.output_schema is None:
            raise RuntimeError(
                "The output columns aren't computed until the workflow "
                "is fit to a dataset or input schema."
            )

        return self.output_schema.column_names

    @property
    def column_mapping(self):
        return self.op.column_mapping(self.input_schema)

    @property
    def label(self):
        if self.op and hasattr(self.op, "label"):
            return self.op.label
        elif self.op:
            return str(type(self.op))
        elif not self.parents:
            return f"input cols=[{self._cols_repr}]"
        else:
            return "??"

    @property
    def _cols_repr(self):
        columns = self.input_schema.column_names
        cols_repr = ", ".join(map(str, columns[:3]))
        if len(columns) > 3:
            cols_repr += "..."
        return cols_repr

    @property
    def graph(self):
        return _to_graphviz(self)

    @classmethod
    def construct_from(cls, nodable):
        if isinstance(nodable, str):
            return Node(ColumnSelector([nodable]))
        if isinstance(nodable, ColumnSelector):
            return Node(nodable)
        elif isinstance(nodable, Node):
            return nodable
        elif isinstance(nodable, list):
            # TODO: is this right ?
            return Node(ColumnSelector(nodable))
        else:
            raise TypeError(
                "Unsupported type: Cannot convert object " f"of type {type(nodable)} to Node."
            )


def iter_nodes(nodes):
    queue = nodes[:]
    while queue:
        current = queue.pop()
        if isinstance(current, list):
            queue.extend(current)
        else:
            yield current
            for node in current.parents:
                if node not in queue:
                    queue.append(node)


# output node (bottom) -> selection leaf nodes (top)
def preorder_iter_nodes(nodes):
    queue = []
    if not isinstance(nodes, list):
        nodes = [nodes]

    def traverse(current_nodes):
        for node in current_nodes:
            # Avoid creating duplicate nodes in the queue
            if node in queue:
                queue.remove(node)

            queue.append(node)

        for node in current_nodes:
            traverse(node.parents)

    traverse(nodes)
    for node in queue:
        yield node


# selection leaf nodes (top) -> output node (bottom)
def postorder_iter_nodes(nodes):
    queue = []
    if not isinstance(nodes, list):
        nodes = [nodes]

    def traverse(current_nodes):
        for node in current_nodes:
            traverse(node.parents)
            if node not in queue:
                queue.append(node)

    traverse(nodes)
    for node in queue:
        yield node


def _combine_schemas(elements):
    combined = Schema()
    for elem in elements:
        if isinstance(elem, Node):
            combined += elem.output_schema
        elif isinstance(elem, list):
            combined += _combine_schemas(elem)
    return combined


def _to_graphviz(output_node):
    """Converts a Node to a GraphViz DiGraph object useful for display in notebooks"""
    from graphviz import Digraph

    graph = Digraph()

    # get all the nodes from parents of this columngroup
    # and add edges between each of them
    allnodes = list(set(iter_nodes([output_node])))
    node_ids = {v: str(k) for k, v in enumerate(allnodes)}
    for node, nodeid in node_ids.items():
        graph.node(nodeid, node.label)
        for parent in node.parents:
            graph.edge(node_ids[parent], nodeid)

        if node.selector and node.selector.names:
            selector_id = f"{nodeid}_selector"
            graph.node(selector_id, str(node.selector.names))
            graph.edge(selector_id, nodeid)

    # add a single node representing the final state
    final_node_id = str(len(allnodes))
    final_string = "output cols"
    if output_node._cols_repr:
        final_string += f"=[{output_node._cols_repr}]"
    graph.node(final_node_id, final_string)
    graph.edge(node_ids[output_node], final_node_id)
    return graph


def _derived_output_cols(input_cols, column_mapping):
    outputs = []
    for input_col in set(input_cols):
        for output_col_name, input_col_list in column_mapping.items():
            if input_col in input_col_list:
                outputs.append(output_col_name)
    return outputs
