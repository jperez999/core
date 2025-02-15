#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: tensorflow_metadata/proto/v0/path.proto, tensorflow_metadata/proto/v0/schema.proto
# plugin: python-betterproto
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import betterproto

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from google.protobuf import json_format  # noqa: E402
from google.protobuf.any_pb2 import Any as AnyPb2  # noqa: E402
from google.protobuf.struct_pb2 import Struct  # noqa: E402


# This is a manual hack to make it possible to parse the Any types in annotations
@dataclass(eq=False, repr=False)
class Any(betterproto.Message):
    type_url: str = betterproto.string_field(1)
    value: betterproto.Message = betterproto.message_field(2)

    def from_dict(self, value: dict):
        if "@type" in value:
            self.value = value["value"]
            self.type_url = value["@type"]
        else:
            msg_struct = Struct()
            any_pack = AnyPb2()
            json_formatted = json_format.ParseDict(value, msg_struct)
            any_pack.Pack(json_formatted)
            self.value = any_pack.value
            self.type_url = any_pack.type_url
        return self

    def to_dict(
        self,
        casing: betterproto.Casing = betterproto.Casing.CAMEL,
        include_default_values: bool = False,
    ):
        msg_struct = Struct()
        msg_struct.ParseFromString(self.value)
        return dict(msg_struct.items())


class LifecycleStage(betterproto.Enum):
    """
    LifecycleStage. Only UNKNOWN_STAGE, BETA, and PRODUCTION features are
    actually validated. PLANNED, ALPHA, DISABLED, and DEBUG are treated as
    DEPRECATED.
    """

    UNKNOWN_STAGE = 0
    PLANNED = 1
    ALPHA = 2
    BETA = 3
    PRODUCTION = 4
    DEPRECATED = 5
    DEBUG_ONLY = 6
    DISABLED = 7


class FeatureType(betterproto.Enum):
    """
    Describes the physical representation of a feature. It may be different
    than the logical representation, which is represented as a Domain.
    """

    TYPE_UNKNOWN = 0
    BYTES = 1
    INT = 2
    FLOAT = 3
    STRUCT = 4


class TimeDomainIntegerTimeFormat(betterproto.Enum):
    FORMAT_UNKNOWN = 0
    UNIX_DAYS = 5
    UNIX_SECONDS = 1
    UNIX_MILLISECONDS = 2
    UNIX_MICROSECONDS = 3
    UNIX_NANOSECONDS = 4


class TimeOfDayDomainIntegerTimeOfDayFormat(betterproto.Enum):
    FORMAT_UNKNOWN = 0
    PACKED_64_NANOS = 1


class TensorRepresentationRowPartitionDType(betterproto.Enum):
    UNSPECIFIED = 0
    INT64 = 1
    INT32 = 2


@dataclass
class Path(betterproto.Message):
    """
    A path is a more general substitute for the name of a field or feature that
    can be used for flat examples as well as structured data. For example, if
    we had data in a protocol buffer: message Person {   int age = 1;
    optional string gender = 2;   repeated Person parent = 3; } Thus, here the
    path {step:["parent", "age"]} in statistics would refer to the age of a
    parent, and {step:["parent", "parent", "age"]} would refer to the age of a
    grandparent. This allows us to distinguish between the statistics of
    parents' ages and grandparents' ages. In general, repeated messages are to
    be preferred to linked lists of arbitrary length. For SequenceExample, if
    we have a feature list "foo", this is represented by {step:["##SEQUENCE##",
    "foo"]}.
    """

    # Any string is a valid step. However, whenever possible have a step be
    # [A-Za-z0-9_]+.
    step: List[str] = betterproto.string_field(1)


@dataclass
class Schema(betterproto.Message):
    """Message to represent schema information. NextID: 14"""

    # Features described in this schema.
    feature: List["Feature"] = betterproto.message_field(1)
    # Sparse features described in this schema.
    sparse_feature: List["SparseFeature"] = betterproto.message_field(6)
    # Weighted features described in this schema.
    weighted_feature: List["WeightedFeature"] = betterproto.message_field(12)
    # declared as top-level features in <feature>. String domains referenced in
    # the features.
    string_domain: List["StringDomain"] = betterproto.message_field(4)
    # top level float domains that can be reused by features
    float_domain: List["FloatDomain"] = betterproto.message_field(9)
    # top level int domains that can be reused by features
    int_domain: List["IntDomain"] = betterproto.message_field(10)
    # Default environments for each feature. An environment represents both a
    # type of location (e.g. a server or phone) and a time (e.g. right before
    # model X is run). In the standard scenario, 99% of the features should be in
    # the default environments TRAINING, SERVING, and the LABEL (or labels) AND
    # WEIGHT is only available at TRAINING (not at serving). Other possible
    # variations: 1. There may be TRAINING_MOBILE, SERVING_MOBILE,
    # TRAINING_SERVICE,    and SERVING_SERVICE. 2. If one is ensembling three
    # models, where the models of the first    three models are available
    # for the ensemble model, there may be    TRAINING, SERVING_INITIAL,
    # SERVING_ENSEMBLE. See FeatureProto::not_in_environment and
    # FeatureProto::in_environment.
    default_environment: List[str] = betterproto.string_field(5)
    # Additional information about the schema as a whole. Features may also be
    # annotated individually.
    annotation: "Annotation" = betterproto.message_field(8)
    # Dataset-level constraints. This is currently used for specifying
    # information about changes in num_examples.
    dataset_constraints: "DatasetConstraints" = betterproto.message_field(11)
    # TensorRepresentation groups. The keys are the names of the groups. Key ""
    # (empty string) denotes the "default" group, which is what should be used
    # when a group name is not provided. See the documentation at
    # TensorRepresentationGroup for more info. Under development. DO NOT USE.
    tensor_representation_group: Dict[str, "TensorRepresentationGroup"] = betterproto.map_field(
        13, betterproto.TYPE_STRING, betterproto.TYPE_MESSAGE
    )


@dataclass
class ValueCountList(betterproto.Message):
    value_count: List["ValueCount"] = betterproto.message_field(1)


@dataclass
class Feature(betterproto.Message):
    """
    Describes schema-level information about a specific feature. NextID: 33
    """

    # The name of the feature.
    name: str = betterproto.string_field(1)
    # This field is no longer supported. Instead, use: lifecycle_stage:
    # DEPRECATED TODO(b/111450258): remove this.
    deprecated: bool = betterproto.bool_field(2)
    # Constraints on the presence of this feature in the examples.
    presence: "FeaturePresence" = betterproto.message_field(14, group="presence_constraints")
    # Only used in the context of a "group" context, e.g., inside a sequence.
    group_presence: "FeaturePresenceWithinGroup" = betterproto.message_field(
        17, group="presence_constraints"
    )
    # The feature has a fixed shape corresponding to a multi-dimensional tensor.
    shape: "FixedShape" = betterproto.message_field(23, group="shape_type")
    # The feature doesn't have a well defined shape. All we know are limits on
    # the minimum and maximum number of values.
    value_count: "ValueCount" = betterproto.message_field(5, group="shape_type")
    # Captures the same information as value_count but for features with nested
    # values. A ValueCount is provided for each nest level.
    value_counts: "ValueCountList" = betterproto.message_field(32, group="shape_type")
    # Physical type of the feature's values. Note that you can have: type: BYTES
    # int_domain: {   min: 0   max: 3 } This would be a field that is
    # syntactically BYTES (i.e. strings), but semantically an int, i.e. it would
    # be "0", "1", "2", or "3".
    type: "FeatureType" = betterproto.enum_field(6)
    # Reference to a domain defined at the schema level.
    domain: str = betterproto.string_field(7, group="domain_info")
    # Inline definitions of domains.
    int_domain: "IntDomain" = betterproto.message_field(9, group="domain_info")
    float_domain: "FloatDomain" = betterproto.message_field(10, group="domain_info")
    string_domain: "StringDomain" = betterproto.message_field(11, group="domain_info")
    bool_domain: "BoolDomain" = betterproto.message_field(13, group="domain_info")
    struct_domain: "StructDomain" = betterproto.message_field(29, group="domain_info")
    # Supported semantic domains.
    natural_language_domain: "NaturalLanguageDomain" = betterproto.message_field(
        24, group="domain_info"
    )
    image_domain: "ImageDomain" = betterproto.message_field(25, group="domain_info")
    mid_domain: "MIDDomain" = betterproto.message_field(26, group="domain_info")
    url_domain: "URLDomain" = betterproto.message_field(27, group="domain_info")
    time_domain: "TimeDomain" = betterproto.message_field(28, group="domain_info")
    time_of_day_domain: "TimeOfDayDomain" = betterproto.message_field(30, group="domain_info")
    # Constraints on the distribution of the feature values. Only supported for
    # StringDomains.
    distribution_constraints: "DistributionConstraints" = betterproto.message_field(15)
    # Additional information about the feature for documentation purpose.
    annotation: "Annotation" = betterproto.message_field(16)
    # Tests comparing the distribution to the associated serving data.
    skew_comparator: "FeatureComparator" = betterproto.message_field(18)
    # Tests comparing the distribution between two consecutive spans (e.g. days).
    drift_comparator: "FeatureComparator" = betterproto.message_field(21)
    # List of environments this feature is present in. Should be disjoint from
    # not_in_environment. This feature is in environment "foo" if: ("foo" is in
    # in_environment or default_environment) AND "foo" is not in
    # not_in_environment. See Schema::default_environment.
    in_environment: List[str] = betterproto.string_field(20)
    # List of environments this feature is not present in. Should be disjoint
    # from of in_environment. See Schema::default_environment and in_environment.
    not_in_environment: List[str] = betterproto.string_field(19)
    # The lifecycle stage of a feature. It can also apply to its descendants.
    # i.e., if a struct is DEPRECATED, its children are implicitly deprecated.
    lifecycle_stage: "LifecycleStage" = betterproto.enum_field(22)
    # Constraints on the number of unique values for a given feature. This is
    # supported for string and categorical features only.
    unique_constraints: "UniqueConstraints" = betterproto.message_field(31)


@dataclass
class Annotation(betterproto.Message):
    """Additional information about the schema or about a feature."""

    # Tags can be used to mark features. For example, tag on user_age feature can
    # be `user_feature`, tag on user_country feature can be `location_feature`,
    # `user_feature`.
    tag: List[str] = betterproto.string_field(1)
    # Free-text comments. This can be used as a description of the feature,
    # developer notes etc.
    comment: List[str] = betterproto.string_field(2)
    # Application-specific metadata may be attached here.
    extra_metadata: List[Any] = betterproto.message_field(3)

    @property
    def metadata(self):
        if self.comment:
            metadata = SimpleNamespace(**json.loads(self.comment[-1]))
            return metadata

        return None


@dataclass
class NumericValueComparator(betterproto.Message):
    """
    Checks that the ratio of the current value to the previous value is not
    below the min_fraction_threshold or above the max_fraction_threshold. That
    is, previous value * min_fraction_threshold <= current value <= previous
    value * max_fraction_threshold. To specify that the value cannot change,
    set both min_fraction_threshold and max_fraction_threshold to 1.0.
    """

    min_fraction_threshold: float = betterproto.double_field(1)
    max_fraction_threshold: float = betterproto.double_field(2)


@dataclass
class DatasetConstraints(betterproto.Message):
    """Constraints on the entire dataset."""

    # Tests differences in number of examples between the current data and the
    # previous span.
    num_examples_drift_comparator: "NumericValueComparator" = betterproto.message_field(1)
    # Tests comparisons in number of examples between the current data and the
    # previous version of that data.
    num_examples_version_comparator: "NumericValueComparator" = betterproto.message_field(2)
    # Minimum number of examples in the dataset.
    min_examples_count: int = betterproto.int64_field(3)
    # Maximum number of examples in the dataset.
    max_examples_count: int = betterproto.int64_field(4)


@dataclass
class FixedShape(betterproto.Message):
    """
    Specifies a fixed shape for the feature's values. The immediate implication
    is that each feature has a fixed number of values. Moreover, these values
    can be parsed in a multi-dimensional tensor using the specified axis sizes.
    The FixedShape defines a lexicographical ordering of the data. For
    instance, if there is a FixedShape {   dim {size:3} dim {size:2} } then
    tensor[0][0]=field[0] then tensor[0][1]=field[1] then tensor[1][0]=field[2]
    then tensor[1][1]=field[3] then tensor[2][0]=field[4] then
    tensor[2][1]=field[5] The FixedShape message is identical with the
    TensorFlow TensorShape proto message.
    """

    # The dimensions that define the shape. The total number of values in each
    # example is the product of sizes of each dimension.
    dim: List["FixedShapeDim"] = betterproto.message_field(2)


@dataclass
class FixedShapeDim(betterproto.Message):
    """An axis in a multi-dimensional feature representation."""

    size: int = betterproto.int64_field(1)
    # Optional name of the tensor dimension.
    name: str = betterproto.string_field(2)


@dataclass
class ValueCount(betterproto.Message):
    """
    Limits on maximum and minimum number of values in a single example (when
    the feature is present). Use this when the minimum value count can be
    different than the maximum value count. Otherwise prefer FixedShape.
    """

    min: int = betterproto.int64_field(1)
    max: int = betterproto.int64_field(2)


@dataclass
class WeightedFeature(betterproto.Message):
    """
    Represents a weighted feature that is encoded as a combination of raw base
    features. The `weight_feature` should be a float feature with identical
    shape as the `feature`. This is useful for representing weights associated
    with categorical tokens (e.g. a TFIDF weight associated with each token).
    TODO(b/142122960): Handle WeightedCategorical end to end in TFX
    (validation, TFX Unit Testing, etc)
    """

    # Name for the weighted feature. This should not clash with other features in
    # the same schema.
    name: str = betterproto.string_field(1)
    # Path of a base feature to be weighted. Required.
    feature: "Path" = betterproto.message_field(2)
    # Path of weight feature to associate with the base feature. Must be same
    # shape as feature. Required.
    weight_feature: "Path" = betterproto.message_field(3)
    # The lifecycle_stage determines where a feature is expected to be used, and
    # therefore how important issues with it are.
    lifecycle_stage: "LifecycleStage" = betterproto.enum_field(4)


@dataclass
class SparseFeature(betterproto.Message):
    """
    A sparse feature represents a sparse tensor that is encoded with a
    combination of raw features, namely index features and a value feature.
    Each index feature defines a list of indices in a different dimension.
    """

    # Name for the sparse feature. This should not clash with other features in
    # the same schema.
    name: str = betterproto.string_field(1)
    # This field is no longer supported. Instead, use: lifecycle_stage:
    # DEPRECATED TODO(b/111450258): remove this.
    deprecated: bool = betterproto.bool_field(2)
    # The lifecycle_stage determines where a feature is expected to be used, and
    # therefore how important issues with it are.
    lifecycle_stage: "LifecycleStage" = betterproto.enum_field(7)
    # Constraints on the presence of this feature in examples. Deprecated, this
    # is inferred by the referred features.
    presence: "FeaturePresence" = betterproto.message_field(4)
    # Shape of the sparse tensor that this SparseFeature represents. Currently
    # not supported. TODO(b/109669962): Consider deriving this from the referred
    # features.
    dense_shape: "FixedShape" = betterproto.message_field(5)
    # Features that represent indexes. Should be integers >= 0.
    index_feature: List["SparseFeatureIndexFeature"] = betterproto.message_field(6)
    # If true then the index values are already sorted lexicographically.
    is_sorted: bool = betterproto.bool_field(8)
    value_feature: "SparseFeatureValueFeature" = betterproto.message_field(9)
    # Type of value feature. Deprecated, this is inferred by the referred
    # features.
    type: "FeatureType" = betterproto.enum_field(10)


@dataclass
class SparseFeatureIndexFeature(betterproto.Message):
    # Name of the index-feature. This should be a reference to an existing
    # feature in the schema.
    name: str = betterproto.string_field(1)


@dataclass
class SparseFeatureValueFeature(betterproto.Message):
    # Name of the value-feature. This should be a reference to an existing
    # feature in the schema.
    name: str = betterproto.string_field(1)


@dataclass
class DistributionConstraints(betterproto.Message):
    """
    Models constraints on the distribution of a feature's values.
    TODO(martinz): replace min_domain_mass with max_off_domain (but slowly).
    """

    # The minimum fraction (in [0,1]) of values across all examples that should
    # come from the feature's domain, e.g.:   1.0  => All values must come from
    # the domain.    .9  => At least 90% of the values must come from the domain.
    min_domain_mass: float = betterproto.double_field(1)


@dataclass
class FeatureCoverageConstraints(betterproto.Message):
    """Encodes vocabulary coverage constraints."""

    # Fraction of feature values that map to a vocab entry (i.e. are not oov).
    min_coverage: float = betterproto.float_field(1)
    # Average length of tokens. Used for cases such as wordpiece that fallback to
    # character-level tokenization.
    min_avg_token_length: float = betterproto.float_field(2)
    # String tokens to exclude when calculating min_coverage and
    # min_avg_token_length. Useful for tokens such as [PAD].
    excluded_string_tokens: List[str] = betterproto.string_field(3)
    # Integer tokens to exclude when calculating min_coverage and
    # min_avg_token_length.
    excluded_int_tokens: List[int] = betterproto.int64_field(4)
    # String tokens to treat as oov tokens (e.g. [UNK]). These tokens are also
    # excluded when calculating avg token length.
    oov_string_tokens: List[str] = betterproto.string_field(5)


@dataclass
class SequenceValueConstraints(betterproto.Message):
    """Encodes constraints on specific values in sequences."""

    int_value: int = betterproto.int64_field(1, group="value")
    string_value: str = betterproto.string_field(2, group="value")
    # Min / max number of times the value can occur in a sequence.
    min_per_sequence: int = betterproto.int64_field(3)
    max_per_sequence: int = betterproto.int64_field(4)
    # Min / max fraction of sequences that must contain the value.
    min_fraction_of_sequences: float = betterproto.float_field(5)
    max_fraction_of_sequences: float = betterproto.float_field(6)


@dataclass
class SequenceLengthConstraints(betterproto.Message):
    """Encodes constraints on sequence lengths."""

    # Token values (int and string) that are excluded when calculating sequence
    # length.
    excluded_int_value: List[int] = betterproto.int64_field(1)
    excluded_string_value: List[str] = betterproto.string_field(2)
    # Min / max sequence length.
    min_sequence_length: int = betterproto.int64_field(3)
    max_sequence_length: int = betterproto.int64_field(4)


@dataclass
class IntDomain(betterproto.Message):
    """
    Encodes information for domains of integer values. Note that FeatureType
    could be either INT or BYTES.
    """

    # Id of the domain. Required if the domain is defined at the schema level. If
    # so, then the name must be unique within the schema.
    name: str = betterproto.string_field(1)
    # Min and max values for the domain.
    min: int = betterproto.int64_field(3)
    max: int = betterproto.int64_field(4)
    # If true then the domain encodes categorical values (i.e., ids) rather than
    # ordinal values.
    is_categorical: bool = betterproto.bool_field(5)


@dataclass
class FloatDomain(betterproto.Message):
    """
    Encodes information for domains of float values. Note that FeatureType
    could be either INT or BYTES.
    """

    # Id of the domain. Required if the domain is defined at the schema level. If
    # so, then the name must be unique within the schema.
    name: str = betterproto.string_field(1)
    # Min and max values of the domain.
    min: float = betterproto.float_field(3)
    max: float = betterproto.float_field(4)
    # If true, feature should not contain NaNs.
    disallow_nan: bool = betterproto.bool_field(5)
    # If true, feature should not contain Inf or -Inf.
    disallow_inf: bool = betterproto.bool_field(6)
    # If True, this indicates that the feature is semantically an embedding. This
    # can be useful for distinguishing fixed dimensional numeric features that
    # should be fed to a prediction unmodified.
    is_embedding: bool = betterproto.bool_field(7)


@dataclass
class StructDomain(betterproto.Message):
    """
    Domain for a recursive struct. NOTE: If a feature with a StructDomain is
    deprecated, then all the child features (features and sparse_features of
    the StructDomain) are also considered to be deprecated.  Similarly child
    features can only be in environments of the parent feature.
    """

    feature: List["Feature"] = betterproto.message_field(1)
    sparse_feature: List["SparseFeature"] = betterproto.message_field(2)


@dataclass
class StringDomain(betterproto.Message):
    """Encodes information for domains of string values."""

    # Id of the domain. Required if the domain is defined at the schema level. If
    # so, then the name must be unique within the schema.
    name: str = betterproto.string_field(1)
    # The values appearing in the domain.
    value: List[str] = betterproto.string_field(2)


@dataclass
class BoolDomain(betterproto.Message):
    """
    Encodes information about the domain of a boolean attribute that encodes
    its TRUE/FALSE values as strings, or 0=false, 1=true. Note that FeatureType
    could be either INT or BYTES.
    """

    # Id of the domain. Required if the domain is defined at the schema level. If
    # so, then the name must be unique within the schema.
    name: str = betterproto.string_field(1)
    # Strings values for TRUE/FALSE.
    true_value: str = betterproto.string_field(2)
    false_value: str = betterproto.string_field(3)


@dataclass
class NaturalLanguageDomain(betterproto.Message):
    """Natural language text."""

    # Name of the vocabulary associated with the NaturalLanguageDomain. When
    # computing and validating stats using TFDV, tfdv.StatsOptions.vocab_paths
    # should map this name to a vocabulary file.
    vocabulary: str = betterproto.string_field(1)
    coverage: "FeatureCoverageConstraints" = betterproto.message_field(2)
    token_constraints: List["SequenceValueConstraints"] = betterproto.message_field(3)
    sequence_length_constraints: "SequenceLengthConstraints" = betterproto.message_field(5)
    # Specifies the location constraints as a function of the tokens specified in
    # token_constraints. String tokens will be specified by S_TOKEN_, (e.g.
    # S_(PAD)_) and integer tokens will be specified as I_#_ (e.g. I_123_). A_T_
    # will match any token that has not been specified in token_constraints.
    # Parenthesis, +, and * are supported. _ will be escapable with a \ for
    # tokens containing it (e.g. FOO\_BAR). For example, a two-sequence BERT
    # model may look as follows: S_(CLS)_ A_T_+ S_(SEP)_ A_T_+ S_(SEP)_ S_(PAD)_*
    # Note: Support for this field is not yet implemented. Please do not use.
    # TODO(b/188095987): Remove warning once field is implemented.
    location_constraint_regex: str = betterproto.string_field(4)


@dataclass
class ImageDomain(betterproto.Message):
    """Image data."""

    # If set, at least this fraction of values should be TensorFlow supported
    # images.
    minimum_supported_image_fraction: float = betterproto.float_field(1)
    # If set, image should have less than this value of undecoded byte size.
    max_image_byte_size: int = betterproto.int64_field(2)


@dataclass
class MIDDomain(betterproto.Message):
    """Knowledge graph ID, see: https://www.wikidata.org/wiki/Property:P646"""


@dataclass
class URLDomain(betterproto.Message):
    """A URL, see: https://en.wikipedia.org/wiki/URL"""


@dataclass
class TimeDomain(betterproto.Message):
    """Time or date representation."""

    # Expected format that contains a combination of regular characters and
    # special format specifiers. Format specifiers are a subset of the strptime
    # standard.
    string_format: str = betterproto.string_field(1, group="format")
    # Expected format of integer times.
    integer_format: "TimeDomainIntegerTimeFormat" = betterproto.enum_field(2, group="format")


@dataclass
class TimeOfDayDomain(betterproto.Message):
    """Time of day, without a particular date."""

    # Expected format that contains a combination of regular characters and
    # special format specifiers. Format specifiers are a subset of the strptime
    # standard.
    string_format: str = betterproto.string_field(1, group="format")
    # Expected format of integer times.
    integer_format: "TimeOfDayDomainIntegerTimeOfDayFormat" = betterproto.enum_field(
        2, group="format"
    )


@dataclass
class FeaturePresence(betterproto.Message):
    """Describes constraints on the presence of the feature in the data."""

    # Minimum fraction of examples that have this feature.
    min_fraction: float = betterproto.double_field(1)
    # Minimum number of examples that have this feature.
    min_count: int = betterproto.int64_field(2)


@dataclass
class FeaturePresenceWithinGroup(betterproto.Message):
    """
    Records constraints on the presence of a feature inside a "group" context
    (e.g., .presence inside a group of features that define a sequence).
    """

    required: bool = betterproto.bool_field(1)


@dataclass
class InfinityNorm(betterproto.Message):
    """
    Checks that the L-infinity norm is below a certain threshold between the
    two discrete distributions. Since this is applied to a
    FeatureNameStatistics, it only considers the top k. L_infty(p,q) = max_i
    |p_i-q_i|
    """

    # The InfinityNorm is in the interval [0.0, 1.0] so sensible bounds should be
    # in the interval [0.0, 1.0).
    threshold: float = betterproto.double_field(1)


@dataclass
class JensenShannonDivergence(betterproto.Message):
    """
    Checks that the approximate Jensen-Shannon Divergence is below a certain
    threshold between the two distributions.
    """

    # The JensenShannonDivergence will be in the interval [0.0, 1.0] so sensible
    # bounds should be in the interval [0.0, 1.0).
    threshold: float = betterproto.double_field(1)


@dataclass
class FeatureComparator(betterproto.Message):
    infinity_norm: "InfinityNorm" = betterproto.message_field(1)
    jensen_shannon_divergence: "JensenShannonDivergence" = betterproto.message_field(2)


@dataclass
class UniqueConstraints(betterproto.Message):
    """
    Checks that the number of unique values is greater than or equal to the
    min, and less than or equal to the max.
    """

    min: int = betterproto.int64_field(1)
    max: int = betterproto.int64_field(2)


@dataclass
class TensorRepresentation(betterproto.Message):
    """
    A TensorRepresentation captures the intent for converting columns in a
    dataset to TensorFlow Tensors (or more generally, tf.CompositeTensors).
    Note that one tf.CompositeTensor may consist of data from multiple columns,
    for example, a N-dimensional tf.SparseTensor may need N + 1 columns to
    provide the sparse indices and values. Note that the "column name" that a
    TensorRepresentation needs is a string, not a Path -- it means that the
    column name identifies a top-level Feature in the schema (i.e. you cannot
    specify a Feature nested in a STRUCT Feature).
    """

    dense_tensor: "TensorRepresentationDenseTensor" = betterproto.message_field(1, group="kind")
    varlen_sparse_tensor: "TensorRepresentationVarLenSparseTensor" = betterproto.message_field(
        2, group="kind"
    )
    sparse_tensor: "TensorRepresentationSparseTensor" = betterproto.message_field(3, group="kind")
    ragged_tensor: "TensorRepresentationRaggedTensor" = betterproto.message_field(4, group="kind")


@dataclass
class TensorRepresentationDefaultValue(betterproto.Message):
    float_value: float = betterproto.double_field(1, group="kind")
    # Note that the data column might be of a shorter integral type. It's the
    # user's responsitiblity to make sure the default value fits that type.
    int_value: int = betterproto.int64_field(2, group="kind")
    bytes_value: bytes = betterproto.bytes_field(3, group="kind")
    # uint_value should only be used if the default value can't fit in a int64
    # (`int_value`).
    uint_value: int = betterproto.uint64_field(4, group="kind")


@dataclass
class TensorRepresentationDenseTensor(betterproto.Message):
    """A tf.Tensor"""

    # Identifies the column in the dataset that provides the values of this
    # Tensor.
    column_name: str = betterproto.string_field(1)
    # The shape of each row of the data (i.e. does not include the batch
    # dimension)
    shape: "FixedShape" = betterproto.message_field(2)
    # If this column is missing values in a row, the default_value will be used
    # to fill that row.
    default_value: "TensorRepresentationDefaultValue" = betterproto.message_field(3)


@dataclass
class TensorRepresentationVarLenSparseTensor(betterproto.Message):
    """A ragged tf.SparseTensor that models nested lists."""

    # Identifies the column in the dataset that should be converted to the
    # VarLenSparseTensor.
    column_name: str = betterproto.string_field(1)


@dataclass
class TensorRepresentationSparseTensor(betterproto.Message):
    """
    A tf.SparseTensor whose indices and values come from separate data columns.
    This will replace Schema.sparse_feature eventually. The index columns must
    be of INT type, and all the columns must co-occur and have the same valency
    at the same row.
    """

    # The dense shape of the resulting SparseTensor (does not include the batch
    # dimension).
    dense_shape: "FixedShape" = betterproto.message_field(1)
    # The columns constitute the coordinates of the values. indices_column[i][j]
    # contains the coordinate of the i-th dimension of the j-th value.
    index_column_names: List[str] = betterproto.string_field(2)
    # The column that contains the values.
    value_column_name: str = betterproto.string_field(3)


@dataclass
class TensorRepresentationRaggedTensor(betterproto.Message):
    """
    A tf.RaggedTensor that models nested lists. Currently there is no way for
    the user to specify the shape of the leaf value (the innermost value tensor
    of the RaggedTensor). The leaf value will always be a 1-D tensor.
    """

    # Identifies the leaf feature that provides values of the RaggedTensor.
    # struct type sub fields. The first step of the path refers to a top-level
    # feature in the data. The remaining steps refer to STRUCT features under the
    # top-level feature, recursively. If the feature has N outer ragged lists,
    # they will become the first N dimensions of the resulting RaggedTensor and
    # the contents will become the flat_values.
    feature_path: "Path" = betterproto.message_field(1)
    # The result RaggedTensor would be of shape: [B, D_0, D_1, ..., D_N, P_0,
    # P_1, ..., P_M, U_0, U_1, ..., U_P] Where the dimensions belong to different
    # categories: * B: Batch size dimension * D_n: Dimensions specified by the
    # nested structure specified by the value path until the leaf node. n>=1. *
    # P_m: Dimensions specified by the partitions that do not define any fixed
    # diomension size. m>=0. * U_0: Dimensions specified by the latest partitions
    # of type uniform_row_length that can define the fixed inner shape of the
    # tensor. If iterationg the partitions from the end to the beginning, these
    # dimensions are defined by all the continuous uniform_row_length partitions
    # present. p>=0.
    partition: List["TensorRepresentationRaggedTensorPartition"] = betterproto.message_field(3)
    # The data type of the ragged tensor's row partitions. This will default to
    # INT64 if it is not specified.
    row_partition_dtype: "TensorRepresentationRowPartitionDType" = betterproto.enum_field(2)


@dataclass
class TensorRepresentationRaggedTensorPartition(betterproto.Message):
    """Further partition of the feature values at the leaf level."""

    # If the final element(s) of partition are uniform_row_lengths [U0, U1, ...]
    # , then the result RaggedTensor will have  their flat values (a dense
    # tensor) being of shape [U0, U1, ...]. Otherwise, a uniform_row_length
    # simply means a ragged dimension with row_lengths
    # [uniform_row_length]*nrows.
    uniform_row_length: int = betterproto.int64_field(1, group="kind")
    # Identifies a leaf feature who share the same parent of value_feature_path
    # that contains the partition row lengths.
    row_length: str = betterproto.string_field(2, group="kind")


@dataclass
class TensorRepresentationGroup(betterproto.Message):
    """
    A TensorRepresentationGroup is a collection of TensorRepresentations with
    names. These names may serve as identifiers when converting the dataset to
    a collection of Tensors or tf.CompositeTensors. For example, given the
    following group: {   key: "dense_tensor"   tensor_representation {
    dense_tensor {       column_name: "univalent_feature"       shape {
    dim {           size: 1         }       }       default_value {
    float_value: 0       }     }   } } {   key: "varlen_sparse_tensor"
    tensor_representation {     varlen_sparse_tensor {       column_name:
    "multivalent_feature"     }   } } Then the schema is expected to have
    feature "univalent_feature" and "multivalent_feature", and when a batch of
    data is converted to Tensors using this TensorRepresentationGroup, the
    result may be the following dict: {   "dense_tensor": tf.Tensor(...),
    "varlen_sparse_tensor": tf.SparseTensor(...), }
    """

    tensor_representation: Dict[str, "TensorRepresentation"] = betterproto.map_field(
        1, betterproto.TYPE_STRING, betterproto.TYPE_MESSAGE
    )
