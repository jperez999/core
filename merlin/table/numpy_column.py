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
from merlin.core.compat import numpy as np
from merlin.table.conversions import (
    _from_array_interface,
    _from_dlpack_cpu,
    _to_array_interface,
    _to_dlpack,
)
from merlin.table.tensor_column import Device, TensorColumn


class NumpyColumn(TensorColumn):
    @classmethod
    def array_type(cls):
        return np.ndarray

    @classmethod
    def supported_devices(cls):
        return [Device.CPU]

    @classmethod
    def cast(cls, other):
        column = cls(to_numpy(other.values), to_numpy(other.offsets))
        column._ref = (other.values, other.offsets)
        return column

    def __init__(self, values: np.ndarray, offsets: np.ndarray = None, dtype=None):
        super().__init__(values, offsets, dtype)

    @property
    def device(self) -> Device:
        return Device.CPU


@_to_array_interface.register_lazy("numpy")
def register_to_array_interface_numpy():
    import numpy as np

    @_to_array_interface.register(np.ndarray)
    def _to_array_interface_numpy(array):
        return array


@_from_array_interface.register_lazy("numpy")
def register_from_array_interface_numpy():
    import numpy as np

    @_from_array_interface.register(np.ndarray)
    def _from_array_interface_numpy(target_type, array_interface):
        return np.asarray(array_interface)


@_from_dlpack_cpu.register_lazy("numpy")
def register_from_dlpack_cpu_to_numpy():
    import numpy as np

    @_from_dlpack_cpu.register(np.ndarray)
    def _from_dlpack_cpu_to_numpy(to, array):
        try:
            return np._from_dlpack(array)
        except AttributeError as exc:
            raise NotImplementedError(
                "NumPy does not implement the DLPack Standard until version 1.22.0, "
                f"currently running {np.__version__}"
            ) from exc


@_to_dlpack.register_lazy("numpy")
def register_from_numpy_to_dlpack_cpu():
    import numpy as np

    @_to_dlpack.register(np.ndarray)
    def _to_dlpack_cpu_from_numpy(array):
        return array.__dlpack__()
