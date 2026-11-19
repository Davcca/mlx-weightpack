"""Locate MLX weight tensors without loading the quantized weight codes."""

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class Tensor:
    path: Path
    offset: int
    size: int
    shape: tuple[int, ...]
    dtype: str

    def raw16(self):
        """Preserve FP16 and BF16 bit patterns, including signed zero and NaN payloads."""
        if self.dtype not in {"F16", "BF16"}:
            raise ValueError(f"Expected 16-bit floating point, found {self.dtype}")
        return np.memmap(self.path, dtype="<u2", mode="r", offset=self.offset, shape=self.shape)


def model_files(source):
    source = Path(source)
    if source.is_file() and source.suffix == ".safetensors":
        return [source]
    if not source.is_dir():
        raise ValueError(f"Expected a checkpoint directory or a .safetensors file: {source}")
    single = source / "model.safetensors"
    pattern = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
    shards = sorted(p for p in source.glob("model-*-of-*.safetensors") if pattern.fullmatch(p.name))
    if single.is_file() and shards:
        raise ValueError(
            "Both a single model and numbered shards exist. Pass a single file or separate the checkpoints."
        )
    if single.is_file():
        return [single]
    if shards:
        parts = [(int(m[1]), int(m[2])) for p in shards if (m := pattern.fullmatch(p.name))]
        if parts != [(i, len(shards)) for i in range(1, len(shards) + 1)]:
            raise ValueError(f"Incomplete set of numbered model shards in {source}")
        return shards
    index = source / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if not names or any(
            Path(name).name != name or not name.endswith(".safetensors") for name in names
        ):
            raise ValueError("Invalid shard name in the checkpoint index")
        return [source / name for name in names]
    raise ValueError(f"No model.safetensors or numbered model shards in {source}")


@dataclass(frozen=True)
class Matrix:
    name: str
    scales: Tensor
    biases: Tensor
    weight: Tensor


class Checkpoint:
    def __init__(self, source):
        self.files = model_files(source)
        self.tensors = {}
        for path in self.files:
            # The Rust reader validates dtypes, dimensions, offsets and file length.
            with safe_open(path, framework="numpy"), path.open("rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors:
                    raise ValueError(f"Duplicate tensor across shards: {name}")
                start, end = entry["data_offsets"]
                self.tensors[name] = Tensor(
                    path,
                    8 + header_size + start,
                    end - start,
                    tuple(entry["shape"]),
                    entry["dtype"],
                )
        self.tensor_bytes = sum(t.size for t in self.tensors.values())

    def matrices(self):
        for key, scales in sorted(self.tensors.items()):
            if not key.endswith(".scales"):
                continue
            name = key.removesuffix(".scales")
            biases = self.tensors.get(name + ".biases")
            weight = self.tensors.get(name + ".weight")
            if biases is None or weight is None or weight.dtype != "U32" or len(weight.shape) != 2:
                continue
            if scales.dtype not in {"F16", "BF16"} or len(scales.shape) != 2:
                continue
            if (
                scales.shape != biases.shape
                or scales.dtype != biases.dtype
                or scales.shape[0] != weight.shape[0]
            ):
                raise ValueError(f"{name}: incompatible weight, scale or bias layout")
            if scales.size:
                yield Matrix(name, scales, biases, weight)
