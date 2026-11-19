"""Write and verify .wpack companions. Weight codes stay in the source checkpoint."""

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from .checkpoint import Checkpoint
from .codec import MAX_ENTRIES, check_pairs, digest, encode, stream_words, table_for, width_for

FORMAT = "mlx-weightpack"
VERSION = 1


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: str
    shape: tuple[int, int]
    weight_shape: tuple[int, int]
    entries: int
    source_sha256: str
    payload_sha256: str

    @property
    def groups(self):
        return math.prod(self.shape)

    @property
    def width(self):
        return width_for(self.entries)


def put_member(z, name, data):
    # Fixed ZIP metadata makes two packs of the same checkpoint byte-identical.
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = 0o600 << 16
    z.writestr(info, memoryview(data).cast("B"))


def process(source, z=None, progress=None):
    checkpoint = Checkpoint(source)
    matrices = list(checkpoint.matrices())
    entries = []
    original = stored = encoded = groups = 0
    for i, matrix in enumerate(matrices, 1):
        if progress:
            progress(i, len(matrices), matrix.name)
        scales, biases = matrix.scales.raw16(), matrix.biases.raw16()
        table = table_for(scales, biases)
        raw = scales.nbytes + biases.nbytes
        groups += scales.size
        original += raw
        size = raw
        if table is not None:
            size = table.nbytes + 4 * stream_words(scales.size, width_for(table.size))
        if size >= raw:
            stored += raw
            continue
        stored += size
        encoded += 1
        if z is None:
            continue
        words = encode(scales, biases, table)
        check_pairs(scales, biases, table, words)
        slot = f"matrices/{len(entries):06d}"
        put_member(z, slot + ".table", table)
        put_member(z, slot + ".ids", words)
        entries.append(
            Entry(
                matrix.name,
                matrix.scales.dtype,
                matrix.scales.shape,
                matrix.weight.shape,
                int(table.size),
                digest(scales, biases),
                digest(table, words),
            )
        )
    result = {
        "matrices": len(matrices),
        "packed_matrices": encoded,
        "raw_matrices": len(matrices) - encoded,
        "groups": groups,
        "source_tensor_bytes": checkpoint.tensor_bytes,
        "original_metadata_bytes": original,
        "packed_metadata_bytes": stored,
        "saved_bytes": original - stored,
    }
    if z is not None:
        if not encoded:
            raise ValueError("No eligible matrix gets smaller. No pack was written.")
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "summary": result,
            "matrices": [asdict(e) for e in entries],
        }
        put_member(z, "manifest.json", json.dumps(manifest, separators=(",", ":")).encode())
    return result


def scan(source, progress=None):
    return process(source, progress=progress)


def pack(source, output, progress=None):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".weightpack-", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with ZipFile(temp, "w", compression=ZIP_STORED) as z:
            result = process(source, z, progress)
        # Publish only a complete file. A concurrent writer cannot replace it.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return result


class Archive:
    def __init__(self, path):
        self.z = ZipFile(path, "r")
        try:
            self._read_index()
        except BaseException:
            self.z.close()
            raise

    def _read_index(self):
        files = self.z.infolist()
        if len({f.filename for f in files}) != len(files):
            raise ValueError("Duplicate archive member")
        info = self.z.getinfo("manifest.json")
        if info.compress_type != ZIP_STORED or info.file_size > 32 * 1024 * 1024:
            raise ValueError("Invalid weightpack manifest")
        manifest = json.loads(self.z.read(info))
        if not isinstance(manifest, dict):
            raise TypeError("Expected a JSON object in the weightpack manifest")
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError("Unknown weightpack format or version")
        if not isinstance(manifest.get("matrices"), list):
            raise TypeError("Missing matrix list in weightpack manifest")
        self.entries = {}
        self.slots = {}
        for i, record in enumerate(manifest["matrices"]):
            entry = Entry(
                **{
                    **record,
                    "shape": tuple(record["shape"]),
                    "weight_shape": tuple(record["weight_shape"]),
                }
            )
            if (
                not isinstance(entry.name, str)
                or not entry.name
                or entry.name in self.entries
                or entry.dtype not in {"F16", "BF16"}
                or len(entry.shape) != 2
                or len(entry.weight_shape) != 2
                or any(type(n) is not int or n <= 0 for n in (*entry.shape, *entry.weight_shape))
                or entry.weight_shape[0] != entry.shape[0]
                or type(entry.entries) is not int
                or not 1 <= entry.entries <= min(MAX_ENTRIES, entry.groups)
                or not all(
                    re.fullmatch("[0-9a-f]{64}", h)
                    for h in (entry.source_sha256, entry.payload_sha256)
                )
            ):
                raise ValueError("Invalid matrix record in weightpack manifest")
            slot = f"matrices/{i:06d}"
            for ext, length in (
                (".table", entry.entries),
                (".ids", stream_words(entry.groups, entry.width)),
            ):
                item = self.z.getinfo(slot + ext)
                if item.compress_type != ZIP_STORED or item.file_size != length * 4:
                    raise ValueError(f"{entry.name}: invalid {ext} data length or encoding")
            self.entries[entry.name] = entry
            self.slots[entry.name] = slot
        if not self.entries or len(files) != 1 + 2 * len(self.entries):
            raise ValueError("Missing or unexpected weightpack members")

    def arrays(self, entry):
        slot = self.slots[entry.name]
        table = np.frombuffer(self.z.read(slot + ".table"), dtype="<u4")
        words = np.frombuffer(self.z.read(slot + ".ids"), dtype="<u4")
        if digest(table, words) != entry.payload_sha256:
            raise ValueError(f"{entry.name}: packed data checksum differs")
        return table, words

    def checked_arrays(self, entry, scales, biases):
        if (
            scales.shape != entry.shape
            or biases.shape != entry.shape
            or digest(scales, biases) != entry.source_sha256
        ):
            raise ValueError(
                f"{entry.name}: scale or bias differs from the source used to build this pack"
            )
        table, words = self.arrays(entry)
        try:
            check_pairs(scales, biases, table, words)
        except ValueError as exc:
            raise ValueError(f"{entry.name}: {exc}") from exc
        return table, words

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.z.close()


def verify(source, sidecar, progress=None):
    matrices = {m.name: m for m in Checkpoint(source).matrices()}
    groups = 0
    with Archive(sidecar) as archive:
        count = len(archive.entries)
        for i, entry in enumerate(archive.entries.values(), 1):
            if progress:
                progress(i, count, entry.name)
            matrix = matrices.get(entry.name)
            if (
                matrix is None
                or matrix.scales.dtype != entry.dtype
                or matrix.weight.shape != entry.weight_shape
            ):
                raise ValueError(f"{entry.name}: no matching source matrix")
            archive.checked_arrays(entry, matrix.scales.raw16(), matrix.biases.raw16())
            groups += entry.groups
    return {"verified_matrices": count, "verified_groups": groups, "bit_identical": True}
