import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from safetensors.numpy import load_file, save_file

from weightpack.archive import Archive, pack, scan, verify
from weightpack.checkpoint import Checkpoint
from weightpack.demo import write_example


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_example(self.root)
        self.source = self.root / "model.safetensors"
        self.dest = self.root / "checkpoint.wpack"

    def test_portable_cli_round_trip(self):
        def cli(*args):
            return subprocess.run(
                [sys.executable, "-m", "weightpack", *map(str, args)],
                capture_output=True,
                text=True,
                check=False,
            )

        for args in [
            ("demo",),
            ("scan", self.root),
            ("pack", self.root, "-o", self.dest),
            ("verify", self.source, self.dest),
        ]:
            proc = cli(*args, "--json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertTrue(result.get("bit_identical") or result["saved_bytes"] > 0)
        before = self.dest.read_bytes()
        self.assertNotEqual(cli("pack", self.source, "-o", self.dest).returncode, 0)
        self.assertEqual(before, self.dest.read_bytes())

    def test_deterministic_archives_and_source_integrity(self):
        before = self.source.read_bytes()
        pack(self.root, self.dest)
        second = self.root / "b.wpack"
        pack(self.source, second)
        self.assertEqual(self.dest.read_bytes(), second.read_bytes())
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(scan(self.root)["packed_metadata_bytes"], 2116)
        self.assertEqual(verify(self.root, self.dest)["verified_groups"], 4096)

    def test_wrong_checkpoint_or_changed_payload(self):
        pack(self.root, self.dest)
        weights = load_file(self.source)
        weights["layer.scales"][0, 0] *= -1
        save_file(weights, self.source)
        with self.assertRaisesRegex(ValueError, "differs"):
            verify(self.root, self.dest)
        write_example(self.root)
        bad = self.root / "changed.wpack"
        with ZipFile(self.dest) as src, ZipFile(bad, "w") as dst:
            for name in src.namelist():
                data = bytearray(src.read(name))
                if name.endswith(".ids"):
                    data[0] ^= 1
                dst.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify(self.root, bad)

    def test_incomplete_archive_and_shard_set(self):
        with ZipFile(self.dest, "w") as z:
            z.writestr("manifest.json", json.dumps({"format": "mlx-weightpack", "version": 2}))
        with self.assertRaisesRegex(ValueError, "version"):
            Archive(self.dest)
        self.source.rename(self.root / "model-00001-of-00002.safetensors")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            scan(self.root)

    def test_tensors_spanning_shards_and_separate_vision_file(self):
        data = load_file(self.source)
        self.source.rename(self.root / "model-vision.safetensors")
        save_file(
            {"layer.weight": data.pop("layer.weight")},
            self.root / "model-00001-of-00002.safetensors",
        )
        save_file(data, self.root / "model-00002-of-00002.safetensors")
        self.assertEqual(pack(self.root, self.dest)["matrices"], 1)
        self.assertEqual(verify(self.root, self.dest)["verified_groups"], 4096)
        shutil.copyfile(
            self.root / "model-00002-of-00002.safetensors",
            self.root / "model-00001-of-00002.safetensors",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            Checkpoint(self.root)

    def test_bfloat16_raw_payload(self):
        # F16 and BF16 have the same byte length. Retag the fixture and give
        # one BF16 value a NaN payload which float conversion could canonicalize.
        content = self.source.read_bytes()
        size = struct.unpack_from("<Q", content)[0]
        header = json.loads(content[8 : 8 + size])
        raw = bytearray(content[8 + size :])
        for name in ("layer.scales", "layer.biases"):
            header[name]["dtype"] = "BF16"
            struct.pack_into("<H", raw, header[name]["data_offsets"][0], 0x7FC1)
        h = json.dumps(header, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        self.source.write_bytes(struct.pack("<Q", len(h)) + h + raw)
        pack(self.root, self.dest)
        self.assertTrue(verify(self.root, self.dest)["bit_identical"])
        with Archive(self.dest) as archive:
            self.assertEqual(archive.entries["layer"].dtype, "BF16")

    def test_unprofitable_matrices_and_atomic_failure(self):
        data = load_file(self.source)
        for name, value in list(data.items()):
            data["small." + name] = value[:1, :1].copy()
        save_file(data, self.source)
        summary = pack(self.root, self.dest)
        self.assertEqual((summary["packed_matrices"], summary["raw_matrices"]), (1, 1))
        other = self.root / "no.wpack"
        save_file({k: v for k, v in data.items() if k.startswith("small.")}, self.source)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            pack(self.root, other)
        self.assertFalse(other.exists())
        self.assertEqual(list(self.root.glob(".weightpack-*")), [])
