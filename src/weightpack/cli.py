"""Command-line interface for Weightpack."""

import argparse
import json
import platform
import sys
from pathlib import Path
from zipfile import BadZipFile

from safetensors import SafetensorError

from . import __version__
from .archive import pack, scan, verify
from .demo import demo

CREDITS = "Small-row dot product: Powered by MTPLX (https://github.com/youssofal/MTPLX)."


def human_bytes(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    if n >= 1000:
        return f"{n / 1000:.2f} kB"
    return f"{n} B"


def progress(current, total, name):
    if sys.stderr.isatty() and (current == 1 or current % 25 == 0 or current == total):
        print(f"[{current}/{total}]  {name}", file=sys.stderr)


def parser():
    root = argparse.ArgumentParser(
        prog="weightpack",
        description="Pack repeated scale and bias values in MLX checkpoints.",
        epilog=CREDITS,
    )
    root.add_argument("--version", action="version", version=f"weightpack {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("demo", "Pack and verify a tiny checkpoint offline"),
        ("scan", "Estimate metadata savings without writing a file"),
        ("pack", "Write a verified .wpack companion file"),
        ("verify", "Compare every packed pair with the original checkpoint"),
        ("bench", "Time a single projection against stock MLX on this Mac"),
        ("generate", "Generate text with a local MLX-LM checkpoint"),
    ):
        cmd = commands.add_parser(name, help=help_text, description=help_text)
        if name != "demo":
            cmd.add_argument(
                "model",
                type=Path,
                help="Local MLX model directory"
                + (" or Safetensors file" if name in {"scan", "pack", "verify", "bench"} else ""),
            )
        if name == "pack":
            cmd.add_argument("-o", "--output", required=True, type=Path, help="New .wpack path")
        if name in {"verify", "bench"}:
            cmd.add_argument("sidecar", type=Path, help="Companion file written by weightpack pack")
        if name == "bench":
            cmd.add_argument(
                "--matrix", help="Full matrix name (default: first packed gate projection)"
            )
            cmd.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16])
            cmd.add_argument("--repeats", type=int, default=20)
        if name == "generate":
            cmd.add_argument(
                "--pack", type=Path, help="Use this .wpack companion (omit for stock MLX)"
            )
            cmd.add_argument("--prompt", required=True)
            cmd.add_argument("--max-tokens", type=int, default=64)
            cmd.add_argument(
                "--raw", action="store_true", help="Pass the prompt directly to the tokenizer"
            )
        else:
            cmd.add_argument(
                "--json", action="store_true", help="Write machine-readable results to stdout"
            )
    return root


def show(result):
    if "matrix" in result:
        print(result["matrix"])
        print("Rows       Stock        Packed    Output bits")
        for row in result["timings"]:
            print(
                f"{row['rows']:>4}   {row['stock_ms']:>7.3f} ms   {row['packed_ms']:>7.3f} ms    identical"
            )
        return
    if "saved_bytes" in result:
        print(f"Packed matrices    {result['packed_matrices']:,} of {result['matrices']:,}")
        print(f"Original metadata  {human_bytes(result['original_metadata_bytes'])}")
        print(f"Packed metadata    {human_bytes(result['packed_metadata_bytes'])}")
        print(f"Payload saved      {human_bytes(result['saved_bytes'])}")
        if result["matrices"] == 0:
            print("No paired affine metadata found in these weights.")
    if result.get("bit_identical"):
        print(f"Verified           {result['verified_groups']:,} scale/bias pairs, bit for bit")
    if "file_bytes" in result:
        print(f"Companion on disk  {human_bytes(result['file_bytes'])}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            result = demo()
        elif args.command == "scan":
            result = scan(args.model, progress)
        elif args.command == "pack":
            result = pack(args.model, args.output, progress)
            result["file_bytes"] = args.output.stat().st_size
        elif args.command == "verify":
            result = verify(args.model, args.sidecar, progress)
        else:
            if sys.platform != "darwin" or platform.machine() != "arm64":
                raise ValueError(
                    "bench and generate require an Apple Silicon Mac. The archive commands run on this platform."
                )
            try:
                from .inference import benchmark, generate_text
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] in {"mlx", "mlx_lm"}:
                    raise ValueError(
                        "Install the Metal extra from the checkout: python -m pip install '.[metal]'"
                    ) from exc
                raise
            if args.command == "generate":
                generate_text(args.model, args.pack, args.prompt, args.max_tokens, args.raw)
                return 0
            result = benchmark(args.model, args.sidecar, args.matrix, args.rows, args.repeats)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        BadZipFile,
        SafetensorError,
    ) as exc:
        print(f"weightpack: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if args.command == "demo":
            print("Offline demo · synthetic Q4 weights")
        show(result)
    return 0
