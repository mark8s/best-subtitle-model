#!/usr/bin/env python3
"""
把 facebook/nllb-200-distilled-600M 导出为 OpenVINO IR（Intel GPU 翻译用）。

原模型与现在的 CTranslate2 nllb-600m-int8 同源，权重格式不同，不能混用。
在 /DATA/ov-venv 里运行，不要用字幕项目的 .venv。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = "facebook/nllb-200-distilled-600M"
DEFAULT_OUT = ROOT / "models" / "openvino-nllb-600m-int8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NLLB-200 Distilled 600M to OpenVINO IR.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Hugging Face model id or local path (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--weight-format",
        choices=["int8", "fp16"],
        default="int8",
        help="Weight format (default: int8)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output if args.output.is_absolute() else ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer
        from optimum.intel import OVModelForSeq2SeqLM
    except ImportError:
        print(
            "[error] 未安装 Optimum Intel。请激活 ov-venv 后执行:\n"
            "  python -m pip install -r requirements-openvino-translate.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Source     : {args.source}")
    print(f"Output     : {out_dir}")
    print(f"Weights    : {args.weight_format}")
    print("Downloading / converting ... (首次约数 GB，请等)")

    tokenizer = AutoTokenizer.from_pretrained(args.source)
    export_kwargs: dict = {"export": True}
    if args.weight_format == "int8":
        export_kwargs["quantization_config"] = {"bits": 8}

    try:
        model = OVModelForSeq2SeqLM.from_pretrained(args.source, **export_kwargs)
    except Exception as exc:
        if args.weight_format != "int8":
            raise
        print(
            f"[warn] INT8 导出失败，改为 FP16: {exc}",
            file=sys.stderr,
        )
        model = OVModelForSeq2SeqLM.from_pretrained(args.source, export=True)

    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saved OpenVINO NLLB to {out_dir}")
    print("Next:")
    print(
        "  python translate_srt_openvino.py "
        "-i input/xxx.srt --src jpn_Jpan --tgt zho_Hans --device GPU"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
