#!/usr/bin/env python3
"""
字幕翻译测试脚本（SRT → SRT）

通过 config.yaml 管理多个本地模型：改 active_model 或 --model-id 即可切换。
当前内置后端：CTranslate2 NLLB（type: ctranslate2-nllb）。

典型流程：
  1. 读取配置，解析要用的模型
  2. 加载模型 + SentencePiece 分词器
  3. 解析 SRT，抽出每条字幕文本
  4. 按 batch 送进模型翻译（时间轴不动）
  5. 写回新的 SRT，并打印耗时统计
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import ctranslate2
import sentencepiece as spm
import srt
import yaml
from tqdm import tqdm

# 项目根目录 = 本脚本所在目录（config.yaml / models / input 都相对这里）
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# NLLB-200 常用语言码（完整列表见 FLORES-200）
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
# ---------------------------------------------------------------------------
COMMON_LANGS = """
  eng_Latn  English
  zho_Hans  Chinese (Simplified)
  zho_Hant  Chinese (Traditional)
  jpn_Jpan  Japanese
  kor_Hang  Korean
  fra_Latn  French
  deu_Latn  German
  spa_Latn  Spanish
  rus_Cyrl  Russian
  por_Latn  Portuguese
  ita_Latn  Italian
  vie_Latn  Vietnamese
  tha_Thai  Thai
  ara_Arab  Arabic
  hin_Deva  Hindi
""".strip()

# 目前支持的模型 type；以后加新后端时在这里扩展
SUPPORTED_TYPES = {"ctranslate2-nllb"}


def format_duration(seconds: float) -> str:
    """把秒数格式化成易读字符串，例如 1.23s / 1m 05.2s。"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:04.1f}s"


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置；文件不存在或内容为空时返回空 dict。"""
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return data


def resolve_model_path(raw: str | Path, base_dir: Path) -> Path:
    """相对路径按配置文件所在目录解析；绝对路径原样使用。"""
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def list_models(config: dict[str, Any]) -> None:
    """打印 config 里已注册的模型，方便查看能切换哪些。"""
    models = config.get("models") or {}
    active = config.get("active_model")
    if not models:
        print("No models registered in config.")
        return
    print("Registered models:")
    for mid, meta in models.items():
        mark = "*" if mid == active else " "
        name = (meta or {}).get("name") or mid
        mtype = (meta or {}).get("type") or "?"
        path = (meta or {}).get("path") or "?"
        print(f"  {mark} {mid}")
        print(f"      name : {name}")
        print(f"      type : {mtype}")
        print(f"      path : {path}")
    print()
    print("* = active_model in config.yaml")


def pick_model_settings(
    config: dict[str, Any],
    model_id: str | None,
    model_path_override: Path | None,
    config_dir: Path,
) -> dict[str, Any]:
    """
    根据配置选出本次运行的模型参数。

    返回字段包括：id / name / type / path / src / tgt / device / ...
    """
    defaults = dict(config.get("defaults") or {})
    models = config.get("models") or {}

    # 1) 确定 model id：命令行 --model-id > config.active_model
    mid = model_id or config.get("active_model")
    if not mid and not model_path_override:
        raise SystemExit(
            "[error] 未指定模型。请在 config.yaml 设置 active_model，"
            "或使用 --model-id / --model-path"
        )

    meta: dict[str, Any] = {}
    if mid:
        if mid not in models:
            known = ", ".join(models) or "(none)"
            raise SystemExit(f"[error] unknown model id '{mid}'. Known: {known}")
        meta = dict(models[mid] or {})
    else:
        # 只给了裸路径、没走注册表时，按默认 NLLB 后端处理
        mid = "_cli_path"
        meta = {"name": str(model_path_override), "type": "ctranslate2-nllb"}

    # 2) 合并：defaults <- 模型配置（模型覆盖全局默认）
    merged = {**defaults, **meta}

    # 3) 解析模型目录
    if model_path_override is not None:
        path = model_path_override.resolve()
    else:
        raw_path = merged.get("path")
        if not raw_path:
            raise SystemExit(f"[error] model '{mid}' has no path in config")
        path = resolve_model_path(raw_path, config_dir)

    mtype = merged.get("type") or "ctranslate2-nllb"
    if mtype not in SUPPORTED_TYPES:
        raise SystemExit(
            f"[error] unsupported model type '{mtype}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_TYPES))}"
        )

    return {
        "id": mid,
        "name": merged.get("name") or mid,
        "type": mtype,
        "path": path,
        "src": merged.get("src") or "eng_Latn",
        "tgt": merged.get("tgt") or "zho_Hans",
        "device": merged.get("device") or "auto",
        "compute_type": merged.get("compute_type") or "auto",
        "beam_size": int(merged.get("beam_size") or 2),
        "batch_size": int(merged.get("batch_size") or 32),
        "input": Path(merged.get("input") or "input"),
        "output": Path(merged.get("output") or "output"),
        "suffix": str(merged.get("suffix") or ""),
    }


class NllbTranslator:
    """封装 CTranslate2 NLLB 模型的加载与批量翻译。"""

    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        compute_type: str = "auto",
        inter_threads: int = 1,
        intra_threads: int = 0,
    ) -> None:
        model_dir = model_dir.resolve()
        sp_path = model_dir / "sentencepiece.bpe.model"

        # 启动前先检查关键文件，避免加载到一半才报错
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(f"model.bin not found in {model_dir}")
        if not sp_path.exists():
            raise FileNotFoundError(f"sentencepiece.bpe.model not found in {model_dir}")

        # auto：有 CUDA 就用 GPU，否则 CPU
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

        # SentencePiece：把原文切成 NLLB 认识的子词 token
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_path))

        # Translator：真正跑推理的引擎（INT8 量化权重在 model.bin 里）
        # compute_type=auto 时，若 CPU 不支持 int8_float16，会自动降到 int8_float32
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self.device = device

    def translate_texts(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        beam_size: int = 2,
        batch_size: int = 32,
        max_decoding_length: int = 256,
    ) -> tuple[list[str], dict[str, float | int]]:
        """
        翻译一组纯文本，返回 (译文列表, 耗时统计)。

        NLLB + CTranslate2 输入格式约定：
          source = [源语言码] + SentencePiece子词 + </s>
          target_prefix = [目标语言码]   # 告诉解码器要生成哪种语言
        """
        stats: dict[str, float | int] = {
            "num_cues": len(texts),
            "num_translated": 0,
            "encode_s": 0.0,
            "infer_s": 0.0,
            "decode_s": 0.0,
            "total_s": 0.0,
        }
        if not texts:
            return [], stats

        t_all = time.perf_counter()

        # 空行 / 纯空白字幕不送模型，原样保留，节省时间
        outputs: list[str | None] = [None] * len(texts)
        nonempty_idx: list[int] = []
        source_tokens: list[list[str]] = []

        # ---- 分词（encode）----
        t0 = time.perf_counter()
        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                outputs[i] = text
                continue
            pieces = self.sp.encode(stripped, out_type=str)
            source_tokens.append([src_lang] + pieces + ["</s>"])
            nonempty_idx.append(i)
        stats["encode_s"] = time.perf_counter() - t0
        stats["num_translated"] = len(source_tokens)

        if not source_tokens:
            stats["total_s"] = time.perf_counter() - t_all
            return [o if o is not None else "" for o in outputs], stats

        # 每条输入都要带上目标语言前缀
        target_prefix = [[tgt_lang]] * len(source_tokens)

        # ---- 模型推理（最耗时的部分）----
        t0 = time.perf_counter()
        results = self.translator.translate_batch(
            source_tokens,
            target_prefix=target_prefix,
            beam_size=beam_size,  # 1=更快；2~4=质量更好、更慢
            batch_type="examples",
            max_batch_size=batch_size,  # 一次送多少条；显存/内存不够就调小
            max_decoding_length=max_decoding_length,
        )
        stats["infer_s"] = time.perf_counter() - t0

        # ---- 解码（decode）：token → 可读字符串 ----
        t0 = time.perf_counter()
        for idx, result in zip(nonempty_idx, results):
            hyp = result.hypotheses[0]  # beam 里取最优一条
            # 模型输出通常以目标语言码开头，需要去掉
            if hyp and hyp[0] == tgt_lang:
                hyp = hyp[1:]
            # 过滤特殊 token，避免写出 <unk> 或乱码替换符
            hyp = [t for t in hyp if t not in {"<unk>", "<s>", "</s>"}]
            outputs[idx] = self.sp.decode(hyp)
        stats["decode_s"] = time.perf_counter() - t0
        stats["total_s"] = time.perf_counter() - t_all

        return [o if o is not None else "" for o in outputs], stats


def build_translator(settings: dict[str, Any]) -> NllbTranslator:
    """按 type 创建翻译器；以后加其他后端在这里分支即可。"""
    mtype = settings["type"]
    if mtype == "ctranslate2-nllb":
        return NllbTranslator(
            model_dir=settings["path"],
            device=settings["device"],
            compute_type=settings["compute_type"],
        )
    raise SystemExit(f"[error] unsupported model type: {mtype}")


def read_srt(path: Path) -> list[srt.Subtitle]:
    """读取 SRT；utf-8-sig 可兼容带 BOM 的文件。"""
    content = path.read_text(encoding="utf-8-sig")
    return list(srt.parse(content))


def write_srt(path: Path, subs: list[srt.Subtitle]) -> None:
    """写出 SRT（UTF-8），必要时自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(srt.compose(subs), encoding="utf-8")


def translate_srt_file(
    translator: NllbTranslator,
    input_path: Path,
    output_path: Path,
    src_lang: str,
    tgt_lang: str,
    beam_size: int,
    batch_size: int,
) -> dict[str, float | int]:
    """
    翻译单个 SRT 文件：时间码/序号不变，只替换 content。
    返回本文件的耗时统计，方便上层汇总打印。
    """
    t_file = time.perf_counter()

    t0 = time.perf_counter()
    subs = read_srt(input_path)
    parse_s = time.perf_counter() - t0

    texts = [sub.content for sub in subs]
    translated, model_stats = translator.translate_texts(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        beam_size=beam_size,
        batch_size=batch_size,
    )

    for sub, text in zip(subs, translated):
        sub.content = text.strip()

    t0 = time.perf_counter()
    write_srt(output_path, subs)
    write_s = time.perf_counter() - t0

    file_total = time.perf_counter() - t_file
    return {
        **model_stats,
        "parse_s": parse_s,
        "write_s": write_s,
        "file_total_s": file_total,
    }


def collect_srt_files(input_dir: Path) -> list[Path]:
    """递归收集目录下所有 .srt（含子目录）。"""
    files = sorted(input_dir.rglob("*.srt"))
    return [p for p in files if p.is_file()]


def print_file_timing(path: Path, stats: dict[str, float | int]) -> None:
    """打印单个文件的耗时明细，方便对比瓶颈在哪一段。"""
    num_cues = int(stats["num_cues"])
    num_translated = int(stats["num_translated"])
    file_total = float(stats["file_total_s"])
    infer_s = float(stats["infer_s"])
    cps = (num_translated / infer_s) if infer_s > 0 else 0.0

    print(f"  file      : {path.name}")
    print(f"  cues      : {num_cues} total, {num_translated} translated")
    print(
        f"  timing    : total {format_duration(file_total)} "
        f"(parse {format_duration(float(stats['parse_s']))}, "
        f"encode {format_duration(float(stats['encode_s']))}, "
        f"infer {format_duration(infer_s)}, "
        f"decode {format_duration(float(stats['decode_s']))}, "
        f"write {format_duration(float(stats['write_s']))})"
    )
    if num_translated > 0 and infer_s > 0:
        print(f"  speed     : {cps:.1f} cues/s (infer only)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    命令行参数。

    注意：多数选项 default=None，表示「未在命令行指定」，
    这样才会回落到 config.yaml 里的模型配置 / defaults。
    """
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files with local models (config.yaml).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Common language codes:\n{COMMON_LANGS}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model id in config.yaml (overrides active_model)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Direct model directory path (overrides config path)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models registered in config and exit",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input SRT file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output SRT file or directory",
    )
    parser.add_argument("--src", default=None, help="Source language code")
    parser.add_argument("--tgt", default=None, help="Target language code")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2 compute type, e.g. auto / int8 / float32",
    )
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional suffix before .srt, e.g. .zh -> movie.zh.srt",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """命令行显式传入的值覆盖配置（优先级最高）。"""
    out = dict(settings)
    if args.input is not None:
        out["input"] = args.input
    if args.output is not None:
        out["output"] = args.output
    if args.src is not None:
        out["src"] = args.src
    if args.tgt is not None:
        out["tgt"] = args.tgt
    if args.device is not None:
        out["device"] = args.device
    if args.compute_type is not None:
        out["compute_type"] = args.compute_type
    if args.beam_size is not None:
        out["beam_size"] = args.beam_size
    if args.batch_size is not None:
        out["batch_size"] = args.batch_size
    if args.suffix is not None:
        out["suffix"] = args.suffix
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wall_t0 = time.perf_counter()

    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    # 若相对路径找不到，再尝试脚本旁的默认 config
    if not config_path.exists() and args.config == ROOT / "config.yaml":
        config_path = ROOT / "config.yaml"
    elif not args.config.is_absolute():
        # 优先 cwd，其次项目根
        cwd_cfg = Path.cwd() / args.config
        root_cfg = ROOT / args.config
        config_path = cwd_cfg if cwd_cfg.exists() else root_cfg

    config = load_config(config_path)
    config_dir = config_path.parent if config_path.exists() else ROOT

    if args.list_models:
        list_models(config)
        return 0

    # 从配置选出模型，再套用命令行覆盖
    settings = pick_model_settings(
        config,
        model_id=args.model_id,
        model_path_override=args.model_path,
        config_dir=config_dir,
    )
    settings = apply_cli_overrides(settings, args)

    # 相对 input/output 相对当前工作目录（一般在项目根运行）
    input_path = settings["input"]
    output_path = settings["output"]
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    if not settings["path"].exists():
        print(f"[error] model directory not found: {settings['path']}", file=sys.stderr)
        print("把模型放到 models/ 下，并在 config.yaml 注册。", file=sys.stderr)
        return 1
    if not input_path.exists():
        print(f"[error] input not found: {input_path}", file=sys.stderr)
        print("Put .srt files under ./input then re-run.", file=sys.stderr)
        return 1

    print(f"Config     : {config_path}")
    print(f"Model id   : {settings['id']}")
    print(f"Model name : {settings['name']}")
    print(f"Model type : {settings['type']}")
    print(f"Model path : {settings['path']}")
    print(f"Loading model ...")
    t_load = time.perf_counter()
    translator = build_translator(settings)
    load_s = time.perf_counter() - t_load
    print(f"Model loaded in {format_duration(load_s)}")
    print(f"Device: {translator.device} | {settings['src']} -> {settings['tgt']}")
    print(f"beam_size={settings['beam_size']}, batch_size={settings['batch_size']}")
    print("-" * 60)

    all_stats: list[dict[str, float | int]] = []
    src_lang = settings["src"]
    tgt_lang = settings["tgt"]
    beam_size = settings["beam_size"]
    batch_size = settings["batch_size"]
    suffix = settings["suffix"]

    # ---- 单文件模式 ----
    if input_path.is_file():
        out = output_path
        if out.is_dir() or str(out).endswith(("/", "\\")) or output_path.suffix.lower() != ".srt":
            out = Path(output_path) / _output_name(input_path, suffix)

        stats = translate_srt_file(
            translator,
            input_path,
            out,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            beam_size=beam_size,
            batch_size=batch_size,
        )
        all_stats.append(stats)
        print_file_timing(input_path, stats)
        print(f"Wrote {out}")
        _print_summary(all_stats, load_s, wall_t0)
        return 0

    # ---- 目录批量模式 ----
    srt_files = collect_srt_files(input_path)
    if not srt_files:
        print(f"[error] no .srt files under {input_path}", file=sys.stderr)
        return 1

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(srt_files, desc="Translating", unit="file"):
        rel = src_path.relative_to(input_path)
        dst_path = out_dir / rel.parent / _output_name(src_path, suffix)
        stats = translate_srt_file(
            translator,
            src_path,
            dst_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            beam_size=beam_size,
            batch_size=batch_size,
        )
        all_stats.append(stats)
        print()
        print_file_timing(src_path, stats)

    print("-" * 60)
    print(f"Done. {len(srt_files)} file(s) -> {out_dir.resolve()}")
    _print_summary(all_stats, load_s, wall_t0)
    return 0


def _print_summary(
    all_stats: list[dict[str, float | int]],
    load_s: float,
    wall_t0: float,
) -> None:
    """打印整次运行的汇总耗时。"""
    wall_s = time.perf_counter() - wall_t0
    total_cues = sum(int(s["num_cues"]) for s in all_stats)
    total_translated = sum(int(s["num_translated"]) for s in all_stats)
    translate_s = sum(float(s["file_total_s"]) for s in all_stats)
    infer_s = sum(float(s["infer_s"]) for s in all_stats)

    print("=" * 60)
    print("Timing summary")
    print(f"  model load     : {format_duration(load_s)}")
    print(f"  translate all  : {format_duration(translate_s)}")
    print(f"  infer only     : {format_duration(infer_s)}")
    print(f"  wall clock     : {format_duration(wall_s)}")
    print(f"  files / cues   : {len(all_stats)} file(s), {total_cues} cues ({total_translated} translated)")
    if infer_s > 0 and total_translated > 0:
        print(f"  avg speed      : {total_translated / infer_s:.1f} cues/s (infer only)")
    print("=" * 60)


def _output_name(src: Path, suffix: str) -> str:
    """可选给输出文件加后缀，例如 movie.srt + .zh → movie.zh.srt。"""
    if suffix:
        return f"{src.stem}{suffix}.srt"
    return src.name


if __name__ == "__main__":
    raise SystemExit(main())
