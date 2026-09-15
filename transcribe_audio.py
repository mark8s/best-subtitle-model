#!/usr/bin/env python3
"""
faster-whisper 音频转字幕测试（独立脚本，不和翻译模型串联）

输入：音频文件（wav / mp3 / m4a / flac 等，需系统有 ffmpeg）
输出：SRT 字幕

典型流程：
  1. 加载 Whisper 模型（CTranslate2 加速版）
  2. 转写音频，得到带时间戳的分段
  3. 写成 SRT，并打印耗时
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "large-v3"
DEFAULT_LOCAL_MODEL_DIR = ROOT / "models" / "faster-whisper-large-v3"

# 常用 Whisper 模型名（faster-whisper 会从 Hugging Face 自动下载到缓存）
# 项目内有 models/faster-whisper-large-v3 时默认走本地，无需 --model
COMMON_MODELS = """
  tiny / tiny.en
  base / base.en
  small / small.en
  medium / medium.en
  large-v2
  large-v3                # 默认；效果最好，也最慢最大
  large-v3-turbo / turbo  # 更快，质量通常不如 large-v3
  distil-large-v3         # 蒸馏版，更快
""".strip()


def format_duration(seconds: float) -> str:
    """秒数 → 易读字符串。"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:04.1f}s"


def format_srt_timestamp(seconds: float) -> str:
    """秒 → SRT 时间码 00:00:01,000。"""
    if seconds < 0:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    total_ms = int(round(td.total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def partial_output_path(path: Path) -> Path:
    """正式输出旁边的增量字幕，例如 movie.partial.srt。"""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def format_srt_entry(index: int, start: float, end: float, text: str) -> str:
    """格式化一个完整 SRT 条目。"""
    return (
        f"{index}\n"
        f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
        f"{text.strip()}\n\n"
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(device: str, compute_type: str) -> str:
    """auto 时：CPU 用 int8，GPU 用 float16（更快更省显存）。"""
    if compute_type != "auto":
        return compute_type
    return "int8" if device == "cpu" else "float16"


def default_output_path(audio: Path, output: Path | None) -> Path:
    """未指定 -o 时：output/<音频名>.srt。"""
    if output is None:
        return ROOT / "output" / f"{audio.stem}.srt"
    if output.is_dir() or str(output).endswith(("/", "\\")) or output.suffix.lower() != ".srt":
        return Path(output) / f"{audio.stem}.srt"
    return output


def is_ct2_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "model.bin").is_file()


def resolve_model_id(model: str) -> str:
    """优先用本地目录，避免无网机器去 Hugging Face 下载。"""
    given = Path(model)
    candidates = [given]
    if not given.is_absolute():
        candidates.extend([Path.cwd() / given, ROOT / given])
    for path in candidates:
        if is_ct2_model_dir(path):
            return str(path.resolve())
        if path.exists() and path.is_dir():
            return str(path.resolve())

    if model == DEFAULT_MODEL_NAME and is_ct2_model_dir(DEFAULT_LOCAL_MODEL_DIR):
        return str(DEFAULT_LOCAL_MODEL_DIR.resolve())

    named = ROOT / "models" / f"faster-whisper-{model}"
    if is_ct2_model_dir(named):
        return str(named.resolve())
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to SRT with faster-whisper (standalone ASR test).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Common model sizes:\n{COMMON_MODELS}",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input audio file (wav/mp3/m4a/flac/...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output SRT path or directory (default: output/<name>.srt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Model size name or local dir "
            f"(default: {DEFAULT_MODEL_NAME}; uses models/faster-whisper-large-v3 if present)"
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code, e.g. ja / zh / en / ko. Default: auto-detect",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="transcribe=原语言字幕；translate=译成英语字幕（Whisper 自带）",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="int8 / float16 / float32 / auto (default: auto)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size (default: 5; use 1 for speed)",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VAD to skip non-speech and reduce hallucinations (default: enabled; use --no-vad to disable)",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use previous output as context (default: disabled to avoid repeated hallucinations)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps (slower; still output sentence SRT)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wall_t0 = time.perf_counter()

    audio = args.input if args.input.is_absolute() else Path.cwd() / args.input
    if not audio.exists():
        print(f"[error] audio not found: {audio}", file=sys.stderr)
        print("把音频放到 input/ 再指定 -i", file=sys.stderr)
        return 1

    out_path = default_output_path(audio, args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "[error] 未安装 faster-whisper。请执行: pip install faster-whisper",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(device, args.compute_type)

    model_id = resolve_model_id(args.model)

    print(f"Audio      : {audio}")
    print(f"Output     : {out_path}")
    print(f"Model      : {model_id}")
    print(f"Device     : {device} | compute_type={compute_type}")
    print(f"Language   : {args.language or 'auto'}")
    print(f"Task       : {args.task}")
    print(
        f"beam_size  : {args.beam_size} | vad={args.vad}"
        f" | previous_text={args.condition_on_previous_text}"
    )
    print("Loading model ...")

    t_load = time.perf_counter()
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    load_s = time.perf_counter() - t_load
    print(f"Model loaded in {format_duration(load_s)}")
    print("-" * 60)

    # ---- 转写（最耗时）----
    t_infer = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio),
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=args.word_timestamps,
    )

    # 迭代 segments 才会真正跑推理。每段立即写入 partial，确保中断后可查看。
    partial_path = partial_output_path(out_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    written_segments = 0
    write_s = 0.0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for seg in tqdm(segments_iter, desc="Transcribing", unit="seg"):
                text = seg.text.strip()
                if not text:
                    continue
                written_segments += 1
                t_write = time.perf_counter()
                output_file.write(
                    format_srt_entry(written_segments, seg.start, seg.end, text)
                )
                output_file.flush()
                write_s += time.perf_counter() - t_write
    except KeyboardInterrupt:
        print("\n" + "-" * 60, file=sys.stderr)
        print(f"Transcription interrupted after {written_segments} segments.", file=sys.stderr)
        print(f"Partial SRT saved: {partial_path}", file=sys.stderr)
        return 130

    infer_s = time.perf_counter() - t_infer
    t_write = time.perf_counter()
    partial_path.replace(out_path)
    write_s += time.perf_counter() - t_write

    audio_dur = float(getattr(info, "duration", 0.0) or 0.0)
    detected = getattr(info, "language", None)
    lang_prob = getattr(info, "language_probability", None)

    print("-" * 60)
    print(f"Detected language : {detected}" + (f" ({lang_prob:.2f})" if lang_prob else ""))
    print(f"Audio duration    : {format_duration(audio_dur)}")
    print(f"Segments          : {written_segments}")
    print(f"Wrote             : {out_path}")
    print("=" * 60)
    print("Timing summary")
    print(f"  model load  : {format_duration(load_s)}")
    print(f"  transcribe  : {format_duration(infer_s)}")
    print(f"  write srt  : {format_duration(write_s)}")
    print(f"  wall clock  : {format_duration(time.perf_counter() - wall_t0)}")
    if audio_dur > 0 and infer_s > 0:
        # RTF < 1 表示比实时快
        print(f"  RTF         : {infer_s / audio_dur:.3f}  (transcribe_s / audio_s)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
