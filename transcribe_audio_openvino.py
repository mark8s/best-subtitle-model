#!/usr/bin/env python3
"""
OpenVINO GenAI Whisper 音频转 SRT（Intel GPU / CPU）。

输入必须是 16 kHz、单声道、PCM s16le WAV。该脚本与 faster-whisper
后端完全独立，默认加载 OpenVINO INT8 large-v3 并在 Intel GPU 上推理。
长音频按块推理，避免整片 WAV 转成 Python list 导致 OOM。
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "openvino-whisper-large-v3-int8"
SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 300.0


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:04.1f}s"


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(timedelta(seconds=seconds).total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def partial_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def format_srt_entry(index: int, start: float, end: float, text: str) -> str:
    return (
        f"{index}\n"
        f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
        f"{text.strip()}\n\n"
    )


def default_output_path(audio: Path, output: Path | None) -> Path:
    if output is None:
        return ROOT / "output" / f"{audio.stem}.openvino.srt"
    if output.is_dir() or str(output).endswith(("/", "\\")) or output.suffix.lower() != ".srt":
        return output / f"{audio.stem}.openvino.srt"
    return output


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def whisper_language_token(language: str | None) -> str | None:
    if not language:
        return None
    if language.startswith("<|") and language.endswith("|>"):
        return language
    return f"<|{language}|>"


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate() or SAMPLE_RATE)


def entries_from_result(result: Any, clip_offset: float, audio_duration: float) -> list[tuple[float, float, str]]:
    chunks = result_chunks(result)
    entries: list[tuple[float, float, str]] = []
    for chunk in chunks:
        text = str(getattr(chunk, "text", "")).strip()
        if not text:
            continue
        start = clip_offset + float(getattr(chunk, "start_ts", 0.0))
        end = clip_offset + float(getattr(chunk, "end_ts", 0.0))
        if end <= start:
            continue
        entries.append((start, end, text))
    if not entries:
        text = result_text(result)
        if text:
            entries.append((clip_offset, clip_offset + audio_duration, text))
    return entries


def read_pcm_wav(
    path: Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[np.ndarray, float]:
    """读取一段 16 kHz mono PCM16 WAV，返回 [-1, 1) float32 和时长。"""
    try:
        wav_file = wave.open(str(path), "rb")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"无法读取 WAV: {exc}") from exc

    with wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()
        total_frames = wav_file.getnframes()

        if (
            channels != 1
            or sample_width != 2
            or sample_rate != SAMPLE_RATE
            or compression != "NONE"
        ):
            raise ValueError(
                "OpenVINO 输入必须是 16kHz、单声道、PCM s16le WAV；"
                f"当前 channels={channels}, sample_width={sample_width}, "
                f"sample_rate={sample_rate}, compression={compression}"
            )

        start_frame = min(int(start_seconds * sample_rate), total_frames)
        wav_file.setpos(start_frame)
        available_frames = total_frames - start_frame
        if duration_seconds is None:
            frame_count = available_frames
        else:
            frame_count = min(int(duration_seconds * sample_rate), available_frames)

        raw = wav_file.readframes(frame_count)

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio, len(audio) / SAMPLE_RATE


def result_chunks(result: Any) -> list[Any]:
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        return []
    # ASRPipeline returns one chunk list per input; WhisperPipeline returns a flat list.
    if chunks and isinstance(chunks[0], (list, tuple)):
        return list(chunks[0])
    return list(chunks)


def result_text(result: Any) -> str:
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0]).strip()
    text = getattr(result, "text", None)
    if text:
        return str(text).strip()
    return str(result).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a 16kHz mono PCM WAV to SRT with OpenVINO GenAI Whisper.",
    )
    parser.add_argument("-i", "--input", type=Path, help="Input 16kHz mono PCM16 WAV")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output SRT path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"OpenVINO model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device, e.g. GPU / GPU.0 / CPU (default: GPU)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Whisper language code, e.g. en / zh / it; default: auto-detect",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="transcribe=原语言字幕；translate=译成英语字幕",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start offset in seconds (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Only process this many seconds (useful for quick tests)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=(
            "Process this many seconds per generate() call "
            f"(default: {int(DEFAULT_CHUNK_SECONDS)}; avoids OOM on long files)"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List OpenVINO devices and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import openvino as ov
    except ImportError:
        print(
            "[error] 未安装 OpenVINO。请激活 ov-venv 后安装 requirements-openvino.txt",
            file=sys.stderr,
        )
        return 1

    core = ov.Core()
    available_devices = list(core.available_devices)
    if args.list_devices:
        print("Available OpenVINO devices:")
        for device in available_devices:
            print(f"  {device}")
        return 0

    if args.input is None:
        print("[error] 必须指定 -i/--input（或使用 --list-devices）", file=sys.stderr)
        return 2
    if args.start < 0 or (args.duration is not None and args.duration <= 0):
        print("[error] --start 不能为负，--duration 必须大于 0", file=sys.stderr)
        return 2
    if args.chunk_seconds <= 0:
        print("[error] --chunk-seconds 必须大于 0", file=sys.stderr)
        return 2

    requested_device = args.device.upper()
    device_base = requested_device.split(".", 1)[0]
    available_bases = {device.split(".", 1)[0] for device in available_devices}
    if device_base not in available_bases and device_base not in {"AUTO", "MULTI", "HETERO"}:
        print(
            f"[error] OpenVINO 设备 {requested_device} 不可用；"
            f"当前设备: {', '.join(available_devices) or 'none'}",
            file=sys.stderr,
        )
        return 1

    audio_path = resolve_path(args.input)
    if not audio_path.is_file():
        print(f"[error] audio not found: {audio_path}", file=sys.stderr)
        return 1

    model_path = resolve_path(args.model)
    if not model_path.is_dir() or not any(model_path.glob("*.xml")):
        print(
            f"[error] OpenVINO model not found or incomplete: {model_path}\n"
            "请下载 OpenVINO/whisper-large-v3-int8-ov 到该目录。",
            file=sys.stderr,
        )
        return 1

    out_path = default_output_path(audio_path, args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    try:
        total_wav = wav_duration_seconds(audio_path)
        remaining = max(0.0, total_wav - args.start)
        audio_duration = remaining if args.duration is None else min(args.duration, remaining)
        # Validate format on a tiny read before loading the pipeline.
        probe, _ = read_pcm_wav(audio_path, start_seconds=args.start, duration_seconds=min(0.1, audio_duration or 0.1))
    except (ValueError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if audio_duration <= 0 or probe.size == 0:
        print("[error] 选择的音频区间为空", file=sys.stderr)
        return 1
    del probe

    try:
        import openvino_genai
    except ImportError:
        print(
            "[error] 未安装 openvino-genai。请执行: "
            "python -m pip install -r requirements-openvino.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Audio      : {audio_path}")
    print(f"Output     : {out_path}")
    print(f"Model      : {model_path}")
    print(f"Device     : {requested_device}")
    print(f"Language   : {args.language or 'auto'}")
    print(f"Task       : {args.task}")
    print(
        f"Clip       : start={args.start:.2f}s | duration={format_duration(audio_duration)}"
        f" | chunk={format_duration(args.chunk_seconds)}"
    )
    print("Loading OpenVINO Whisper model ...")

    wall_t0 = time.perf_counter()
    t_load = time.perf_counter()
    try:
        pipeline = openvino_genai.WhisperPipeline(str(model_path), requested_device)
    except Exception as exc:
        print(f"[error] OpenVINO model load failed: {exc}", file=sys.stderr)
        return 1
    load_s = time.perf_counter() - t_load
    print(f"Model loaded in {format_duration(load_s)}")
    print("-" * 60)

    generation_options: dict[str, Any] = {
        "task": args.task,
        "return_timestamps": True,
    }
    language_token = whisper_language_token(args.language)
    if language_token:
        generation_options["language"] = language_token

    partial_path = partial_output_path(out_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    written_segments = 0
    write_s = 0.0
    t_infer = time.perf_counter()
    cursor = args.start
    clip_end = args.start + audio_duration
    chunk_index = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as output_file:
            while cursor < clip_end - 1e-6:
                chunk_index += 1
                this_duration = min(args.chunk_seconds, clip_end - cursor)
                audio, actual_duration = read_pcm_wav(
                    audio_path,
                    start_seconds=cursor,
                    duration_seconds=this_duration,
                )
                if audio.size == 0:
                    break
                print(
                    f"Chunk {chunk_index}: {format_srt_timestamp(cursor)}  "
                    f"({format_duration(actual_duration)})"
                )
                result = pipeline.generate(audio.tolist(), **generation_options)
                del audio
                entries = entries_from_result(result, cursor, actual_duration)
                t_write = time.perf_counter()
                for start, end, text in entries:
                    written_segments += 1
                    output_file.write(format_srt_entry(written_segments, start, end, text))
                output_file.flush()
                write_s += time.perf_counter() - t_write
                cursor += actual_duration
    except KeyboardInterrupt:
        print("\n" + "-" * 60, file=sys.stderr)
        print(f"Interrupted after {written_segments} segments.", file=sys.stderr)
        print(f"Partial SRT saved: {partial_path}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] OpenVINO transcription failed: {exc}", file=sys.stderr)
        return 1
    infer_s = time.perf_counter() - t_infer

    t_write = time.perf_counter()
    partial_path.replace(out_path)
    write_s += time.perf_counter() - t_write
    wall_s = time.perf_counter() - wall_t0
    segment_count = written_segments

    print("-" * 60)
    print(f"Audio duration : {format_duration(audio_duration)}")
    print(f"Segments       : {segment_count}")
    print(f"Wrote          : {out_path}")
    print("=" * 60)
    print("Timing summary")
    print(f"  model load : {format_duration(load_s)}")
    print(f"  transcribe : {format_duration(infer_s)}")
    print(f"  write srt  : {format_duration(write_s)}")
    print(f"  wall clock : {format_duration(wall_s)}")
    if audio_duration > 0 and infer_s > 0:
        print(f"  RTF        : {infer_s / audio_duration:.3f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
