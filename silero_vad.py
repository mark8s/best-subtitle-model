"""Silero VAD，给 OpenVINO Whisper 跳过静音。

16 kHz 单声道按 512 点一窗跑 Silero ONNX。这份模型含 8k/16k 动态分支，
OpenVINO 前端转不了，所以 VAD 用 ONNX Runtime，Whisper 仍走 OpenVINO。
切段规则与 Silero 的 get_speech_timestamps 一致：短暂停顿留在同一段里，
超过 Whisper 窗口的长对白在静音处切开。
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import wave
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path(
    os.environ.get("VAD_MODEL", ROOT / "models" / "silero-vad" / "silero_vad.onnx")
)
VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SAMPLE_RATE = 16_000
WINDOW = 512
CONTEXT = 64
STATE_SHAPE = (2, 1, 128)


class VadCancelled(Exception):
    """扫描过程中收到停止。"""


class SileroVad:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "未安装 onnxruntime。请在 ov-venv 里执行: python -m pip install onnxruntime"
            ) from exc

        self.path = model_path
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def probability(self, window: np.ndarray) -> float:
        if window.shape != (WINDOW,):
            raise ValueError(f"VAD 窗口必须是 {WINDOW} 个采样，当前 {window.shape}")
        audio = np.empty((1, CONTEXT + WINDOW), dtype=np.float32)
        audio[0, :CONTEXT] = self._context
        audio[0, CONTEXT:] = window
        output, state = self._session.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        state = np.asarray(state, dtype=np.float32)
        if state.size != int(np.prod(STATE_SHAPE)):
            raise RuntimeError(f"Silero state 形状异常: {state.shape}")
        self._state = state.reshape(STATE_SHAPE).copy()
        self._context = audio[0, -CONTEXT:].copy()
        return float(np.asarray(output).reshape(-1)[0])


def ensure_vad_model(path: Path | None = None) -> Path:
    model_path = path or DEFAULT_MODEL_PATH
    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with urllib.request.urlopen(VAD_URL, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(model_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"找不到 Silero VAD 模型 {model_path}，自动下载失败：{exc}\n"
            "可手动下载：\n"
            f"  curl -L -o {model_path} {VAD_URL}"
        ) from exc
    return model_path


def speech_ranges(
    path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float,
    max_chunk_seconds: float = 30.0,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
    model_path: Path | None = None,
    model: SileroVad | None = None,
    cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float, float], None] | None = None,
) -> list[tuple[float, float]]:
    """返回绝对时间轴上的语音段 (start, end)，每段不超过 max_chunk_seconds 太多。"""
    return list(
        iter_speech_ranges(
            path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            max_chunk_seconds=max_chunk_seconds,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            model_path=model_path,
            model=model,
            cancel=cancel,
            on_progress=on_progress,
        )
    )


def iter_speech_ranges(
    path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float,
    max_chunk_seconds: float = 30.0,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
    model_path: Path | None = None,
    model: SileroVad | None = None,
    cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float, float], None] | None = None,
):
    """扫到一段完整语音就产出 (start, end)，不用等整段音频扫完。"""
    if duration_seconds <= 0 or max_chunk_seconds <= 0:
        return
    vad = model if model is not None else SileroVad(ensure_vad_model(model_path))
    vad.reset()
    total_samples = int(duration_seconds * SAMPLE_RATE)
    segmenter = SpeechSegmenter(
        audio_length_samples=total_samples,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_chunk_seconds,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    pending = np.empty(0, dtype=np.float32)
    scanned = 0
    since_check = 0
    last_report = -1.0
    span_end = start_seconds + duration_seconds

    def report(samples: int, force: bool = False) -> None:
        nonlocal last_report
        if on_progress is None:
            return
        done = start_seconds + samples / SAMPLE_RATE
        if force or done - last_report >= 1.0:
            last_report = done
            on_progress(min(done, span_end), span_end)

    def clips_from(ranges: list[tuple[int, int]]):
        return _split_long(
            _timestamps_to_seconds(ranges, offset=start_seconds, limit=span_end),
            max_chunk_seconds,
        )

    for block in _pcm_blocks(path, start_seconds, total_samples):
        pending = np.concatenate((pending, block))
        usable = (len(pending) // WINDOW) * WINDOW
        if not usable:
            continue
        for row in pending[:usable].reshape(-1, WINDOW):
            for clip in clips_from(segmenter.push(float(vad.probability(row)))):
                yield clip
            scanned += WINDOW
            since_check += WINDOW
            if since_check >= SAMPLE_RATE:
                since_check = 0
                if cancel is not None and cancel():
                    raise VadCancelled()
                report(min(scanned, total_samples))
        pending = pending[usable:].copy()

    if len(pending):
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: len(pending)] = pending
        for clip in clips_from(segmenter.push(float(vad.probability(padded)))):
            yield clip
        scanned += len(pending)
    if cancel is not None and cancel():
        raise VadCancelled()
    report(total_samples, force=True)
    yield from clips_from(segmenter.finish())


class SpeechSegmenter:
    """增量切段。上一段要等下一段的起点确定后才输出，结果与整段计算一致。"""

    def __init__(
        self,
        *,
        audio_length_samples: int,
        sampling_rate: int = SAMPLE_RATE,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = 30.0,
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
        min_silence_at_max_speech: int = 98,
        use_max_poss_sil_at_max_speech: bool = True,
    ) -> None:
        self.audio_length = audio_length_samples
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01) if neg_threshold is None else neg_threshold
        self.window = 512 if sampling_rate == 16000 else 256
        self.min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        self.pad = sampling_rate * speech_pad_ms / 1000
        self.max_speech_samples = sampling_rate * max_speech_duration_s - self.window - 2 * self.pad
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000
        self.use_max_poss_sil_at_max_speech = use_max_poss_sil_at_max_speech
        self.index = 0
        self.triggered = False
        self.current: dict[str, int] = {}
        self.raw: list[dict[str, int]] = []
        self.temp_end = 0
        self.prev_end = 0
        self.next_start = 0
        self.possible_ends: list[tuple[int, float]] = []

    def push(self, speech_prob: float) -> list[tuple[int, int]]:
        index = self.index
        self.index += 1
        cur_sample = self.window * index
        ready: list[tuple[int, int]] = []
        if speech_prob >= self.threshold and self.temp_end:
            silence_duration = cur_sample - self.temp_end
            if silence_duration > self.min_silence_samples_at_max_speech:
                self.possible_ends.append((self.temp_end, silence_duration))
            self.temp_end = 0
            if self.next_start < self.prev_end:
                self.next_start = cur_sample

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            self.current = {"start": cur_sample}
            return ready

        if self.triggered and cur_sample - self.current["start"] > self.max_speech_samples:
            if self.use_max_poss_sil_at_max_speech and self.possible_ends:
                prev_end, duration = max(self.possible_ends, key=lambda item: item[1])
                self.current["end"] = prev_end
                ready.extend(self._append(self.current))
                self.current = {}
                self.next_start = int(prev_end + duration)
                if self.next_start < prev_end + cur_sample:
                    self.current = {"start": self.next_start}
                else:
                    self.triggered = False
                    self.prev_end = self.next_start = self.temp_end = 0
                    self.possible_ends = []
            elif self.prev_end:
                self.current["end"] = self.prev_end
                ready.extend(self._append(self.current))
                self.current = {}
                if self.next_start < self.prev_end:
                    self.triggered = False
                else:
                    self.current = {"start": self.next_start}
                self.prev_end = self.next_start = self.temp_end = 0
                self.possible_ends = []
            else:
                self.current["end"] = cur_sample
                ready.extend(self._append(self.current))
                self.current = {}
                self.prev_end = self.next_start = self.temp_end = 0
                self.triggered = False
                self.possible_ends = []
            return ready

        if speech_prob < self.neg_threshold and self.triggered:
            if not self.temp_end:
                self.temp_end = cur_sample
            silence_now = cur_sample - self.temp_end
            if (
                not self.use_max_poss_sil_at_max_speech
                and silence_now > self.min_silence_samples_at_max_speech
            ):
                self.prev_end = self.temp_end
            if silence_now < self.min_silence_samples:
                return ready
            self.current["end"] = self.temp_end
            if self.current["end"] - self.current["start"] > self.min_speech_samples:
                ready.extend(self._append(self.current))
            self.current = {}
            self.prev_end = self.next_start = self.temp_end = 0
            self.triggered = False
            self.possible_ends = []
        return ready

    def finish(self) -> list[tuple[int, int]]:
        ready: list[tuple[int, int]] = []
        if self.current and self.audio_length - self.current["start"] > self.min_speech_samples:
            self.current["end"] = self.audio_length
            ready.extend(self._append(self.current))
        self.current = {}
        if not self.raw:
            return ready
        last = self.raw[-1]
        if len(self.raw) == 1:
            last["start"] = int(max(0, last["start"] - self.pad))
        last["end"] = int(min(self.audio_length, last["end"] + self.pad))
        if last["end"] > last["start"]:
            ready.append((last["start"], last["end"]))
        self.raw.clear()
        return ready

    def _append(self, speech: dict[str, int]) -> list[tuple[int, int]]:
        self.raw.append({"start": int(speech["start"]), "end": int(speech["end"])})
        if len(self.raw) < 2:
            return []
        return [self._finalize_pair(len(self.raw) - 2)]

    def _finalize_pair(self, index: int) -> tuple[int, int]:
        speech = self.raw[index]
        nxt = self.raw[index + 1]
        if index == 0:
            speech["start"] = int(max(0, speech["start"] - self.pad))
        gap = nxt["start"] - speech["end"]
        if gap < 2 * self.pad:
            speech["end"] += int(gap // 2)
            nxt["start"] = int(max(0, nxt["start"] - gap // 2))
        else:
            speech["end"] = int(min(self.audio_length, speech["end"] + self.pad))
            nxt["start"] = int(max(0, nxt["start"] - self.pad))
        return (speech["start"], speech["end"])


def speech_timestamps_from_probs(
    speech_probs: list[float],
    *,
    audio_length_samples: int,
    sampling_rate: int = SAMPLE_RATE,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = 30.0,
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[tuple[int, int]]:
    """把每窗语音概率切成采样点区间。逻辑对齐 Silero get_speech_timestamps_from_probs。"""
    segmenter = SpeechSegmenter(
        audio_length_samples=audio_length_samples,
        sampling_rate=sampling_rate,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        min_silence_at_max_speech=min_silence_at_max_speech,
        use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
    )
    ready: list[tuple[int, int]] = []
    for prob in speech_probs:
        ready.extend(segmenter.push(prob))
    ready.extend(segmenter.finish())
    return [(start, end) for start, end in ready if end > start]


def _timestamps_to_seconds(
    ranges: list[tuple[int, int]],
    *,
    offset: float,
    limit: float,
) -> list[tuple[float, float]]:
    clips: list[tuple[float, float]] = []
    for start, end in ranges:
        clip_start = max(offset, offset + start / SAMPLE_RATE)
        clip_end = min(limit, offset + end / SAMPLE_RATE)
        if clip_end - clip_start >= 0.05:
            clips.append((clip_start, clip_end))
    return _merge(clips)


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1e-3:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_long(ranges: list[tuple[float, float]], max_chunk_seconds: float) -> list[tuple[float, float]]:
    limit = max_chunk_seconds + 2.0
    clips: list[tuple[float, float]] = []
    for start, end in ranges:
        if end - start <= limit:
            clips.append((start, end))
            continue
        cursor = start
        while end - cursor > max_chunk_seconds:
            clips.append((cursor, cursor + max_chunk_seconds))
            cursor += max_chunk_seconds
        if end - cursor >= 0.05:
            clips.append((cursor, end))
    return clips


def _pcm_blocks(path: Path, start_seconds: float, total_samples: int, block_seconds: float = 10.0):
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getnchannels() != 1
            or wav_file.getsampwidth() != 2
            or wav_file.getframerate() != SAMPLE_RATE
            or wav_file.getcomptype() != "NONE"
        ):
            raise ValueError(
                "VAD 输入必须是 16kHz、单声道、PCM s16le WAV；"
                f"当前 channels={wav_file.getnchannels()}, sample_width={wav_file.getsampwidth()}, "
                f"sample_rate={wav_file.getframerate()}, compression={wav_file.getcomptype()}"
            )
        start_frame = min(int(start_seconds * SAMPLE_RATE), wav_file.getnframes())
        wav_file.setpos(start_frame)
        remaining = min(total_samples, wav_file.getnframes() - start_frame)
        block = max(WINDOW, int(block_seconds * SAMPLE_RATE))
        while remaining > 0:
            count = min(block, remaining)
            raw = wav_file.readframes(count)
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if audio.size == 0:
                break
            remaining -= audio.size
            yield audio


def _check() -> None:
    silence = [0.0] * 300
    assert speech_timestamps_from_probs(silence, audio_length_samples=300 * WINDOW) == []

    probs = [0.0] * 50 + [0.9] * 80 + [0.0] * 270
    ranges = speech_timestamps_from_probs(probs, audio_length_samples=len(probs) * WINDOW)
    assert ranges == [(19200, 72960)], ranges

    blip = [0.0] * 20 + [0.9] * 2 + [0.0] * 100
    assert speech_timestamps_from_probs(blip, audio_length_samples=len(blip) * WINDOW) == []

    long_speech = [0.9] * 400
    pieces = speech_timestamps_from_probs(
        long_speech,
        audio_length_samples=len(long_speech) * WINDOW,
        max_speech_duration_s=1.0,
        min_silence_duration_ms=2000,
        speech_pad_ms=0,
    )
    assert len(pieces) > 1, pieces

    two = [0.0] * 20 + [0.9] * 80 + [0.0] * 400 + [0.9] * 80 + [0.0] * 200
    segmenter = SpeechSegmenter(audio_length_samples=len(two) * WINDOW)
    streamed: list[tuple[int, int]] = []
    first_at = None
    for index, prob in enumerate(two):
        ready = segmenter.push(prob)
        if ready and first_at is None:
            first_at = index
        streamed.extend(ready)
    streamed.extend(segmenter.finish())
    assert first_at is not None and first_at < len(two) - 1, first_at
    assert streamed == speech_timestamps_from_probs(two, audio_length_samples=len(two) * WINDOW)

    import tempfile

    class Energy:
        def reset(self) -> None:
            return None

        def probability(self, window: np.ndarray) -> float:
            return 0.9 if float(np.max(np.abs(window))) > 0.2 else 0.0

    with tempfile.TemporaryDirectory() as directory:
        wav_path = Path(directory) / "clip.wav"
        _write_clip(wav_path, [(2.0, 0.0), (1.0, 0.5), (3.0, 0.0)])
        found = speech_ranges(
            wav_path,
            start_seconds=0.0,
            duration_seconds=6.0,
            max_chunk_seconds=30.0,
            model=Energy(),  # type: ignore[arg-type]
        )
        assert len(found) == 1, found
        start, end = found[0]
        assert 1.5 <= start <= 2.0, found
        assert 3.0 <= end <= 3.5, found
        offset = speech_ranges(
            wav_path,
            start_seconds=2.0,
            duration_seconds=1.0,
            max_chunk_seconds=30.0,
            model=Energy(),  # type: ignore[arg-type]
        )
        assert offset and offset[0][0] >= 2.0 and offset[0][1] <= 3.0, offset
    print("silero_vad checks ok")


def _write_clip(path: Path, parts: list[tuple[float, float]]) -> None:
    chunks = []
    for duration, amplitude in parts:
        count = int(duration * SAMPLE_RATE)
        if amplitude == 0:
            chunks.append(np.zeros(count, dtype=np.float32))
        else:
            ticks = np.arange(count, dtype=np.float32)
            chunks.append(amplitude * np.sin(2 * np.pi * 440 * ticks / SAMPLE_RATE))
    audio = np.concatenate(chunks)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


if __name__ == "__main__":
    _check()
