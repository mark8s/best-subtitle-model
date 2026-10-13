"""听写、翻译，以及把字幕句推给 WebSocket。"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_server import db, media

WHISPER_DIR = Path(os.environ.get("WHISPER_MODEL", ROOT / "models" / "openvino-whisper-large-v3-int8"))
NLLB_DIR = Path(os.environ.get("NLLB_MODEL", ROOT / "models" / "nllb-600m-int8"))
M2M_DIR = Path(os.environ.get("M2M_MODEL", ROOT / "models" / "m2m100-418m-int8"))
DEVICE = os.environ.get("OV_DEVICE", "GPU").upper()
CHUNK_SECONDS = 30.0


class Hub:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._listeners: dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._listeners.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            listeners = self._listeners.get(job_id, [])
            if queue in listeners:
                listeners.remove(queue)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners.get(job_id, []))
        loop = self.loop
        if loop is None:
            return
        for queue in listeners:
            loop.call_soon_threadsafe(queue.put_nowait, event)


hub = Hub()
_runner_lock = threading.Lock()
_cancel = threading.Event()
_thread: threading.Thread | None = None
_positions: dict[str, float] = {}
_whisper: Any = None
_translators: dict[str, Any] = {}


def note_position(job_id: str, position_sec: float) -> None:
    _positions[job_id] = max(0.0, position_sec)


def position_of(job_id: str) -> float | None:
    return _positions.get(job_id)


def models_ready() -> dict[str, bool]:
    return {
        "whisper": WHISPER_DIR.is_dir() and any(WHISPER_DIR.glob("*.xml")),
        "nllb": (NLLB_DIR / "model.bin").is_file(),
        "m2m": (M2M_DIR / "model.bin").is_file(),
    }


def job_busy() -> bool:
    return _thread is not None and _thread.is_alive()


def start_job(job: dict[str, Any]) -> None:
    global _thread
    with _runner_lock:
        if job_busy():
            raise RuntimeError("已有任务在跑")
        _cancel.clear()
        _thread = threading.Thread(target=_run_job, args=(job,), name="demo-job", daemon=True)
        _thread.start()


def cancel_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job["status"] != "running":
        return
    _cancel.set()
    db.append_log(job_id, "已停止")
    db.finish_job(job_id, "stopped")
    hub.publish(job_id, {"type": "status", "status": "stopped"})


def _stopped() -> bool:
    return _cancel.is_set()


def _run_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    try:
        if job["kind"] == "transcribe":
            _transcribe(job)
        else:
            _translate(job)
        if _stopped():
            return
        current = db.get_job(job_id)
        if current and current["status"] == "running":
            db.append_log(job_id, "任务完成" + _spent_suffix(current))
            db.finish_job(job_id, "done")
            hub.publish(job_id, {"type": "done"})
    except Exception as exc:
        if _stopped():
            return
        db.append_log(job_id, f"失败：{exc}")
        db.finish_job(job_id, "failed", str(exc))
        hub.publish(job_id, {"type": "error", "message": str(exc)})


def _transcribe(job: dict[str, Any]) -> None:
    video = media.resolve_media(job["video_path"])
    language = job["language"]
    index = int(job["stream_index"])
    wav_path = media.artifact_path(video, index, language, "wav")
    srt_path = media.artifact_path(video, index, language, "srt")
    label = media.language_label(language)
    if srt_path.is_file():
        # 结果已在磁盘上：不记任务（接口侧也会拦一层，这里防竞态）
        db.remove_job(job["id"])
        return
    db.append_log(job["id"], f"开始把音频转成文字 {video.name}，音轨 {label}")
    if not wav_path.is_file():
        db.update_job(job["id"], stage="extract", done_count=0, total_count=None)
        hub.publish(job["id"], {"type": "status", "stage": "extract"})
        db.append_log(job["id"], f"抽取 16kHz 单声道音频：{wav_path.name}")
        media.extract_wav(video, index, wav_path, _stopped)
        if _stopped():
            return
        db.append_log(job["id"], "音频抽取完成")
    else:
        db.append_log(job["id"], f"音频已存在，跳过抽取：{wav_path.name}")
    if _stopped():
        return
    _write_whisper_srt(job, wav_path, srt_path)


def _write_whisper_srt(job: dict[str, Any], wav_path: Path, srt_path: Path) -> None:
    import transcribe_audio_openvino as asr

    global _whisper
    if _whisper is None:
        db.append_log(job["id"], f"加载 Whisper 模型，设备 {DEVICE}")
        import openvino_genai

        _whisper = openvino_genai.WhisperPipeline(str(WHISPER_DIR), DEVICE)
    duration = asr.wav_duration_seconds(wav_path)
    total = max(1, int(duration))
    db.update_job(job["id"], stage="transcribe", done_count=0, total_count=total)
    db.append_log(job["id"], f"开始转写，设备 {DEVICE}，音频 {total} 秒，每段 {int(CHUNK_SECONDS)} 秒")
    short = media.short_lang(job["language"])
    options: dict[str, Any] = {"task": "transcribe", "return_timestamps": True}
    if short and short != "und":
        options["language"] = asr.whisper_language_token(short)
    cursor = 0.0
    index = 0
    partial = srt_path.with_suffix(srt_path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        while cursor < duration - 1e-3:
            if _stopped():
                partial.unlink(missing_ok=True)
                return
            audio, actual = asr.read_pcm_wav(wav_path, start_seconds=cursor, duration_seconds=CHUNK_SECONDS)
            if audio.size == 0:
                break
            result = _whisper.generate(audio.tolist(), **options)
            del audio
            for start, end, text in asr.entries_from_result(result, cursor, actual):
                if asr.is_ignored_subtitle(text):
                    continue
                index += 1
                handle.write(asr.format_srt_entry(index, start, end, text))
            handle.flush()
            cursor += actual
            done = min(int(cursor), int(duration))
            db.update_job(job["id"], done_count=done)
            db.append_log(job["id"], f"已转写 {done} / {int(duration)} 秒")
            hub.publish(
                job["id"],
                {"type": "status", "stage": "transcribe", "doneCount": int(cursor), "totalCount": int(duration)},
            )
    if index == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("没有转出字幕")
    partial.replace(srt_path)
    db.append_log(job["id"], f"转写完成，写入 {srt_path.name}，共 {index} 条")


def _translate(job: dict[str, Any]) -> None:
    video = media.resolve_media(job["video_path"])
    language = job["language"]
    index = int(job["stream_index"])
    target = job["target_lang"]
    model = _job_model(job)
    source_path = media.artifact_path(video, index, language, "srt")
    target_path = media.translation_path(video, index, language, target, model)
    source_label = media.language_label(language)
    target_label = media.language_label(target)
    if target_path.is_file():
        # 结果已在磁盘上：不记任务（接口侧也会拦一层，这里防竞态）
        db.remove_job(job["id"])
        return
    db.append_log(job["id"], f"开始翻译 {video.name}：{source_label} → {target_label}")
    if not source_path.is_file():
        raise RuntimeError("还没有原语言字幕，请先把音频转成文字")
    source_cues = parse_srt(source_path)
    short = media.short_lang(language)
    if short == target:
        # 同源同目标只是复制，不记任务（接口侧也会拦）
        _copy_cues(job["id"], source_cues, target_path)
        db.remove_job(job["id"])
        return
    _load_translator(job["id"], model)
    db.append_log(job["id"], f"调用 {_model_label(model)}，CPU int8，共 {len(source_cues)} 条")
    _translate_cues(job, source_cues, target_path, short, target)


def _translate_cues(
    job: dict[str, Any],
    cues: list[dict[str, Any]],
    target_path: Path,
    source_short: str,
    target_short: str,
) -> None:
    import transcribe_audio_openvino as asr

    translator = _load_translator(job["id"], _job_model(job))
    src = source_short
    tgt = target_short
    db.update_job(job["id"], stage="translate", done_count=0, total_count=len(cues))
    partial = target_path.with_suffix(target_path.suffix + ".partial")
    batch = 8
    done = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(cues), batch):
            if _stopped():
                partial.unlink(missing_ok=True)
                return
            group = cues[offset : offset + batch]
            texts = translator(src, tgt, [cue["text"] for cue in group])
            for cue, text in zip(group, texts):
                done += 1
                line = text.strip() or cue["text"]
                handle.write(asr.format_srt_entry(done, cue["startSec"], cue["endSec"], line))
                hub.publish(
                    job["id"],
                    {
                        "type": "cue",
                        "seq": done,
                        "startSec": cue["startSec"],
                        "endSec": cue["endSec"],
                        "text": line,
                    },
                )
            handle.flush()
            db.update_job(job["id"], done_count=done, total_count=len(cues))
            if done == len(cues) or done % 80 == 0:
                db.append_log(job["id"], f"已翻译 {done} / {len(cues)} 条")
    partial.replace(target_path)
    db.append_log(job["id"], f"翻译完成，写入 {target_path.name}")


def _job_model(job: dict[str, Any]) -> str:
    model = str(job.get("model") or "nllb")
    return model if model in {"nllb", "m2m"} else "nllb"


def _model_label(model: str) -> str:
    return "M2M100 418M" if model == "m2m" else "NLLB 600M"


def _load_translator(job_id: str, model: str) -> Callable[[str, str, list[str]], list[str]]:
    loaded = _translators.get(model)
    if loaded is None:
        db.append_log(job_id, f"加载 {_model_label(model)}，CPU int8")
        loaded = _Ct2Nllb() if model == "nllb" else _Ct2M2m()
        _translators[model] = loaded
    return loaded


class _Ct2Nllb:
    def __init__(self) -> None:
        from translate_srt import NllbTranslator

        self.translator = NllbTranslator(NLLB_DIR, device="cpu", compute_type="int8")

    def __call__(self, src: str, tgt: str, texts: list[str]) -> list[str]:
        translated, _stats = self.translator.translate_texts(
            texts,
            media.nllb_code(src),
            media.nllb_code(tgt),
            beam_size=2,
            batch_size=16,
        )
        return translated


class _Ct2M2m:
    def __init__(self) -> None:
        from translate_srt import M2m100Translator

        self.translator = M2m100Translator(M2M_DIR, device="cpu", compute_type="int8")

    def __call__(self, src: str, tgt: str, texts: list[str]) -> list[str]:
        try:
            translated, _stats = self.translator.translate_texts(
                texts, src, tgt, beam_size=2, batch_size=16
            )
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        return translated


def _copy_cues(job_id: str, cues: list[dict[str, Any]], target_path: Path) -> None:
    import transcribe_audio_openvino as asr

    with target_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cue in cues:
            handle.write(asr.format_srt_entry(cue["seq"], cue["startSec"], cue["endSec"], cue["text"]))
            hub.publish(job_id, {"type": "cue", **cue})
    db.update_job(job_id, done_count=len(cues), total_count=len(cues))


def _publish_file(job_id: str, path: Path) -> None:
    cues = parse_srt(path)
    for cue in cues:
        hub.publish(job_id, {"type": "cue", **cue})
    db.update_job(job_id, done_count=len(cues), total_count=len(cues))


def parse_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re_split_blocks(text)
    cues = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "-->" not in lines[1] and "-->" not in lines[0]:
            continue
        if "-->" in lines[0]:
            timing = lines[0]
            body = lines[1:]
            seq = len(cues) + 1
        else:
            seq = _int_or(lines[0], len(cues) + 1)
            timing = lines[1]
            body = lines[2:]
        start_s, end_s = timing.split("-->")
        cues.append(
            {
                "seq": seq,
                "startSec": _timestamp(start_s),
                "endSec": _timestamp(end_s),
                "text": "\n".join(body).strip(),
            }
        )
    return cues


def re_split_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]


def _int_or(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _timestamp(value: str) -> float:
    clock = value.strip().replace(".", ",")
    hms, ms = clock.split(",")
    hours, minutes, seconds = [int(part) for part in hms.split(":")]
    return hours * 3600 + minutes * 60 + seconds + int(ms) / 1000.0


def delete_translation_output(job: dict[str, Any]) -> None:
    """删除这条翻译任务对应的字幕。转写结果和别的模型、语言不动。"""
    if job.get("kind") != "translate" or not job.get("target_lang"):
        return
    try:
        video = media.resolve_media(job["video_path"])
    except media.MediaError:
        return
    path = media.translation_path(
        video,
        int(job["stream_index"]),
        job["language"],
        job["target_lang"],
        _job_model(job),
    )
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)


def srt_response_path(job: dict[str, Any]) -> Path:
    video = media.resolve_media(job["video_path"])
    if job["kind"] == "translate" and job["target_lang"]:
        path = media.translation_path(
            video, int(job["stream_index"]), job["language"], job["target_lang"], _job_model(job)
        )
    else:
        path = media.artifact_path(video, int(job["stream_index"]), job["language"], "srt")
    partial = path.with_suffix(path.suffix + ".partial")
    if path.is_file():
        return path
    if partial.is_file():
        return partial
    raise media.MediaError("字幕还没生成")


def _cue_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return text.count("-->")


def _spent_suffix(job: dict[str, Any]) -> str:
    text = _spent_between(job.get("created_at"), None)
    return f"，{text}" if text else ""


def _spent_between(start: str | None, end: str | None) -> str:
    from datetime import datetime

    if not start:
        return ""
    try:
        opened = datetime.fromisoformat(start)
        closed = datetime.fromisoformat(end) if end else datetime.now().astimezone()
    except ValueError:
        return ""
    seconds = max(0.0, (closed - opened).total_seconds())
    if seconds < 60:
        return f"用时 {seconds:.0f} 秒"
    return f"用时 {int(seconds // 60)} 分 {int(seconds % 60)} 秒"


def job_log(job: dict[str, Any]) -> str:
    stored = str(job.get("log") or "").strip()
    if stored:
        return stored
    return _legacy_log(job)


def _legacy_log(job: dict[str, Any]) -> str:
    lines = ["这条记录是加上日志之前跑的，下面按影片旁边现在的文件还原。"]
    spent = _spent_between(job.get("created_at"), job.get("finished_at"))
    if job.get("created_at"):
        lines.append(
            f"开始 {job.get('created_at')}，结束 {job.get('finished_at') or '未结束'}"
            + (f"，{spent}。" if spent else "。")
        )
    try:
        video = media.resolve_media(job["video_path"])
    except media.MediaError as exc:
        lines.append(str(exc))
        return "\n".join(lines)
    index = int(job["stream_index"])
    language = job["language"]
    if job["kind"] == "translate":
        target_lang = job.get("target_lang") or ""
        source_label = media.language_label(language)
        target_label = media.language_label(target_lang)
        target = media.translation_path(video, index, language, target_lang, _job_model(job))
        if target.is_file():
            count = _cue_count(target)
            if media.short_lang(language) == target_lang:
                lines.append(
                    f"结果文件 {target.name}，共 {count} 条。音轨是{source_label}，目标也是{target_label}，不会调用翻译模型。"
                    "文件如果当时已经在，连复制都跳过。所以进度会一下子到头。"
                )
            else:
                lines.append(
                    f"结果文件 {target.name}，共 {count} 条。文件如果在任务开始前就存在，会直接跳过翻译，进度一下子是 {count}/{count}。"
                )
        else:
            lines.append("现在找不到翻译结果文件。")
    else:
        srt_path = media.artifact_path(video, index, language, "srt")
        if srt_path.is_file():
            count = _cue_count(srt_path)
            lines.append(
                f"原语言字幕 {srt_path.name}，共 {count} 条。发现这个文件已在会跳过模型，进度一下子是 {count}/{count}。"
            )
        else:
            lines.append("现在找不到原语言字幕。")
    if job.get("error"):
        lines.append(f"错误：{job['error']}")
    return "\n".join(lines)
