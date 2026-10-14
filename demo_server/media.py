"""影片目录、音轨、抽音频、MPEG-TS 转码。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]


def _media_root() -> Path:
    raw = os.environ.get("MEDIA_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("/DATA/videos")


MEDIA_ROOT = _media_root()
VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts"}
SIDECAR_RE = re.compile(
    r"\.a\d+\.[a-z0-9]{2,3}(\.[a-z]{2})?(\.m2m)?\.(wav|srt)$",
    re.IGNORECASE,
)
SUB_ID_RE = re.compile(
    r"^a(\d+)\.([a-z0-9]{2,3})(?:\.([a-z]{2})(?:\.(m2m))?)?$",
    re.IGNORECASE,
)

# 容器里的音轨标记 -> 翻译用的短码。文件名仍保留原始标记，例如 ita。
TAG_TO_SHORT = {
    "ita": "it",
    "it": "it",
    "chi": "zh",
    "zho": "zh",
    "zh": "zh",
    "eng": "en",
    "en": "en",
    "jpn": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
    "spa": "es",
    "es": "es",
    "fra": "fr",
    "fre": "fr",
    "fr": "fr",
    "deu": "de",
    "ger": "de",
    "de": "de",
    "rus": "ru",
    "ru": "ru",
    "tur": "tr",
    "tr": "tr",
    "und": "und",
}

SHORT_TO_NLLB = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "it": "ita_Latn",
}

LANGUAGES = [
    {"id": "zh", "label": "中文"},
    {"id": "en", "label": "英文"},
    {"id": "ko", "label": "韩语"},
    {"id": "ja", "label": "日语"},
    {"id": "es", "label": "西班牙语"},
    {"id": "fr", "label": "法语"},
    {"id": "de", "label": "德语"},
    {"id": "ru", "label": "俄语"},
    {"id": "tr", "label": "土耳其语"},
    {"id": "it", "label": "意大利语"},
]


class MediaError(Exception):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaError("找不到 ffmpeg")
    return path


def require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MediaError("找不到 ffprobe")
    return path


def resolve_media(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel or rel.startswith("..") or "/../" in f"/{rel}/":
        raise MediaError("路径不合法")
    root = MEDIA_ROOT
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise MediaError("路径超出媒体目录")
    return target


def relative_media(path: Path) -> str:
    return path.resolve().relative_to(MEDIA_ROOT).as_posix()


def is_sidecar(name: str) -> bool:
    return SIDECAR_RE.search(name) is not None


def list_dir(rel: str) -> dict:
    if not MEDIA_ROOT.is_dir():
        raise MediaError(f"媒体目录不存在: {MEDIA_ROOT}")
    folder = resolve_media(rel) if rel else MEDIA_ROOT
    if not folder.is_dir():
        raise MediaError(f"目录不存在: {folder}")
    skip = {"proc", "sys", "dev", "run"} if folder == Path("/") else set()
    entries = []
    try:
        children = list(folder.iterdir())
    except OSError as exc:
        raise MediaError(f"无法读取目录 {folder}: {exc}") from exc

    def sort_key(item: Path) -> tuple[bool, str]:
        try:
            is_dir = item.is_dir()
        except OSError:
            is_dir = False
        return (not is_dir, item.name.lower())

    for child in sorted(children, key=sort_key):
        if child.name.startswith(".") or child.name in skip:
            continue
        if is_sidecar(child.name):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append({"name": child.name, "path": relative_media(child), "kind": "dir"})
        elif child.suffix.lower() in VIDEO_SUFFIXES:
            try:
                size = child.stat().st_size
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": relative_media(child),
                    "kind": "video",
                    "size": size,
                }
            )
    return {
        "dir": "" if folder == MEDIA_ROOT else relative_media(folder),
        "root": str(MEDIA_ROOT),
        "absolute": str(folder),
        "entries": entries,
    }


def probe(path: Path) -> dict:
    cmd = [
        require_ffprobe(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaError(proc.stderr.strip() or "ffprobe 失败")
    return json.loads(proc.stdout or "{}")


def audio_tracks(path: Path, data: dict | None = None) -> list[dict]:
    if data is None:
        data = probe(path)
    tracks = []
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags") or {}
        language = str(tags.get("language") or "und").lower()
        index = int(stream["index"])
        tracks.append(
            {
                "streamIndex": index,
                "language": language,
                "languageShort": TAG_TO_SHORT.get(language, language),
                "title": tags.get("title") or "",
                "codec": stream.get("codec_name") or "",
                "channels": stream.get("channels"),
                "default": bool((stream.get("disposition") or {}).get("default")),
                "wavReady": artifact_path(path, index, language, "wav").is_file(),
                "srtReady": artifact_path(path, index, language, "srt").is_file(),
                "translations": ready_translations(path, index, language),
            }
        )
    return tracks


def subtitle_tracks(data: dict) -> list[dict]:
    tracks = []
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags") or {}
        tracks.append(
            {
                "streamIndex": int(stream["index"]),
                "language": str(tags.get("language") or "und").lower(),
                "title": tags.get("title") or "",
                "codec": stream.get("codec_name") or "",
            }
        )
    return tracks


def artifact_path(video: Path, stream_index: int, language: str, suffix: str) -> Path:
    safe_lang = re.sub(r"[^a-z0-9]", "", language.lower()) or "und"
    return video.with_name(f"{video.name}.a{stream_index}.{safe_lang}.{suffix}")


def translation_path(
    video: Path,
    stream_index: int,
    language: str,
    target_lang: str,
    model: str = "nllb",
) -> Path:
    safe_lang = re.sub(r"[^a-z0-9]", "", language.lower()) or "und"
    safe_target = re.sub(r"[^a-z]", "", target_lang.lower())
    name = f"{video.name}.a{stream_index}.{safe_lang}.{safe_target}"
    if model == "m2m":
        name += ".m2m"
    return video.with_name(f"{name}.srt")


def ready_translations(video: Path, stream_index: int, language: str) -> list[dict[str, str]]:
    found = []
    for item in LANGUAGES:
        for model in ("nllb", "m2m"):
            if translation_path(video, stream_index, language, item["id"], model).is_file():
                found.append({"lang": item["id"], "model": model})
    return found


def ai_subtitles(video: Path) -> list[dict]:
    prefix = video.name + "."
    rows = []
    if not video.parent.is_dir():
        return rows
    for child in sorted(video.parent.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_file() or not child.name.startswith(prefix):
            continue
        if child.suffix.lower() != ".srt":
            continue
        sub_id = child.name[len(prefix) : -4]
        match = SUB_ID_RE.match(sub_id)
        if not match:
            continue
        language = match.group(2).lower()
        target = match.group(3).lower() if match.group(3) else ""
        model = match.group(4).lower() if match.group(4) else ""
        label = language_label(target or language)
        if target and model == "m2m":
            label = f"{label} · M2M100"
        elif target:
            label = f"{label} · NLLB"
        else:
            label = f"{label}原语言 · AI"
        rows.append(
            {
                "id": sub_id,
                "streamIndex": int(match.group(1)),
                "language": language,
                "target": target or None,
                "label": label,
                "ai": True,
                "url": "/api/files/subtitles.vtt?" + urlencode({"path": relative_media(video), "id": sub_id}),
            }
        )
    return rows


def ai_subtitle_file(video: Path, sub_id: str) -> Path:
    if not SUB_ID_RE.match(sub_id):
        raise MediaError("字幕不存在")
    path = video.with_name(f"{video.name}.{sub_id}.srt")
    if not path.is_file():
        raise MediaError("字幕不存在")
    return path


def cues_to_vtt(cues: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for cue in cues:
        begin = float(cue["startSec"])
        end = float(cue["endSec"])
        text = str(cue.get("text") or "").replace("\r", "").strip()
        if not text or end <= begin:
            continue
        lines.append(f"{_vtt_clock(begin)} --> {_vtt_clock(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) + "\n"


def _vtt_clock(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def file_info(rel: str) -> dict:
    path = resolve_media(rel)
    if not path.is_file():
        raise MediaError("影片不存在")
    data = probe(path)
    duration = float((data.get("format") or {}).get("duration") or 0)
    return {
        "path": relative_media(path),
        "name": path.name,
        "durationSec": duration,
        "tracks": audio_tracks(path, data),
        "subtitles": subtitle_tracks(data),
        "aiSubtitles": ai_subtitles(path),
    }


def extract_wav(video: Path, stream_index: int, wav_path: Path, cancel: callable) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        require_ffmpeg(),
        "-y",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    while proc.poll() is None:
        if cancel():
            proc.kill()
            proc.wait(timeout=5)
            wav_path.unlink(missing_ok=True)
            raise MediaError("已停止")
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            continue
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace").strip()
        wav_path.unlink(missing_ok=True)
        raise MediaError(err or "抽取音频失败")


def _encoder_plans(start: float, video: Path, stream_index: int) -> list[list[str]]:
    ffmpeg = require_ffmpeg()
    maps = ["-map", "0:v:0", "-map", f"0:{stream_index}"]
    # 时间戳从 0 起。浏览器的 MediaSource 接不住影片原始时间戳，会把缓冲区拆掉。
    tail = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-f", "mpegts", "-muxdelay", "0", "-muxpreload", "0", "pipe:1"]
    seek = ["-ss", f"{start:.3f}", "-i", str(video)]
    plans = []
    if os.name != "nt" and Path("/dev/dri/renderD128").exists():
        plans.append(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-vaapi_device",
                "/dev/dri/renderD128",
                *seek,
                *maps,
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-bf",
                "0",
                "-g",
                "48",
                *tail,
            ]
        )
    if os.name == "nt":
        plans.append(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                *seek,
                *maps,
                "-c:v",
                "h264_qsv",
                "-bf",
                "0",
                "-g",
                "48",
                *tail,
            ]
        )
    plans.append(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            *seek,
            *maps,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "48",
            "-keyint_min",
            "48",
            "-bf",
            "0",
            *tail,
        ]
    )
    return plans


async def open_mpegts(start: float, video: Path, stream_index: int) -> asyncio.subprocess.Process:
    last_error = "ffmpeg 没有输出"
    for cmd in _encoder_plans(start, video, stream_index):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        first = await proc.stdout.read(1)
        if first:
            proc._first_byte = first  # type: ignore[attr-defined]
            return proc
        err = b""
        if proc.stderr:
            err = await proc.stderr.read()
        await proc.wait()
        last_error = err.decode("utf-8", "replace").strip() or last_error
    raise MediaError(last_error)


def language_label(tag: str) -> str:
    short = TAG_TO_SHORT.get((tag or "").lower(), (tag or "").lower())
    for item in LANGUAGES:
        if item["id"] == short:
            return item["label"]
    if not tag or tag == "und":
        return "未知语言"
    return tag


def short_lang(tag: str) -> str:
    return TAG_TO_SHORT.get(tag.lower(), tag.lower())


def nllb_code(short: str) -> str:
    try:
        return SHORT_TO_NLLB[short]
    except KeyError as exc:
        raise MediaError(f"不支持的语言: {short}") from exc
