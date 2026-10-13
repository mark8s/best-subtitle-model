"""本地字幕演示服务。

网页和安卓电视共用这一套接口。网页播片把 mkv 转成 MPEG-TS，
安卓在请求头里带 X-Client-Type: android，直接拿原文件。

启动（在 ov-venv 里）。默认读取 /DATA/videos，访问日志在 logs/demo.log::

    python -m demo_server
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from demo_server import db, engine, media

STATIC = media.ROOT / "demo_server" / "static"
app = FastAPI(title="subtitle demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def startup() -> None:
    db.init()
    engine.hub.loop = asyncio.get_event_loop()
    print(f"MEDIA_ROOT : {media.MEDIA_ROOT}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    devices: list[str] = []
    try:
        import openvino as ov

        devices = list(ov.Core().available_devices)
    except Exception:
        devices = []
    return {
        "ok": True,
        "mediaRoot": str(media.MEDIA_ROOT),
        "devices": devices,
        "models": engine.models_ready(),
        "busy": engine.job_busy(),
    }


@app.get("/api/languages")
def languages() -> list[dict[str, str]]:
    return media.LANGUAGES


@app.get("/api/files")
def files(path: str = "") -> dict[str, Any]:
    try:
        return media.list_dir(path)
    except media.MediaError as exc:
        print(f"/api/files {path!r}: {exc}")
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/files/info")
def file_info(path: str) -> dict[str, Any]:
    try:
        return media.file_info(path)
    except media.MediaError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/files/subtitles")
def list_subtitles(path: str) -> dict[str, Any]:
    try:
        video = media.resolve_media(path)
    except media.MediaError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not video.is_file():
        raise HTTPException(404, "影片不存在")
    return {"subtitles": media.ai_subtitles(video)}


@app.get("/api/files/subtitles.vtt")
def subtitle_vtt(path: str, id: str) -> Response:
    try:
        video = media.resolve_media(path)
        srt = media.ai_subtitle_file(video, id)
        body = media.cues_to_vtt(engine.parse_srt(srt))
    except media.MediaError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(body, media_type="text/vtt; charset=utf-8")


@app.get("/api/files/stream")
async def stream(path: str, request: Request, start: float = 0, streamIndex: int | None = None):
    client = (request.headers.get("x-client-type") or "web").lower()
    try:
        video = media.resolve_media(path)
    except media.MediaError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not video.is_file():
        raise HTTPException(404, "影片不存在")
    if client == "android":
        return FileResponse(video, filename=video.name)
    if streamIndex is None:
        audio_index = _default_audio_index(video)
    else:
        audio_index = streamIndex
    try:
        proc = await media.open_mpegts(max(0.0, start), video, audio_index)
    except media.MediaError as exc:
        raise HTTPException(500, str(exc)) from exc
    return StreamingResponse(_mpegts(proc), media_type="video/mp2t")


async def _mpegts(proc):
    stderr_task = asyncio.create_task(_drain(proc.stderr))
    first = getattr(proc, "_first_byte", b"")
    try:
        if first:
            yield first
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        stderr_task.cancel()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _drain(stream) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break


def _default_audio_index(video) -> int:
    tracks = media.audio_tracks(video)
    if not tracks:
        raise HTTPException(400, "没有音轨")
    default = next((track for track in tracks if track["default"]), tracks[0])
    return int(default["streamIndex"])


@app.post("/api/jobs")
def create_job(body: dict[str, Any], request: Request) -> dict[str, Any]:
    if engine.job_busy() or db.running_job():
        raise HTTPException(409, "已有任务在跑")
    kind = body.get("mode") or body.get("kind")
    if kind not in {"transcribe", "translate"}:
        raise HTTPException(400, "mode 只能是 transcribe 或 translate")
    rel = str(body.get("path") or "")
    try:
        info = media.file_info(rel)
    except media.MediaError as exc:
        raise HTTPException(400, str(exc)) from exc
    tracks = info["tracks"]
    if not tracks:
        raise HTTPException(400, "没有音轨")
    requested = body.get("streamIndex")
    track = next((item for item in tracks if item["streamIndex"] == requested), None)
    if track is None:
        track = next((item for item in tracks if item["default"]), tracks[0])
    target = body.get("targetLang")
    if kind == "translate":
        if target not in {item["id"] for item in media.LANGUAGES}:
            raise HTTPException(400, "要选择字幕语言")
        if not track["srtReady"]:
            raise HTTPException(400, "还没有原语言字幕，请先把音频转成文字")
    client = (request.headers.get("x-client-type") or "web").lower()
    if kind == "transcribe":
        client_type = None
    else:
        client_type = "android" if client == "android" else "web"
    video = media.resolve_media(info["path"])
    index = int(track["streamIndex"])
    language = track["language"]
    if kind == "transcribe":
        if media.artifact_path(video, index, language, "srt").is_file():
            raise HTTPException(400, "原语言字幕已存在，无需再生成")
    else:
        if media.short_lang(language) == target:
            raise HTTPException(400, "源语言和目标相同，无需翻译")
        if media.translation_path(video, index, language, target).is_file():
            raise HTTPException(400, "该语言字幕已存在，无需再翻译")
    job = db.create_job(
        kind=kind,
        video_path=info["path"],
        stream_index=index,
        language=language,
        target_lang=target if kind == "translate" else None,
        client_type=client_type,
    )
    try:
        engine.start_job(job)
    except RuntimeError as exc:
        db.finish_job(job["id"], "failed", str(exc))
        raise HTTPException(409, str(exc)) from exc
    return _public_job(job["id"])


@app.get("/api/jobs")
def jobs() -> dict[str, Any]:
    rows = [_public_job(row["id"]) for row in db.list_jobs()]
    return {"jobs": rows}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    data = _public_job(job_id)
    data["log"] = engine.job_log(job)
    return data


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, remove: int = 0) -> dict[str, str]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if remove:
        if job["status"] == "running":
            engine.cancel_job(job_id)
        db.remove_job(job_id)
        return {"status": "deleted"}
    if job["status"] != "running":
        raise HTTPException(400, "任务没在跑")
    engine.cancel_job(job_id)
    return {"status": "stopped"}


@app.get("/api/jobs/{job_id}/subtitles.srt")
def download_srt(job_id: str) -> FileResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    try:
        path = engine.srt_response_path(job)
    except media.MediaError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="application/x-subrip", filename=path.name)


@app.websocket("/api/jobs/{job_id}")
async def job_socket(socket: WebSocket, job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        await socket.close(code=4404)
        return
    await socket.accept()
    queue = engine.hub.subscribe(job_id)
    sent = 0
    try:
        raw = await asyncio.wait_for(socket.receive_text(), timeout=2)
        message = json.loads(raw)
        if message.get("type") == "resume":
            sent = int(message.get("after") or 0)
        elif message.get("type") == "position":
            engine.note_position(job_id, float(message.get("positionSec") or 0))
    except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
        pass
    try:
        path = engine.srt_response_path(job)
        for cue in engine.parse_srt(path):
            if cue["seq"] <= sent:
                continue
            await socket.send_json({"type": "cue", **cue})
            sent = cue["seq"]
    except media.MediaError:
        pass
    current = db.get_job(job_id)
    if current and current["status"] != "running":
        await socket.send_json({"type": "status", "status": current["status"]})
        engine.hub.unsubscribe(job_id, queue)
        return
    try:
        while True:
            receive = asyncio.create_task(socket.receive_text())
            incoming = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait({receive, incoming}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if receive in done:
                message = json.loads(receive.result())
                if message.get("type") == "position":
                    engine.note_position(job_id, float(message.get("positionSec") or 0))
            if incoming in done:
                event = incoming.result()
                seq = int(event.get("seq") or 0)
                if event.get("type") == "cue" and seq <= sent:
                    continue
                if event.get("type") == "cue":
                    sent = seq
                await socket.send_json(event)
                if event.get("type") in {"done", "error"}:
                    break
            current = db.get_job(job_id)
            if current and current["status"] != "running" and queue.empty():
                await socket.send_json({"type": "status", "status": current["status"]})
                break
    except WebSocketDisconnect:
        pass
    finally:
        engine.hub.unsubscribe(job_id, queue)


def _public_job(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    assert job is not None
    position = engine.position_of(job_id)
    return {
        "id": job["id"],
        "kind": job["kind"],
        "videoPath": job["video_path"],
        "streamIndex": job["stream_index"],
        "language": job["language"],
        "targetLang": job["target_lang"],
        "clientType": job["client_type"],
        "status": job["status"],
        "stage": job["stage"],
        "doneCount": job["done_count"],
        "totalCount": job["total_count"],
        "error": job["error"],
        "positionSec": position,
        "createdAt": job["created_at"],
        "updatedAt": job["updated_at"],
        "finishedAt": job["finished_at"],
    }
