"""Session-scoped REST API for the outcrop → seismic web client.

Every tool route runs the registry tool through ToolLoopRunner.execute_call —
the same per-call path as a chat turn — so validators, physics guards,
sandboxes, trace events and provenance apply. Files are served only when
registered on the session."""
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from interfaces.serialize import (interpretation_caps, model_summary, section_payload,
                                  to_jsonable)
from interfaces.sessions import (SessionBusy, SessionEntry, SessionLimit,
                                 SessionNotFound, SessionStore)
from tools.image_safety import MIME_BY_EXTENSION, image_size, stage_upload
from tools.outcrop_tools import validate_interpretation

logger = logging.getLogger(__name__)

MAX_INTERPRETATION_BYTES = 1_000_000


def _http(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": message})


def _image_info(entry: SessionEntry, sid: str) -> Optional[Dict[str, Any]]:
    path = entry.bot.context_manager.get_context("last_image")
    if not path:
        return None
    w, h = image_size(path)
    name = os.path.basename(path)
    return {"width": w, "height": h, "url": f"/sessions/{sid}/files/{name}"}


def build_router(store: SessionStore, auth_dependency: Callable, upload_dir: str,
                 max_image_mb: float = 10.0) -> APIRouter:
    router = APIRouter(prefix="/sessions", dependencies=[Depends(auth_dependency)])

    @contextmanager
    def session(sid: str) -> Iterator[SessionEntry]:
        try:
            with store.acquire(sid) as entry:
                yield entry
        except SessionNotFound:
            raise _http(404, f"unknown session {sid}")
        except SessionBusy:
            raise _http(409, "session is busy with another request")

    # -- lifecycle ---------------------------------------------------------
    @router.post("", status_code=201)
    def create_session():
        try:
            entry = store.create()
        except SessionLimit as e:
            raise _http(503, str(e))
        return {"session_id": entry.bot.session_id}

    @router.delete("/{sid}", status_code=204)
    def delete_session(sid: str):
        try:
            store.delete(sid)
        except SessionNotFound:
            raise _http(404, f"unknown session {sid}")
        except SessionBusy:
            raise _http(409, "session is busy with another request")
        return Response(status_code=204)

    # -- upload + files ----------------------------------------------------
    @router.post("/{sid}/image")
    def upload_image(sid: str, file: UploadFile = File(...)):
        with session(sid) as entry:
            suffix = os.path.splitext(file.filename or "")[1].lower()
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            try:
                max_bytes = max_image_mb * 1024 * 1024
                total = 0
                with os.fdopen(fd, "wb") as out:
                    while True:
                        chunk = file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise _http(413, f"upload exceeds {max_image_mb:g} MB")
                        out.write(chunk)
                try:
                    staged = stage_upload(tmp, upload_dir, entry.bot.session_id, max_image_mb)
                except ValueError as e:
                    raise _http(400, str(e))
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            entry.bot.attach_image(staged)
            store.register_file(entry, staged)
            return _image_info(entry, sid)

    @router.get("/{sid}/files/{name}")
    def get_file(sid: str, name: str):
        with session(sid) as entry:
            path = entry.allowed_files.get(name)
            if path is None or os.path.basename(name) != name or not os.path.isfile(path):
                raise _http(404, "no such file on this session")
            ext = os.path.splitext(path)[1].lower()
            return FileResponse(path, media_type=MIME_BY_EXTENSION.get(ext, "image/png"))

    # -- tool execution ----------------------------------------------------
    def _run_tool(entry: SessionEntry, name: str, args: Dict[str, Any]):
        """execute_call with the per-turn bookkeeping a chat turn does; returns
        (result, warnings). Maps tool errors to HTTP statuses."""
        cm = entry.bot.context_manager
        cm.trace.begin_turn(f"api:{name}")
        cm.begin_turn_recording(f"api:{name}")
        images: list = []
        try:
            result = entry.bot._tool_loop.execute_call(name, args, images, auto_plot=False)
        except ValueError as e:
            raise _http(400, str(e))
        except RuntimeError as e:
            msg = str(e)
            if "vision" in msg.lower() or "credential" in msg.lower():
                raise _http(503, msg)
            raise _http(500, msg)
        finally:
            events = list(cm.trace.events)
            cm.trace.end_turn()
        warnings = [e.get("message", "") for e in events if e.get("t") == "physics_warning"]
        for p in images:
            store.register_file(entry, p)
        return result, warnings

    def _interp_public(interp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(interp, dict):
            return None
        return to_jsonable({k: v for k, v in interp.items() if k != "image_path"})

    # -- interpretation ------------------------------------------------------
    @router.post("/{sid}/interpret")
    def interpret(sid: str):
        with session(sid) as entry:
            result, warnings = _run_tool(entry, "interpret_outcrop", {})
        entry = store.get(sid)
        return {"interpretation": _interp_public(result), "warnings": warnings,
                "version": entry.version}

    @router.put("/{sid}/interpretation")
    async def put_interpretation(sid: str, request: Request):
        raw = await request.body()
        if len(raw) > MAX_INTERPRETATION_BYTES:
            raise _http(413, "interpretation body exceeds 1 MB")
        try:
            data = json.loads(raw)
        except ValueError:
            raise _http(400, "body is not valid JSON")
        try:
            interpretation_caps(data)
        except ValueError as e:
            raise _http(413, str(e))
        with session(sid) as entry:
            cm = entry.bot.context_manager
            try:
                normalized = validate_interpretation(data)
            except ValueError as e:
                raise _http(400, str(e))
            image = cm.get_context("last_image")
            if image:
                normalized["image_path"] = image
                normalized["image_size"] = list(image_size(image))
            cm.set_context("last_outcrop", normalized)
        entry = store.get(sid)
        return {"interpretation": _interp_public(normalized), "warnings": [],
                "version": entry.version}

    # -- state -------------------------------------------------------------
    @router.get("/{sid}/state")
    def get_state(sid: str):
        with session(sid) as entry:
            return _state(entry, sid)

    def _state(entry: SessionEntry, sid: str) -> Dict[str, Any]:
        cm = entry.bot.context_manager
        return {"version": entry.version,
                "image": _image_info(entry, sid),
                "interpretation": _interp_public(cm.get_context("last_outcrop")),
                "model_summary": None,
                "section_meta": None}

    router.state_builder = _state   # used by later routes to return state with results
    return router


def install_error_handlers(app) -> None:
    """Render HTTPException details consistently.

    A dict detail (our own `_http` helper) becomes the JSON body verbatim
    (e.g. `{"error": ...}`). A plain string detail — as raised by callers
    outside this module, such as the legacy `/chat` auth gate — is preserved
    as `{"detail": ...}`, matching FastAPI's default shape."""
    from fastapi.exceptions import HTTPException as _HTTPException

    @app.exception_handler(_HTTPException)
    async def _handler(request: Request, exc: _HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=detail)
