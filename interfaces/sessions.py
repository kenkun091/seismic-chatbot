"""Server-owned chat sessions for the browser client.

One SeismicChatBotToolUse session per id (shared heavy components, fresh
ContextManager), a per-session lock (the tool loop is not concurrency-safe),
an allow-list of files the file route may serve, a version counter that
tracks changes to the outcrop context keys, and an idle TTL."""
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_TRACKED_KEYS = ("last_outcrop", "last_earth_model", "last_section")


class SessionNotFound(KeyError):
    pass


class SessionBusy(RuntimeError):
    pass


class SessionLimit(RuntimeError):
    pass


@dataclass
class SessionEntry:
    bot: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    created: float = 0.0
    last_used: float = 0.0
    allowed_files: Dict[str, str] = field(default_factory=dict)
    plot_files: List[str] = field(default_factory=list)
    version: int = 0


class SessionStore:
    def __init__(self, base_bot: Any, ttl_seconds: float = 7200.0, max_sessions: int = 50,
                 upload_dir: Optional[str] = None, clock: Callable[[], float] = time.time):
        self._base = base_bot
        self._ttl = float(ttl_seconds)
        self._max = int(max_sessions)
        self._upload_dir = upload_dir
        self._clock = clock
        self._entries: Dict[str, SessionEntry] = {}
        self._guard = threading.Lock()   # protects _entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> SessionEntry:
        self.sweep()
        with self._guard:
            if len(self._entries) >= self._max:
                raise SessionLimit(f"session limit reached ({self._max})")
            bot = self._base.new_session()
            now = self._clock()
            entry = SessionEntry(bot=bot, created=now, last_used=now)
            self._entries[bot.session_id] = entry
            return entry

    def get(self, session_id: str) -> SessionEntry:
        try:
            return self._entries[session_id]
        except KeyError:
            raise SessionNotFound(session_id)

    def delete(self, session_id: str) -> None:
        with self._guard:
            try:
                entry = self._entries[session_id]
            except KeyError:
                raise SessionNotFound(session_id)
            if not entry.lock.acquire(blocking=False):
                raise SessionBusy(session_id)
            try:
                self._entries.pop(session_id)
            except KeyError:
                pass
        try:
            self._cleanup(session_id, entry)
        finally:
            entry.lock.release()

    def sweep(self) -> List[str]:
        now = self._clock()
        expired: List[Tuple[str, SessionEntry]] = []
        with self._guard:
            for sid, entry in list(self._entries.items()):
                if now - entry.last_used > self._ttl:
                    if entry.lock.acquire(blocking=False):
                        try:
                            self._entries.pop(sid)
                            expired.append((sid, entry))
                        except KeyError:
                            pass
        for sid, entry in expired:
            try:
                self._cleanup(sid, entry)
            finally:
                entry.lock.release()
        return [sid for sid, _ in expired]

    def _cleanup(self, session_id: str, entry: SessionEntry) -> None:
        if self._upload_dir:
            shutil.rmtree(os.path.join(self._upload_dir, session_id), ignore_errors=True)
        for path in entry.plot_files:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- per-request access ------------------------------------------------
    @staticmethod
    def identity_snapshot(entry: SessionEntry) -> Tuple[int, ...]:
        cm = entry.bot.context_manager
        return tuple(id(cm.get_context(k)) for k in _TRACKED_KEYS)

    @contextmanager
    def acquire(self, session_id: str) -> Iterator[SessionEntry]:
        with self._guard:
            try:
                entry = self._entries[session_id]
            except KeyError:
                raise SessionNotFound(session_id)
            if not entry.lock.acquire(blocking=False):
                raise SessionBusy(session_id)
        before = self.identity_snapshot(entry)
        try:
            yield entry
        finally:
            if self.identity_snapshot(entry) != before:
                entry.version += 1
            entry.last_used = self._clock()
            entry.lock.release()

    @staticmethod
    def register_file(entry: SessionEntry, path: str) -> str:
        name = os.path.basename(path)
        entry.allowed_files[name] = path
        if path.endswith(".png") and path not in entry.plot_files:
            entry.plot_files.append(path)
        return name
