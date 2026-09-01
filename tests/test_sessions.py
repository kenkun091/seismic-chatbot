import os
import threading

import pytest

from interfaces.sessions import (SessionStore, SessionNotFound, SessionBusy, SessionLimit)


class _Ctx:
    def __init__(self):
        self.d = {}

    def get_context(self, k, default=None):
        return self.d.get(k, default)

    def set_context(self, k, v):
        self.d[k] = v


class _Session:
    _n = 0

    def __init__(self):
        _Session._n += 1
        self.session_id = f"sid{_Session._n}"
        self.context_manager = _Ctx()


class _Base:
    def new_session(self):
        return _Session()


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    return SessionStore(_Base(), ttl_seconds=100, max_sessions=2,
                        upload_dir=str(tmp_path), clock=_Clock())


def test_create_get_delete(store):
    e = store.create()
    assert store.get(e.bot.session_id) is e and len(store) == 1
    store.delete(e.bot.session_id)
    assert len(store) == 0
    with pytest.raises(SessionNotFound):
        store.get(e.bot.session_id)
    with pytest.raises(SessionNotFound):
        store.delete(e.bot.session_id)


def test_cap_and_sweep(store):
    a = store.create(); b = store.create()
    with pytest.raises(SessionLimit):
        store.create()
    store._clock.t += 101            # both idle past ttl
    assert sorted(store.sweep()) == sorted([a.bot.session_id, b.bot.session_id])
    assert len(store) == 0
    store.create()                   # room again after sweep


def test_delete_removes_upload_dir_and_plots(store, tmp_path):
    e = store.create()
    sub = tmp_path / e.bot.session_id
    sub.mkdir()
    (sub / "photo.png").write_bytes(b"x")
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"y")
    assert store.register_file(e, str(plot)) == "plot.png"
    store.delete(e.bot.session_id)
    assert not sub.exists() and not plot.exists()


def test_acquire_updates_last_used_and_version_on_context_change(store):
    e = store.create()
    v0 = e.version
    store._clock.t += 5
    with store.acquire(e.bot.session_id) as entry:
        assert entry is e
    assert e.last_used == store._clock.t and e.version == v0   # nothing changed
    with store.acquire(e.bot.session_id) as entry:
        entry.bot.context_manager.set_context("last_outcrop", {"regions": []})
    assert e.version == v0 + 1
    with store.acquire(e.bot.session_id):
        pass
    assert e.version == v0 + 1                                # same object → no bump


def test_acquire_is_exclusive(store):
    e = store.create()
    with store.acquire(e.bot.session_id):
        with pytest.raises(SessionBusy):
            with store.acquire(e.bot.session_id):
                pass
    with store.acquire(e.bot.session_id):     # released after the block
        pass


def test_register_file_allowlists_by_basename(store, tmp_path):
    e = store.create()
    p = tmp_path / "a.png"; p.write_bytes(b"z")
    name = store.register_file(e, str(p))
    assert e.allowed_files[name] == str(p)
    assert str(p) in e.plot_files
    q = tmp_path / "photo.jpg"; q.write_bytes(b"z")
    store.register_file(e, str(q))
    assert str(q) not in e.plot_files       # only .png plots are cleanup targets


def test_delete_raises_busy_if_lock_held(store, tmp_path):
    """Deleting a session while its lock is held raises SessionBusy and leaves it intact."""
    e = store.create()
    sub = tmp_path / e.bot.session_id
    sub.mkdir()
    (sub / "photo.png").write_bytes(b"x")
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"y")
    store.register_file(e, str(plot))

    # Hold the lock and try to delete
    with store.acquire(e.bot.session_id):
        with pytest.raises(SessionBusy):
            store.delete(e.bot.session_id)

    # Verify the entry and files still exist
    assert store.get(e.bot.session_id) is e
    assert sub.exists() and plot.exists()


def test_sweep_skips_locked_entries(store, tmp_path):
    """sweep() with an expired-but-locked entry skips it, leaving files intact."""
    e = store.create()
    sub = tmp_path / e.bot.session_id
    sub.mkdir()
    (sub / "photo.png").write_bytes(b"x")
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"y")
    store.register_file(e, str(plot))

    # Advance time past TTL
    store._clock.t += 101

    # Manually hold the lock to simulate in-flight request
    e.lock.acquire()
    try:
        # sweep() should skip it
        swept = store.sweep()
        assert e.bot.session_id not in swept
        assert store.get(e.bot.session_id) is e
        assert sub.exists() and plot.exists()
    finally:
        e.lock.release()

    # Now sweep() should remove it
    swept = store.sweep()
    assert e.bot.session_id in swept
    assert not sub.exists() and not plot.exists()
