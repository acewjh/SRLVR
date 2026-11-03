"""Parallel ALFWorld environment actors (one persistent process per env).

Background
----------
TextWorld's PDDL parser is a module-level mutable singleton, so two threads that
construct/reset/step environments in one interpreter corrupt each other (see the
long note in ``tasks/alfworld/v1/env_adapter.py``).  The previous in-process path
therefore serialised *every* environment operation behind ``_TEXTWORLD_LOCK``,
which capped rollout throughput at one env step at a time no matter how many
episodes were "concurrently" generated.

This module gives each environment its own interpreter.  A pool of persistent
worker processes (``tasks/alfworld_env_worker.py``) serves
``CREATE / RESET / STEP / CLOSE`` over a duplex pipe; the client side is
asyncio-friendly: the blocking pipe round-trip runs in a dedicated thread, so the
event loop (and therefore generation) never blocks on the environment.

Design rules
------------
* Workers are spawned with ``multiprocessing`` ``spawn`` (fresh interpreter, no
  inherited torch/NPU/SGLang state) and reused across rollouts.  They are only
  replaced after a crash or an operation timeout.
* Exactly one episode owns a worker while it runs (1:1 with the concurrent
  episode count by default), so no state is shared between episodes.
* Every request carries a monotonically increasing id and replies are matched on
  it; a reply from an abandoned call can never be mistaken for a newer one.
* A call that times out or whose worker dies marks the worker dead, kills it,
  and spawns a replacement.  The affected episode raises; state is never
  silently resumed on a possibly-corrupted environment.
* Workers are daemonic and also exit when the parent pid changes, so no orphan
  interpreters survive a parent crash.

Environment knobs
-----------------
``ALFWORLD_ENV_ACTORS``       ``1`` (default) to use actor processes, ``0`` for
                              the legacy in-process/threaded path.
``ALFWORLD_ENV_WORKERS``      pool size (default 32).
``ALFWORLD_ENV_CREATE_TIMEOUT`` seconds allowed for CREATE, incl. first import
                              (default 180).
``ALFWORLD_ENV_OP_TIMEOUT``   seconds allowed per RESET/STEP (default 90).
``ALFWORLD_ENV_CLOSE_TIMEOUT`` seconds allowed for CLOSE (default 30).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "ActorEnv",
    "EnvActorError",
    "EnvActorTimeout",
    "acquire_env",
    "env_actors_enabled",
    "get_pool",
    "shutdown_pool",
]

logger = logging.getLogger(__name__)


class EnvActorError(RuntimeError):
    """The env actor failed, died, or returned an error."""


class EnvActorTimeout(EnvActorError):
    """An env actor operation exceeded its deadline."""


def env_actors_enabled() -> bool:
    return os.environ.get("ALFWORLD_ENV_ACTORS", "1").lower() not in ("0", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class _Worker:
    """Parent-side handle to one env actor process."""

    def __init__(self, ctx, index: int) -> None:
        from tasks.alfworld_env_worker import worker_main

        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self.index = index
        self._conn = parent_conn
        self._seq = 0
        self._lock = threading.Lock()
        self.dead = False
        self.proc = ctx.Process(
            target=worker_main,
            args=(child_conn, os.getpid()),
            name=f"alfworld-env-{index}",
            daemon=True,
        )
        self.proc.start()
        child_conn.close()

    # -- synchronous IPC (always called from the pool's executor) -----------
    def call_sync(self, op: str, args: tuple = (), *, timeout: float):
        if self.dead:
            raise EnvActorError(f"env actor {self.index} is dead")
        with self._lock:
            if self.dead:
                raise EnvActorError(f"env actor {self.index} is dead")
            self._seq += 1
            req = self._seq
            try:
                self._conn.send((op, req, args))
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.dead = True
                raise EnvActorError(f"env actor {self.index} send failed: {exc}") from exc

            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.dead = True
                    raise EnvActorTimeout(
                        f"env actor {self.index} op {op} exceeded {timeout:.0f}s"
                    )
                try:
                    ready = self._conn.poll(remaining)
                except (OSError, ValueError) as exc:
                    self.dead = True
                    raise EnvActorError(f"env actor {self.index} poll failed: {exc}") from exc
                if not ready:
                    self.dead = True
                    raise EnvActorTimeout(
                        f"env actor {self.index} op {op} exceeded {timeout:.0f}s"
                    )
                try:
                    status, reply_req, payload = self._conn.recv()
                except (EOFError, OSError, ValueError) as exc:
                    self.dead = True
                    raise EnvActorError(f"env actor {self.index} exited during {op}") from exc
                if reply_req != req:
                    # A late reply from a previous (abandoned) call. It can never
                    # be the answer to this one; keep waiting for our id.
                    logger.warning(
                        "[env-actors] dropped stale reply req=%s while waiting req=%s",
                        reply_req, req,
                    )
                    continue
                if status == "OK":
                    return payload
                raise EnvActorError(f"env actor {self.index} {op} failed: {payload}")

    def kill(self) -> None:
        self.dead = True
        try:
            self.proc.terminate()
        except Exception:  # noqa: BLE001 - best effort teardown
            pass
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    def join(self, timeout: float = 5.0) -> None:
        try:
            self.proc.join(timeout)
        except Exception:  # noqa: BLE001
            pass


class EnvActorPool:
    """Fixed-size pool of persistent env-actor processes."""

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self.workers: list[_Worker] = []
        self._ctx = mp.get_context("spawn")
        self._executor = ThreadPoolExecutor(
            max_workers=self.size, thread_name_prefix="env-actor-ipc"
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None
        self._idle: asyncio.Queue | None = None
        self._closing = False
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        self._sem = asyncio.Semaphore(self.size)
        self._idle = asyncio.Queue()
        for index in range(self.size):
            self._spawn(index)
        self._started = True
        logger.info(
            "[env-actors] started %d persistent TextWorld workers "
            "(create_timeout=%ss op_timeout=%ss)",
            self.size, _env_int("ALFWORLD_ENV_CREATE_TIMEOUT", 180),
            _env_int("ALFWORLD_ENV_OP_TIMEOUT", 90),
        )

    def _spawn(self, index: int) -> _Worker | None:
        worker = None
        try:
            worker = _Worker(self._ctx, index)
        except Exception:  # noqa: BLE001 - spawn failure must not kill the pool
            logger.exception("[env-actors] failed to spawn worker %d", index)
            return None
        self.workers.append(worker)
        assert self._idle is not None
        self._idle.put_nowait(worker)
        return worker

    # -- worker calls ------------------------------------------------------
    async def _call(self, worker: _Worker, op: str, args: tuple, timeout: float):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: worker.call_sync(op, args, timeout=timeout),
        )

    # -- public API --------------------------------------------------------
    async def acquire(self, game, max_steps: int) -> "ActorEnv":
        if not self._started:
            self._ensure_started()
        assert self._sem is not None and self._idle is not None and self._loop is not None
        if asyncio.get_running_loop() is not self._loop:
            raise EnvActorError("env actor pool used from a different event loop")

        await self._sem.acquire()
        worker = await self._idle.get()
        if worker.dead:
            # Raced with a crash; take another worker and discard this one.
            await self._discard(worker)
            worker = await self._idle.get()
        lease = ActorEnv(self, worker)
        try:
            await self._call(
                worker, "CREATE", (game, int(max_steps)),
                _env_int("ALFWORLD_ENV_CREATE_TIMEOUT", 180),
            )
        except BaseException:
            await self._discard(worker)
            raise
        return lease

    async def _release(self, worker: _Worker, *, dirty: bool) -> None:
        assert self._sem is not None
        try:
            if dirty or worker.dead:
                await self._discard(worker)
            else:
                try:
                    await self._call(
                        worker, "CLOSE", (),
                        _env_int("ALFWORLD_ENV_CLOSE_TIMEOUT", 30),
                    )
                except Exception:  # noqa: BLE001 - a bad worker is replaced
                    logger.warning("[env-actors] CLOSE failed; replacing worker %d", worker.index)
                    await self._discard(worker)
                else:
                    self._idle.put_nowait(worker)
        finally:
            self._sem.release()

    async def _discard(self, worker: _Worker) -> None:
        """Kill a worker (after a crash/timeout/error) and spawn a replacement."""
        worker.kill()
        if worker in self.workers:
            self.workers.remove(worker)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, worker.join, 5.0)
        if not self._closing:
            index = worker.index
            self._spawn(index)

    def shutdown(self) -> None:
        self._closing = True
        for worker in list(self.workers):
            worker.kill()
        for worker in list(self.workers):
            worker.join(2.0)
        self.workers.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)


class ActorEnv:
    """Lease of one env actor: async RESET/STEP/CLOSE with in-process semantics."""

    def __init__(self, pool: EnvActorPool, worker: _Worker) -> None:
        self._pool = pool
        self._worker = worker
        self._released = False
        self._dirty = False
        self.task_description = ""

    async def reset(self):
        try:
            payload = await self._pool._call(
                self._worker, "RESET", (), _env_int("ALFWORLD_ENV_OP_TIMEOUT", 90)
            )
        except BaseException:
            self._dirty = True
            raise
        result, task_description = payload
        self.task_description = task_description
        return result

    async def step(self, action: str):
        try:
            return await self._pool._call(
                self._worker, "STEP", (action,), _env_int("ALFWORLD_ENV_OP_TIMEOUT", 90)
            )
        except BaseException:
            self._dirty = True
            raise

    async def aclose(self) -> None:
        """Release the actor. Safe to call from a cancelled task.

        The actual cleanup runs in a shielded task so a cancellation of the
        enclosing episode cannot abandon a half-used worker: it is either
        returned to the pool after CLOSE or killed and replaced.
        """
        if self._released:
            return
        self._released = True
        cleanup = asyncio.ensure_future(self._pool._release(self._worker, dirty=self._dirty))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # ``cleanup`` keeps running; the worker is still accounted for.
            raise


class _InProcessEnv:
    """Legacy path used when ``ALFWORLD_ENV_ACTORS=0`` (or in-process tests)."""

    def __init__(self, game, max_steps: int) -> None:
        from tasks.alfworld.v1.env_adapter import AlfWorldEnv

        self._env = AlfWorldEnv(game, int(max_steps))
        self.task_description = ""

    async def reset(self):
        result = await asyncio.to_thread(self._env.reset)
        self.task_description = self._env.task_description
        return result

    async def step(self, action: str):
        return await asyncio.to_thread(self._env.step, action)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._env.close)


_POOL: EnvActorPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> EnvActorPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = EnvActorPool(_env_int("ALFWORLD_ENV_WORKERS", 32))
        return _POOL


def shutdown_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown()


atexit.register(shutdown_pool)


async def acquire_env(game, max_steps: int):
    """Create/lease an environment for one episode.

    Returns an object with ``await reset()``, ``await step(action)``,
    ``await aclose()`` and a ``task_description`` attribute, matching the
    synchronous ``AlfWorldEnv`` semantics used by ``run_episode``.
    """
    if not env_actors_enabled():
        return _InProcessEnv(game, max_steps)
    return await get_pool().acquire(game, max_steps)
