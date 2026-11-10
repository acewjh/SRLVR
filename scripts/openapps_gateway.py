#!/usr/bin/env python3
"""OpenApps gateway: speaks slime's OSWorld /v1 protocol, backed by browsergym.

The GUI-RL rollout in ``slime/osworld_rollout.py`` is environment-agnostic: it
binds an env, resets it with a task payload, sends pyautogui action scripts and
reads back ``{screenshot, reward, done}``. This server implements that contract
on top of a local OpenApps deployment, so the existing GRPO stack trains on
OpenApps tasks without touching the rollout code.

Protocol (same as the ScaleCUA k8s gateway):
    POST   /v1/bind?type=&worker_id=              -> {"env_id": ...}
    POST   /v1/proxy/reset?env_id=                 body {task_config, options...}
    POST   /v1/proxy/step?env_id=                  body {action: <pyautogui src>}
    GET    /v1/proxy/observation?env_id=&idx_key=  -> {"observation": {"screenshot": b64}}
    POST   /v1/proxy/evaluate?env_id=              -> {"result": <score>}
    POST   /v1/keepalive?env_id=                   -> {}
    DELETE /v1/delete?env_id=                      -> {}

Threading model: browsergym keeps ONE global Playwright instance per process
and the sync API is bound to the thread that created it, so every environment
call runs on a single dedicated service thread. Episodes therefore serialise on
screen capture/actions (milliseconds each); model generation, which dominates
rollout latency, stays concurrent.

Run:
    OPENAPPS_BASE_URL=http://127.0.0.1:5058 python scripts/openapps_gateway.py --port 5111
"""

from __future__ import annotations

import argparse
import ast
import base64
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

BASE_URL = os.environ.get("OPENAPPS_BASE_URL", "http://127.0.0.1:5058")
# Episodes mutate the app databases, so a single server shared by concurrent
# rollouts leaks state between them ("remove X" scores 1 once X is already
# gone). Run one seeded server per concurrent env and reset that server
# between episodes; the launcher exports the comma-separated list.
POOL_BASE_URLS = [
    u.strip().rstrip("/")
    for u in os.environ.get("OPENAPPS_POOL_BASE_URLS", BASE_URL).split(",")
    if u.strip()
]
SCREEN_SIZE = (1920, 1080)
PAGE_TIMEOUT_MS = int(os.environ.get("OPENAPPS_PAGE_TIMEOUT_MS", "30000"))


def _reset_app_state(base_url: str) -> None:
    """Restore an app server's databases to the configured seed state.

    A slot's server is reused across episodes. Without a reset a task that
    mutates state (delete an event, add a todo) leaks into the next episode,
    and "remove X" tasks score 1 before the agent does anything because the
    initial state already lacks X. The route lives in open_apps' start_page.
    """
    import requests

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(f"{base_url}/__reset_apps", timeout=180)
            resp.raise_for_status()
            return
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"app state reset failed for {base_url}: {last_exc}")


def _patch_task_timeout() -> None:
    """OpenAppsTask hardcodes timeout=5000 ms; a 1080p screenshot of the app
    dashboard routinely exceeds that when more than one browser is alive.
    Raise it before any task is registered (the value is set in __init__)."""
    import open_apps.tasks.add_tasks_to_browsergym as oat

    if getattr(oat.OpenAppsTask.__init__, "_openapps_timeout_patched", False):
        return
    original = oat.OpenAppsTask.__init__

    def init(self, *args, **kwargs):  # noqa: ANN001
        original(self, *args, **kwargs)
        self.timeout = PAGE_TIMEOUT_MS

    init._openapps_timeout_patched = True  # type: ignore[attr-defined]
    oat.OpenAppsTask.__init__ = init


def _patch_task_setup() -> None:
    """The stock setup() waits for the ``load`` event, which never fires on the
    OpenApps start page (some embedded resource keeps the page busy), so every
    reset died after the navigation timeout. Wait for DOMContentLoaded and give
    the htmx handlers a moment instead; the task's initial state is fetched
    over HTTP, not from the page, so nothing downstream depends on ``load``."""
    import open_apps.tasks.add_tasks_to_browsergym as oat
    from open_apps.state import get_current_state

    if getattr(oat.OpenAppsTask.setup, "_openapps_fast_setup", False):
        return

    def setup(self, page):  # noqa: ANN001
        # The apps pull a webfont from fonts.googleapis.com. This pod has no
        # direct internet, so that request hangs forever and the document never
        # leaves readyState=loading (the reset timeout). Abort every non-local
        # request: the UI keeps working with fallback fonts, which is also the
        # deterministic behaviour we want for RL.
        def _local_only(route):  # noqa: ANN001
            url = route.request.url
            if url.startswith(("http://127.0.0.1", "http://localhost", "data:", "blob:")):
                route.continue_()
            else:
                route.abort()

        page.route("**/*", _local_only)
        self.page = page
        page.goto(self.url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        page.wait_for_timeout(500)
        self.initial_state = get_current_state(self.url)
        return self._get_goal(), self._get_info()

    setup._openapps_fast_setup = True  # type: ignore[attr-defined]
    oat.OpenAppsTask.setup = setup

# pyautogui names -> Playwright key names
_KEY_MAP = {
    "enter": "Enter", "return": "Enter", "tab": "Tab", "esc": "Escape",
    "escape": "Escape", "backspace": "Backspace", "delete": "Delete",
    "space": "Space", "up": "ArrowUp", "down": "ArrowDown",
    "left": "ArrowLeft", "right": "ArrowRight", "pageup": "PageUp",
    "pagedown": "PageDown", "home": "Home", "end": "End", "insert": "Insert",
    "ctrl": "Control", "control": "Control", "shift": "Shift",
    "alt": "Alt", "option": "Alt", "cmd": "Meta", "command": "Meta",
    "win": "Meta", "super": "Meta",
}

_CALL = re.compile(r"^pyautogui\.(\w+)\((.*)\)$")


def _key(name: str) -> str:
    return _KEY_MAP.get(name.lower(), name)


class EnvState:
    def __init__(self, slot: int) -> None:
        self.env = None
        self.task = None
        self.slot = slot
        self.base_url = POOL_BASE_URLS[slot % len(POOL_BASE_URLS)]
        self.exp_dir = tempfile.mkdtemp(prefix="openapps_env_")
        self.mouse = [SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2]


class EnvService(threading.Thread):
    """Owns every BrowserEnv; browsergym's Playwright singleton lives here."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="openapps-env-service")
        self._q: queue.Queue = queue.Queue()
        self._envs: dict[str, EnvState] = {}
        self._registered: set[str] = set()
        self._slot_owner: dict[int, str] = {}

    # ---- submission ------------------------------------------------------
    def run(self) -> None:
        while True:
            fn, args, result_q = self._q.get()
            if fn is None:
                break
            try:
                result_q.put((True, fn(*args)))
            except Exception as exc:  # noqa: BLE001 - surfaced to the HTTP caller
                result_q.put((False, exc))

    def submit(self, fn: Callable[..., Any], *args: Any, timeout: float = 300.0) -> Any:
        result_q: queue.Queue = queue.Queue()
        self._q.put((fn, args, result_q))
        try:
            ok, value = result_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"env service timed out after {timeout}s") from exc
        if not ok:
            raise value
        return value

    # ---- env operations (all on the service thread) ----------------------
    def new_env(self, env_id: str) -> None:
        # One seeded server per live env. Prefer a slot nobody owns; if every
        # slot is busy (more concurrent envs than servers), fall back to
        # round-robin and warn -- two envs on one server will interfere.
        free = [i for i in range(len(POOL_BASE_URLS)) if i not in self._slot_owner]
        if free:
            slot = free[0]
        else:
            slot = len(self._envs) % len(POOL_BASE_URLS)
            print(f"[gateway] WARNING: {len(self._envs)} envs > "
                  f"{len(POOL_BASE_URLS)} app servers; slot {slot} is shared")
        self._slot_owner[slot] = env_id
        self._envs[env_id] = EnvState(slot)

    def drop_env(self, env_id: str) -> None:
        state = self._envs.pop(env_id, None)
        if state is not None:
            self._slot_owner.pop(state.slot, None)
        if state is not None and state.env is not None:
            try:
                state.env.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass

    def _state(self, env_id: str) -> EnvState:
        state = self._envs.get(env_id)
        if state is None:
            raise KeyError(f"unknown env_id {env_id}")
        return state

    def reset(self, env_id: str, task_payload: dict) -> dict:
        from browsergym.core import env as bg_env

        from open_apps.agent.env_args import DeviceEnvArgs
        from open_apps.tasks import load_task
        from open_apps.tasks.add_tasks_to_browsergym import register_tasks_with_browsergym

        task_key = task_payload.get("task_key") or task_payload.get("task_id")
        if not task_key:
            raise ValueError("reset payload has no task_key")
        task = load_task(task_key)  # cheap: YAML config -> dataclass
        if task.task_id not in self._registered:
            register_tasks_with_browsergym([task])
            self._registered.add(task.task_id)

        state = self._state(env_id)
        if state.env is not None:  # reuse the slot for a new episode
            try:
                state.env.close()
            except Exception:  # noqa: BLE001
                pass
            state.env = None

        _reset_app_state(state.base_url)

        env_args = DeviceEnvArgs(
            task_name=task.task_id,
            task_seed=42,
            max_steps=int(os.environ.get("OPENAPPS_MAX_STEPS", "30")),
            headless=True,
            task_kwargs={"base_url": state.base_url, "screen_resolution": SCREEN_SIZE},
        )
        state.env = env_args.make_env(
            action_mapping=lambda action: action,  # env.step is never used
            exp_dir=Path(state.exp_dir),
        )
        obs, info = state.env.reset()
        state.env.unwrapped.context.set_default_timeout(PAGE_TIMEOUT_MS)
        state.task = state.env.unwrapped.task
        state.mouse = [SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2]
        return {"status": "success", "observation": {"screenshot": self._screenshot(state)}, "info": info or {}}

    def _screenshot(self, state: EnvState) -> str:
        page = state.env.unwrapped.page
        errors = []
        for attempt in range(2):
            try:
                png = page.screenshot(type="png", timeout=60000)
                return base64.b64encode(png).decode("ascii")
            except Exception as exc:  # noqa: BLE001 - one retry, then bubble
                errors.append(exc)
                page.wait_for_timeout(500)
        raise errors[-1]

    def observation(self, env_id: str) -> dict:
        state = self._state(env_id)
        return {"observation": {"screenshot": self._screenshot(state)}}

    def step(self, env_id: str, action: str) -> dict:
        state = self._state(env_id)
        if state.env is None:
            raise RuntimeError("env is not reset")
        page = state.env.unwrapped.page
        error = None
        if action not in ("DONE", "FAIL"):
            try:
                self._execute(state, page, action)
            except Exception as exc:  # noqa: BLE001 - reported as tool feedback
                error = str(exc)
        reward = 0.0
        try:
            reward = float(state.task.reward())
        except Exception as exc:  # noqa: BLE001 - grading must not kill the episode
            error = error or f"reward failed: {exc}"
        return {
            "observation": {"screenshot": self._screenshot(state)},
            "reward": reward,
            "done": bool(reward >= 1.0),
            "info": {"error": error} if error else {},
        }

    def evaluate(self, env_id: str) -> dict:
        state = self._state(env_id)
        try:
            return {"result": float(state.task.reward())}
        except Exception as exc:  # noqa: BLE001
            return {"result": 0.0, "error": str(exc)}

    # ---- pyautogui script executor --------------------------------------
    @staticmethod
    def _parse_args(argstr: str) -> list[Any]:
        if not argstr.strip():
            return []
        call = ast.parse(f"action({argstr})", mode="eval").body
        if any(keyword.arg != "duration" for keyword in call.keywords):
            raise ValueError("unsupported pyautogui keyword argument")
        for keyword in call.keywords:
            ast.literal_eval(keyword.value)
        values = [ast.literal_eval(arg) for arg in call.args]
        if any(not isinstance(value, (str, int, float)) for value in values):
            raise ValueError("pyautogui arguments must be strings or numbers")
        return values

    def _execute(self, state: EnvState, page, script: str) -> None:
        for raw in script.splitlines():
            line = raw.strip()
            if not line or line in ("DONE", "FAIL"):
                continue
            m = _CALL.match(line)
            if not m:
                raise ValueError(f"unsupported command: {line!r}")
            self._dispatch(state, page, m.group(1), self._parse_args(m.group(2)))

    def _dispatch(self, state: EnvState, page, name: str, args: list[Any]) -> None:
        mouse, kb = page.mouse, page.keyboard
        x = y = None
        if name in ("click", "rightClick", "middleClick", "doubleClick", "tripleClick",
                    "moveTo", "dragTo"):
            x = int(args[0]) if args else None
            y = int(args[1]) if len(args) > 1 else None
        if name in ("click", "rightClick", "middleClick", "doubleClick", "tripleClick"):
            button = {"rightClick": "right", "middleClick": "middle"}.get(name, "left")
            clicks = {"doubleClick": 2, "tripleClick": 3}.get(name, 1)
            if x is not None and y is not None:
                state.mouse = [x, y]
                mouse.move(x, y)
            mouse.click(state.mouse[0], state.mouse[1], button=button, click_count=clicks)
        elif name == "moveTo":
            state.mouse = [x, y]
            mouse.move(x, y)
        elif name == "mouseDown":
            mouse.down()
        elif name == "mouseUp":
            mouse.up()
        elif name == "dragTo":
            mouse.move(state.mouse[0], state.mouse[1])
            mouse.down()
            mouse.move(x, y)
            mouse.up()
            state.mouse = [x, y]
        elif name == "press":
            key = str(args[0]).lower()
            if len(key) == 1:
                kb.type(str(args[0]))
            else:
                kb.press(_key(key))
        elif name == "keyDown":
            kb.down(_key(str(args[0])))
        elif name == "keyUp":
            kb.up(_key(str(args[0])))
        elif name in ("scroll", "hscroll"):
            amount = float(args[0]) if args else 0.0
            if name == "scroll":
                mouse.wheel(0, -amount * 100)
            else:
                mouse.wheel(amount * 100, 0)
        elif name == "sleep":
            page.wait_for_timeout(int(float(args[0]) * 1000))
        elif name == "position":
            return
        else:
            raise ValueError(f"unsupported pyautogui command: {name}")


_patch_task_timeout()
_patch_task_setup()
SERVICE = EnvService()
SERVICE.start()
APP = FastAPI(title="openapps-gateway")


class ResetBody(BaseModel):
    task_config: dict = {}
    seed: Any = None
    options: dict = {}
    idx_key: Any = None


class StepBody(BaseModel):
    action: str = ""
    pause: float = 0
    task_type: str = "desktop"
    idx_key: Any = None
    action_space: str = "pyautogui"


def _get(env_id: str) -> None:
    try:
        SERVICE.submit(SERVICE._state, env_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown env_id {env_id}") from exc


@APP.get("/health")
def health() -> dict:
    return {"status": "ok", "base_url": BASE_URL}


@APP.post("/v1/bind")
def bind(type: str = "", worker_id: str = "") -> dict:  # noqa: A002 - protocol name
    env_id = uuid.uuid4().hex
    SERVICE.submit(SERVICE.new_env, env_id)
    return {"env_id": env_id}


@APP.post("/v1/proxy/reset")
def reset(body: ResetBody, env_id: str = Query(...)) -> dict:
    _get(env_id)
    return SERVICE.submit(SERVICE.reset, env_id, body.task_config)


@APP.post("/v1/proxy/step")
def step(body: StepBody, env_id: str = Query(...)) -> dict:
    _get(env_id)
    return SERVICE.submit(SERVICE.step, env_id, body.action)


@APP.get("/v1/proxy/observation")
def observation(env_id: str = Query(...), idx_key: Any = None) -> dict:
    _get(env_id)
    return SERVICE.submit(SERVICE.observation, env_id)


@APP.post("/v1/proxy/evaluate")
def evaluate(env_id: str = Query(...)) -> dict:
    _get(env_id)
    return SERVICE.submit(SERVICE.evaluate, env_id)


@APP.post("/v1/keepalive")
def keepalive(env_id: str = Query(...)) -> dict:
    _get(env_id)
    return {"status": "ok"}


@APP.delete("/v1/delete")
def delete(env_id: str = Query(...)) -> dict:
    SERVICE.submit(SERVICE.drop_env, env_id)
    return {"status": "ok"}


def main() -> None:
    import logging

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5111)
    args = ap.parse_args()
    logging.info("OpenApps gateway on %s:%d -> %s", args.host, args.port, BASE_URL)
    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
