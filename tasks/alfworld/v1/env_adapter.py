"""ALFWorld TextWorld environment adapter (text-only; no THOR, no vision).

Why this bypasses ``alfworld.agents.environment.AlfredTWEnv``
------------------------------------------------------------
``AlfredTWEnv.collect_game_files`` requires a ``traj_data.json`` next to every
``game.tw-pddl`` and silently yields **zero** games when it is absent -- which is
exactly the case for the official ``json_2.1.3_tw-pddl`` release, whose games are
self-contained. Rather than half-install a second dataset release, this adapter
registers the ``game.tw-pddl`` files directly with ``textworld.gym``, reusing
ALFWorld's own ``AlfredDemangler`` / ``AlfredInfos`` wrappers so entity names and
the ``extra.gamefile`` extra behave exactly as in the reference implementation.

Registering per game (rather than one batch over the whole split) is what makes
episodes addressable: the training set needs to name a specific game for a given
prompt, and evaluation needs a stable game order per split.

Everything here is CPU-only and holds no torch/npu state, so it is safe to
exercise while the NPUs are occupied by another experiment.
"""

import glob
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .paths import ensure_alfworld_importable, game_data_root

# ---------------------------------------------------------------------------
# TextWorld is single-threaded. This lock is the whole reason it is safe to build
# several episodes concurrently.
#
# `textworld/envs/pddl/logic/__init__.py` builds a MODULE-LEVEL tatsu parser
# (`_PARSER = PddlLogicParser(...)`) and `_parse_and_convert` calls
# `_PARSER.parse(...)` directly. A tatsu parser carries mutable parse state
# (context, memo table, `_furthest_exception`), so two threads parsing at once
# corrupt each other and raise
# `tatsu.exceptions.FailedToken: expecting 'action' / 'template' / '{'` at
# positions that move run to run -- the signature of a race, not of bad data.
#
# Measured on the 12 smoke games with the same interpreter:
#     sequential load        -> 12/12 OK
#     12 threads (asyncio.to_thread, the rollout's path) -> 11-12/12 FAIL
# A single-threaded probe can never see this, which is why the CPU suite passed.
#
# The lock covers construction, reset, step and close because a game's grammar is
# parsed on load and commands are parsed on every step.
#
# Env-actor processes (tasks/alfworld_env_worker.py) each own exactly one
# environment and execute one operation at a time, so there is no concurrent
# parser access to serialise and the module lock becomes a no-op there. Keeping
# it enabled for the legacy in-process path (several asyncio.to_thread workers
# sharing this module) preserves the original safety.
class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_TEXTWORLD_LOCK = (
    _NullLock()
    if os.environ.get("ALFWORLD_ENV_ACTOR_PROCESS") == "1"
    else threading.RLock()
)

__all__ = [
    "GameRef",
    "StepResult",
    "AlfWorldEnv",
    "discover_games",
    "extract_task_description",
    "task_type_from_game_id",
    "DEFAULT_MAX_STEPS",
]

#: Hard cap on environment steps per episode. ALFWorld tasks are solvable well
#: within this; the reference ReAct recipes use 30-50.
DEFAULT_MAX_STEPS = 30

_TASK_DESC_RE = re.compile(r"Your task is to:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

#: Known ALFWorld task_type prefixes, longest first so
#: ``pick_clean_then_place_in_recep`` is not shadowed by ``pick_and_place_simple``.
_KNOWN_TASK_TYPES = (
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_and_place_simple",
    "look_at_obj_in_light",
)


@dataclass(frozen=True)
class GameRef:
    """A single ALFWorld game file, addressable and reproducible."""

    game_id: str
    path: str
    task_type: str
    split: str

    def to_dict(self) -> dict:
        return {"game_id": self.game_id, "path": self.path, "task_type": self.task_type, "split": self.split}


@dataclass
class StepResult:
    """One environment interaction."""

    observation: str
    admissible_actions: list[str]
    done: bool
    won: bool
    score: float


def task_type_from_game_id(game_id: str) -> str:
    """Task type from a game directory name.

    Game dirs are named ``<task_type>-<obj>-<parent>-<recep>-<id>``. Falls back to
    the leading component when the prefix is not one of the six known types, so
    an unfamiliar dataset still produces a usable (if unlabelled) category.
    """
    for task_type in _KNOWN_TASK_TYPES:
        if game_id.startswith(task_type):
            return task_type
    return game_id.split("-", 1)[0]


def discover_games(split: str, *, limit: int | None = None, root: str | None = None) -> list[GameRef]:
    """All ``game.tw-pddl`` files in ``split``, sorted for reproducibility.

    ``limit`` truncates AFTER sorting, so a truncated index is a deterministic
    prefix of the full one -- re-running with a larger limit only appends.
    """
    root = root or game_data_root()
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"ALFWorld split {split!r} not found under {root!r}. "
            "Run tasks/alfworld/v1/scripts/setup_alfworld.sh to download and extract the dataset."
        )

    pattern = os.path.join(split_dir, "*", "*", "game.tw-pddl")
    games: list[GameRef] = []
    for path in sorted(glob.glob(pattern)):
        # <split>/<task-dir>/<trial-dir>/game.tw-pddl -> task-dir is the game id
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        game_id = f"{split}/{task_dir}/{os.path.basename(os.path.dirname(path))}"
        games.append(
            GameRef(
                game_id=game_id,
                path=path,
                task_type=task_type_from_game_id(task_dir),
                split=split,
            )
        )
    if limit is not None:
        games = games[:limit]
    return games


def write_game_index(splits, out_path: str, *, limit: int | None = None) -> dict:
    """Persist the discovered game list so rollouts never re-walk the tree."""
    index = {split: [g.to_dict() for g in discover_games(split, limit=limit)] for split in splits}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    return {split: len(games) for split, games in index.items()}


def load_game_index(path: str) -> dict[str, list[GameRef]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {split: [GameRef(**g) for g in games] for split, games in payload.items()}


def extract_task_description(observation: str) -> str:
    """Pull ``Your task is to: ...`` out of ALFWorld's initial observation.

    The ``tw-pddl`` games encode the goal as a PDDL form, so the natural-language
    objective is only available from the rendered observation. Returns ``""`` when
    the line is absent rather than inventing one.
    """
    match = _TASK_DESC_RE.search(observation or "")
    return match.group(1).strip() if match else ""


class AlfWorldEnv:
    """A single-game ALFWorld TextWorld environment.

    Usage::

        env = AlfWorldEnv(game, max_steps=30)
        result = env.reset()
        result = env.step("go to shelf 1")
        env.close()
    """

    def __init__(self, game: GameRef, max_steps: int = DEFAULT_MAX_STEPS):
        ensure_alfworld_importable()

        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.game = game
        self.max_steps = max_steps

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        with _TEXTWORLD_LOCK:
            self._env_id = textworld.gym.register_games(
                [game.path],
                request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
            )
            self._env = textworld.gym.make(self._env_id)
        self.task_description = ""

    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """Start the episode and read the objective from the opening observation."""
        with _TEXTWORLD_LOCK:
            observation, info = self._unwrap_reset(self._env.reset())
        self.task_description = extract_task_description(observation)
        return StepResult(
            observation=observation,
            admissible_actions=self._admissible(info),
            done=False,
            won=False,
            score=0.0,
        )

    @staticmethod
    def _info(info: dict, key: str, default):
        """Read one key out of textworld's *batched* info dict.

        ``TextworldBatchGymEnv`` returns ``{"won": [False], "admissible_commands":
        [[...]]}`` -- a dict whose values are batch-of-one lists. Reading it as a
        plain dict makes ``bool([False])`` truthy, which silently reports every
        step as won; unwrap explicitly.
        """
        if not isinstance(info, dict) or key not in info:
            return default
        return _first(info[key])

    def step(self, action: str) -> StepResult:
        """Execute one command; unknown commands are still forwarded to the env.

        Forwarding is deliberate: ALFWorld answers an inadmissible command with
        "That is not a valid action", which is itself useful feedback and keeps
        the environment's step counter authoritative.
        """
        with _TEXTWORLD_LOCK:
            raw = self._env.step([action])
        observation, _reward, done, info = self._unwrap_step(raw)
        return StepResult(
            observation=observation,
            admissible_actions=self._admissible(info),
            done=bool(done),
            won=bool(self._info(info, "won", False)),
            score=float(self._info(info, "score", 0.0) or 0.0),
        )

    def close(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None and hasattr(env, "close"):
            try:
                with _TEXTWORLD_LOCK:
                    env.close()
            except Exception:  # noqa: BLE001 - teardown must never mask a rollout error
                pass

    def __enter__(self) -> "AlfWorldEnv":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Textworld return-shape normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _admissible(info: dict) -> list[str]:
        commands = info.get("admissible_commands") or []
        # batch_size=1 -> either a flat list of commands or a [1, n] nested list.
        if commands and isinstance(commands[0], (list, tuple)):
            commands = commands[0]
        return [str(c) for c in commands]

    @staticmethod
    def _unwrap_reset(raw) -> tuple[str, dict]:
        """Normalise ``reset`` across textworld gym API generations.

        Old API: ``(observation, infos)``.  Newer API: ``(obs, info)`` or a bare
        observation with ``env.infos`` populated. All shapes are reduced to
        ``(single_observation, single_info_dict)`` for batch_size=1.
        """
        if isinstance(raw, tuple):
            obs, info = raw[0], raw[1]
        else:
            obs, info = raw, None
        return _first(obs), _first(info if info is not None else {})

    @staticmethod
    def _unwrap_step(raw) -> tuple[str, float, bool, dict]:
        """Normalise ``step`` across textworld gym API generations."""
        if isinstance(raw, tuple):
            if len(raw) == 5:  # new gym: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = raw
                return _first(obs), float(_first(reward) or 0.0), bool(_first(terminated) or _first(truncated)), _first(info)
            if len(raw) == 4:  # old gym: obs, reward, done, info
                obs, reward, done, info = raw
                return _first(obs), float(_first(reward) or 0.0), bool(_first(done)), _first(info)
        raise TypeError(f"unexpected textworld step return shape: {type(raw).__name__}")


def _first(value):
    """Unwrap a batch-of-one container; pass scalars/dicts through unchanged."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value
