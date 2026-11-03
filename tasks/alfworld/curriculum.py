"""Historical task sampling; no reward rejection or extra environment attempts.

GRPO rotates families across task slots, weighting by mixed-group probability.
Single-rollout methods use the same four-family window per batch: the 32 task
positions cycle four families eight times each, so calibration and update
positions share the same family mixture. Their weights estimate
E[(R-b_family)^2], never the degenerate G=1 group score.
"""
import json
import os
import random
from pathlib import Path

FAMILIES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
DISCOUNT = 0.9
EXPLORATION = 0.1


def success_probability(stats):
    wins, trials = stats
    return (wins + 1.0) / (trials + 2.0)


def task_weight(p, method, group_size, baseline=0.5):
    if method in ("grpo", "gspo"):
        if group_size < 2:
            raise ValueError("Grouped curriculum requires G >= 2")
        return 1 - p ** group_size - (1 - p) ** group_size
    return p * (1 - baseline) ** 2 + (1 - p) * baseline ** 2


class Curriculum:
    def __init__(self, directory):
        self.directory = Path(directory) / "curriculum"

    def prior(self, rollout_id):
        # Snapshot k contains outcomes from k. Replayed/rewound batches use only
        # snapshots strictly earlier than their policy version.
        candidates = [(int(p.stem), p) for p in self.directory.glob("*.json")
                      if p.stem.isdigit() and int(p.stem) < rollout_id]
        if not candidates:
            return {"tasks": {}, "families": {}}
        state = json.loads(max(candidates)[1].read_text())
        return {key: state[key] for key in ("tasks", "families")}

    def sample(self, pool, rollout_id, count, seed, method, group_size):
        state = self.prior(rollout_id)
        families = {f: [g for g in pool if g.task_type == f] for f in FAMILIES}
        if any(not games for games in families.values()):
            raise ValueError("Frontier curriculum requires all six training families")
        rng = random.Random(seed + 1_000_003 * rollout_id)
        single = method not in ("grpo", "gspo")
        mixed_families = 4
        games = []
        for position in range(count):
            if single:
                # Four-family mixed sampling: the same family window the grouped
                # sampler uses for its four task slots, cycled over positions
                # (8 positions/family when count=32). Calibration and update
                # positions therefore see the same four-family mixture.
                start = (rollout_id * mixed_families) % len(FAMILIES)
                family = FAMILIES[(start + position % mixed_families) % len(FAMILIES)]
            else:
                phase = rollout_id * count + position
                family = FAMILIES[phase % len(FAMILIES)]
            candidates = families[family]
            baseline = success_probability(state["families"].get(family, [0., 0.]))
            scores = [task_weight(success_probability(state["tasks"].get(g.game_id, [0., 0.])),
                                  method, group_size, baseline) for g in candidates]
            total = sum(scores)
            weights = [(1 - EXPLORATION) * w / total + EXPLORATION / len(scores)
                       for w in scores]
            games.append(rng.choices(candidates, weights=weights, k=1)[0])
        return games, [rng.randrange(2 ** 31 - 1) for _ in games]

    def observe(self, rollout_id, groups):
        state = self.prior(rollout_id)
        for table in state.values():
            for key, (wins, trials) in table.items():
                table[key] = [DISCOUNT * wins, DISCOUNT * trials]
        for game, rewards in groups:
            for table, key in ((state["tasks"], game.game_id),
                               (state["families"], game.task_type)):
                stats = table.setdefault(key, [0., 0.])
                stats[0] += sum(rewards)
                stats[1] += len(rewards)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{rollout_id:07d}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state))
        temp.replace(path)
        # Discard abandoned future snapshots after a checkpoint rewind.
        for future in self.directory.glob("*.json"):
            if future.stem.isdigit() and int(future.stem) > rollout_id:
                future.unlink()


def current_curriculum(args):
    directory = os.environ.get("ALFWORLD_CURRICULUM_DIR") or args.save
    return Curriculum(directory)
