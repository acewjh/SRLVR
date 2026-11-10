"""Probe: can this pod create a Ray placement group with NPU bundles?

Run with the launcher environment (PYTHONPATH from train_scripts/lib/npu_env.sh)
after `ray start --head`. Prints each step so a hang is attributable.
"""

import sys
import time

import torch  # noqa: F401
import torch_npu  # noqa: F401
import ray

from slime.utils.common import is_npu

print("is_npu:", is_npu(), flush=True)
t0 = time.time()
ray.init(address="auto", ignore_reinit_error=True)
print("init ok %.1fs" % (time.time() - t0), flush=True)
print("resources:", ray.cluster_resources(), flush=True)

pg = ray.util.placement_group([{"NPU": 1, "CPU": 1} for _ in range(2)], strategy="PACK")
print("pg created:", pg.id, flush=True)
ray.get(pg.ready(), timeout=120)
print("pg ready %.1fs" % (time.time() - t0), flush=True)


@ray.remote(resources={"NPU": 1})
class Probe:
    def where(self):
        import torch_npu  # noqa: F401

        ctx = ray.get_runtime_context()
        return ctx.get_accelerator_ids(), ray.util.get_node_ip_address()


from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

actors = [
    Probe.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i
        )
    ).remote()
    for i in range(2)
]
print("actors:", ray.get([a.where.remote() for a in actors], timeout=180), flush=True)
print("PROBE_OK", flush=True)
sys.exit(0)
