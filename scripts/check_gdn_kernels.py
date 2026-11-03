"""Import-only GDN preflight; run in the intended training runtime."""
import importlib


def check():
    package = importlib.import_module("sgl_kernel_npu")
    if hasattr(package, "SHIM_BANNER"):
        raise RuntimeError(f"GDN requires real kernels, found stub: {package.__file__}")
    required = {
        "mamba.causal_conv1d": ("causal_conv1d_fn_npu", "causal_conv1d_update_npu"),
        "fla.fused_gdn_gating": ("fused_gdn_gating_npu",),
        "fla.chunk": ("chunk_gated_delta_rule_npu",),
        "fla.fused_sigmoid_gating_recurrent": ("fused_sigmoid_gating_delta_rule_update_npu",),
    }
    for suffix, symbols in required.items():
        module = importlib.import_module(f"sgl_kernel_npu.{suffix}")
        for name in symbols:
            if not callable(getattr(module, name, None)):
                raise RuntimeError(f"Missing GDN kernel {module.__name__}.{name}")
        print(f"[gdn-kernels] {module.__file__}", flush=True)


if __name__ == "__main__":
    check()
