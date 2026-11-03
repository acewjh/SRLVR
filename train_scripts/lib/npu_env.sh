# shellcheck shell=bash
# Shared conda / CANN / NPU / PYTHONPATH scaffold for every opsd-slime launcher.
#
# R9: this block was copied verbatim into six launchers
# (train_scripts/run_math_*, scripts/run_grpo*.sh, train_scripts/run_dev_eval.sh,
# tasks.alfworld/scripts/_common_v2.sh, tasks.code/scripts/run_code_*.sh). The
# copies had already drifted in shape (GLOO_* on one line vs three; the alloc
# conf from a model-card variable vs a hardcoded default), which is exactly the
# drift this extraction removes.
#
# Callers must have `PROJ` set (the repo root) before sourcing.
#
# Usage:
#     source "$PROJ/train_scripts/lib/npu_env.sh"
#     opsd_enter_env                              # conda + CANN
#     opsd_export_npu_env "$MODEL_ALLOC_CONF"     # "" => leave alloc conf alone

# conda + CANN. `cua-rl` is this machine's slime env since the 09-14 container
# rebuild; `slime-npu` is the pre-rebuild name kept as a fallback.
# OPSD_CONDA_ENV names the env explicitly, for hosts where neither of those
# exists (e.g. the four-node job, whose complete stack is `PyTorch-2.9.0`:
# py3.11 / torch 2.9.0 / sglang 0.5.5 / ray 2.54). It is tried first, and a
# failed activate falls through to the historical names instead of aborting.
opsd_enter_env() {
  source /opt/conda/etc/profile.d/conda.sh
  local want="${OPSD_CONDA_ENV:-}"
  if [ -n "$want" ]; then
    conda activate "$want" || echo "[npu-env] FATAL: conda env '$want' not found" >&2
  else
    conda activate cua-rl 2>/dev/null || conda activate slime-npu 2>/dev/null ||       echo "[npu-env] no cua-rl/slime-npu conda env; relying on OPSD_VENV" >&2
  fi
  # gy-rl-4 adaptation. This host has no conda env that carries sglang (only
  # PyTorch-2.7.1, which lacks it); the validated interpreter here is a VENV
  # built on top of conda's python-3.11.10, so `conda activate` cannot reach it.
  # Layer it on afterwards. Unset on the colleague's hosts => no behaviour change.
  if [ -n "${OPSD_VENV:-}" ]; then
    [ -f "$OPSD_VENV/bin/activate" ] || {
      echo "[npu-env] FATAL: OPSD_VENV=$OPSD_VENV has no bin/activate" >&2; return 1; }
    source "$OPSD_VENV/bin/activate"
    echo "[npu-env] venv active: $OPSD_VENV"
  fi
  # Vendor environment scripts read optional unset variables.
  local restore_nounset=0
  [[ $- == *u* ]] && restore_nounset=1
  set +u
  # gy-rl-4 adaptation. CANN 8.5.1 does not exist on this host. Take the first
  # toolkit that does, preferring whatever the caller pins, then the private
  # 8.5.2 this venv's torch_npu 2.8.0.post2 was built against, then the system
  # tree. Announcing the choice matters: picking a different CANN silently is
  # exactly the class of failure the Qwen3.5-9B campaign lost two days to.
  local _cann_env
  for _cann_env in "${OPSD_CANN_SET_ENV:-}"                    /usr/local/Ascend/cann-8.5.1/set_env.sh                    /cache/hp-rlvr/cann-private/cann-8.5.2/set_env.sh                    /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if [ -n "$_cann_env" ] && [ -f "$_cann_env" ]; then
      source "$_cann_env"
      echo "[npu-env] CANN: $_cann_env"
      break
    fi
  done
  source /usr/local/Ascend/nnal/atb/set_env.sh
  [ "$restore_nounset" = 1 ] && set -u
  return 0
}

# The validated cluster recipe.
#   $1 = repo root (the launchers spell it PROJ or P; pass it explicitly rather
#        than guessing which name is in scope)
#   $2 = PYTORCH_NPU_ALLOC_CONF default: the model card's value,
#        "expandable_segments:True", or "" to leave it unset. It must be exported
#        BEFORE `ray start` so the actors inherit it.
opsd_remove_gdn_kernel_shim() {
  local root="${1:?repo root required}" part filtered=""
  local -a kernel_paths=()
  [ "${OPSD_QWEN35_GDN:-0}" = "1" ] || return 0
  IFS=: read -r -a kernel_paths <<< "${PYTHONPATH:-}"
  for part in "${kernel_paths[@]+${kernel_paths[@]}}"; do
    [ "${part%/}" = "$root/compat/sgl_kernel_npu_shim" ] && continue
    [ -z "$part" ] && continue
    filtered="${filtered:+$filtered:}$part"
  done
  export PYTHONPATH="$filtered"
}

opsd_export_npu_env() {
  local root="${1:?opsd_export_npu_env: repo root required}"
  local alloc_default="${2:-}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NODE_IFNAME="${NODE_IFNAME:-eth0}"
  export GLOO_SOCKET_IFNAME="$NODE_IFNAME"
  export TP_SOCKET_IFNAME="$NODE_IFNAME"
  export HCCL_SOCKET_IFNAME="$NODE_IFNAME"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export PYTHONBUFFERED=16

  if [ -n "$alloc_default" ]; then
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-$alloc_default}"
  fi

  # PYTHONPATH must resolve THIS project's frameworks + the torch-2.1 compat shim.
  #
  # One deliberate difference from the copied blocks this replaces: they ended
  # with `:$PYTHONPATH` unconditionally, which appends an EMPTY element (i.e. the
  # current directory) to sys.path whenever PYTHONPATH was unset. `${PYTHONPATH:+...}`
  # omits it. Behaviourally inert here — the launchers run from the repo root,
  # which is already on the path — but it removes a stray CWD import surface.
  export PYTHONPATH="$root/compat"\
":$root/frameworks/Megatron-LM"\
":$root/frameworks/Megatron-Bridge/src"\
":$root/frameworks/mbridge"\
":$root/frameworks/MindSpeed"\
":$root/frameworks/sglang/python"\
":$root/frameworks/slime-ascend"\
":$root${PYTHONPATH:+:$PYTHONPATH}"

  # GDN calls fused kernels on every prefill. Also remove a shim inherited
  # from a previous dense-model launch, rather than merely not adding it.
  opsd_remove_gdn_kernel_shim "$root"

  export WANDB_MODE=disabled WANDB_DISABLED=true
}
