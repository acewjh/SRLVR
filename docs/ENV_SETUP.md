# Replicating the `cua-rl` environment on another server

Companion doc for `requirements.txt` in this directory. Snapshot taken 2026-09-23
from pod `5acbcf28` (the OPSD baseline node), while the `reward_only_vnext` run
was live.

| File | What it is |
|---|---|
| `requirements.txt` | 235 PyPI-installable pins. **This is the one you `pip install -r`.** |
| `requirements-freeze.txt` | Raw `pip freeze` output, unmodified, kept for provenance. Do **not** install this file — it contains `file:///root/selfgz*` paths that exist on no other machine. |
| `ENV_SETUP.md` | This file. |

`pip install -r requirements.txt` gets you maybe 80% of the way. The remaining
20% is the part that goes wrong on a fresh server, and it is all below.

---

## 1. Create the env

The source env is a plain conda env with **Python 3.10.19** (not 3.11 — the
Ascend 2.1.0 torch stack is 3.10-only):

```bash
conda create -n cua-rl python=3.10 -y     # source env landed on 3.10.19
conda activate cua-rl
```

## 2. Install the PyPI layer

The source env resolves from the ModelArts-internal mirror:

```
http://pip.modelarts.private.com:8888/repository/pypi/simple
```

which is only reachable inside the ModelArts network. On another server use a
public index:

```bash
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple
pip install -r requirements.txt
```

## 3. What pip cannot give you

### 3.1 CANN toolchain (21 packages — must come from the CANN installer)

In the source env these resolve to `file:///usr/local/Ascend/...` and
`file:///root/selfgz*` temporary unpack directories. Those paths **do not exist
on any other machine** — they are created and deleted by the CANN install
process. Do not try to copy them.

`asc_op_compile_base`, `asc_opc_tool`, `auto_tune`, `dataflow`, `es_math`,
`ge-py`, `hccl`, `llm_datadist`, `llm_datadist_v1`, `msobjdump`, `mspti`,
`op_compile_tool`, `op_gen`, `op_test_frame`, `opc_tool`, `schedule_search`,
`show_kernel_debug_data`, `superkernel`, `te`

Install **CANN 8.5.1** from the Huawei image/toolkit on the target server and
they appear on their own. The training script sources it explicitly:

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
```

### 3.2 NPU torch stack (not on public PyPI)

```
torch==2.1.0
torch-npu==2.1.0.post17      # Huawei Ascend index only
torchvision==0.16.0
triton-ascend==3.2.0         # Huawei Ascend index only
```

`torch-npu` and `triton-ascend` have no public PyPI release. Take them from the
Ascend PyPI mirror / the CANN install. These are normally **baked into the
ModelArts NPU base image** — check what the image already has before installing,
and do not let pip overwrite `torch-npu` with a version mismatch against CANN.

### 3.3 ModelArts SDK and image-baked `.pth` paths

`modelarts==1.5.0`, `ma-cli==1.2.3`, `ma_mindstudio_insight`, `ma_tensorboard`
are Huawei-internal wheels with no public release. They are also injected via
`site-packages/modelarts.pth`, which appends these absolute paths to `sys.path`:

```
/workspace/modelarts-dev/modelarts-sdk
/workspace/modelarts-dev/ma-cli
/modelarts/tools/solution/advisor
/modelarts/tools/visualization/ma_tensorboard
/modelarts/tools/visualization/ma_mindstudio_insight
/modelarts/tools/visualization/tensorboard
/workspace/Megatron-LM
/workspace/vllm
/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages
/usr/local/seccomponent/lib
/workspace/infer/model/1
```

**Trap:** `vllm==0.16.0+empty` in `requirements-freeze.txt` is *not* a pip
install. It is a **source checkout at `/workspace/vllm`** that reaches
`sys.path` purely through that `.pth` file; the `vllm.egg-info` sitting in the
checkout is what makes `pip freeze` list it. There is no such distribution on
PyPI, and `pip install vllm==0.16.0+empty` will fail with "no matching
distribution". If the target image does not ship `/workspace/vllm`, copy the
directory — or drop it, since the slime runs here use SGLang, not vLLM.

Note `/workspace/Megatron-LM` is a *second* Megatron copy on `sys.path`,
separate from the one in this repo. Which one wins depends on finder order.

### 3.4 Source / editable installs

These are the parts that actually matter, and they are pinned to git commit
`cb1d10e89ffb00037b909403dedad25584cf1258` of
`https://github.com/94wen94/slime-geo3k-rl-npu.git`:

```bash
R=https://github.com/94wen94/slime-geo3k-rl-npu.git
C=cb1d10e89ffb00037b909403dedad25584cf1258
pip install -e "git+$R@$C#egg=mbridge&subdirectory=frameworks/mbridge"
pip install -e "git+$R@$C#egg=megatron_bridge&subdirectory=frameworks/Megatron-Bridge"
pip install -e "git+$R@$C#egg=megatron_core&subdirectory=frameworks/Megatron-LM"
pip install -e "git+$R@$C#egg=mindspeed&subdirectory=frameworks/MindSpeed"
pip install -e "git+$R@$C#egg=slime&subdirectory=frameworks/slime-ascend"
pip install -e "<repo>/frameworks/sglang/python"     # sglang
```

**Prefer copying the tree over cloning.** A local clone of that repo already
exists at:

```
/workspace/cua/slime-geo3k-rl-npu
```

It has **no git remote configured**, so `git clone` cannot reproduce it — copy
the directory. Better still: this `opsd-slime` repo is self-contained. Its
`frameworks/` directory holds real copies (not symlinks) of mbridge,
Megatron-Bridge, Megatron-LM, MindSpeed, sglang and slime-ascend, marked by a
`.copy_done` sentinel. Copying `opsd-slime` wholesale is the shortest path.

**Trap:** PEP 660 editable installs register a *meta path finder*, which takes
precedence over `PYTHONPATH` (see `__editable__.*.pth` in site-packages, and
`compat/sitecustomize.py` in this repo which exists to manage that). So if you
install the editables from one tree but train from another, you silently import
the wrong code. Either install the editables from the same tree you train from,
or skip the editable installs entirely and let `PYTHONPATH` in the launch script
resolve everything.

## 4. Runtime setup that the launch script does

`train_scripts/run_math_opsd_slime_npu.sh` is the source of truth. The parts
that must be reproduced on any new server:

```bash
conda activate cua-rl 2>/dev/null || conda activate slime-npu   # slime-npu = pre-09-14 name
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NODE_IFNAME=eth0
export GLOO_SOCKET_IFNAME=$NODE_IFNAME
export TP_SOCKET_IFNAME=$NODE_IFNAME
export HCCL_SOCKET_IFNAME=$NODE_IFNAME
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONBUFFERED=16

export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ/frameworks/Megatron-Bridge/src:$PROJ/frameworks/mbridge:$PROJ/frameworks/MindSpeed:$PROJ/frameworks/sglang/python:$PROJ/frameworks/slime-ascend:$PROJ:$PYTHONPATH"
```

`NODE_IFNAME` is the one to check per-server — it is `eth0` here, and a wrong
value makes the HCCL/GLOO rendezvous hang rather than error out.

## 5. Verification

Do not trust "pip install finished". The minimum gate is a real 10-step run on
the full 4+4 layout, not a 1-step smoke test:

```bash
cd /workspace/cua/repos/opsd-slime
SMOKE10=1 bash train_scripts/run_math_opsd_slime_npu.sh
```

Also check the grader canary first — this dependency has silently disappeared
from the env before and the failure mode is a full abort:

```bash
python -c "import latex2sympy2_extended, sympy; print('grader OK')"
```

Expected on a healthy env: `latex2sympy2_extended==1.11.0`, `sympy==1.14.0`,
`numpy==1.26.0`, `transformers==4.51.3`, `datasets==5.0.1`, `ray==2.54.0`,
`peft==0.20.0`.

## 6. Traps that will look like env problems but are not

- **`--qkv-format bshd` is mandatory.** The thd/TND attention path computes
  wrong values (null KL ~6.9 nats) with no error raised. A run using `thd` looks
  like a broken environment; it is not.
- **Negative flags crash at import** on this stack, and `loss_masks` is a
  variable-length list, not a tensor. See `frameworks/slime-ascend` for how the
  existing scripts work around both.
- **Guard thresholds must not be made relative** — the existing absolute values
  are load-bearing.
