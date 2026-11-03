#!/bin/bash
# ============================================================
# One-shot ALFWorld setup: isolated overlay env, dataset, game index, eval config.
#
# Isolation contract (why this is safe to run while other experiments are live):
#   * NOTHING is installed into the cua-rl base environment. ALFWorld's packages
#     go into $BASE/envs/alfworld_overlay, a venv created with
#     --system-site-packages so it can see cua-rl's interpreter but not modify it.
#   * The training interpreter reaches that overlay by APPENDING its
#     site-packages to sys.path (tasks/alfworld/v1/paths.py), so it can only supply
#     modules cua-rl lacks, never shadow one it has. Verified overlap is limited
#     to pip/setuptools.
#   * No NPU, no Ray, no SGLang: this script is pure CPU + filesystem.
#
# Idempotent: re-running skips completed steps.
#
# Usage: bash tasks/alfworld/v1/scripts/setup_alfworld.sh
# ============================================================
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
OVERLAY=$BASE/envs/alfworld_overlay
ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-$BASE/data/alfworld}"
DATA_ROOT="$ALFWORLD_DATA_DIR/json_2.1.1"

# Pinned versions. The tw-pddl release ships self-contained games with no
# traj_data.json, which is why tasks/alfworld/v1/env_adapter.py registers game files
# directly instead of going through AlfredTWEnv (whose collect_game_files
# requires traj_data.json and silently yields ZERO games without it).
ALFWORLD_VERSION="0.4.2"
TEXTWORLD_VERSION="1.7.0"
DATASET_ASSET="https://github.com/alfworld/alfworld/releases/download/${ALFWORLD_VERSION}/json_2.1.3_tw-pddl.zip"

log() { echo "[setup-alfworld] $*"; }
die() { echo "[setup-alfworld] FATAL: $*" >&2; exit 1; }

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || die "cannot activate the cua-rl environment"

# ---------------------------------------------------------------------------
# 1. overlay venv
# ---------------------------------------------------------------------------
if [ ! -x "$OVERLAY/bin/python" ]; then
  log "creating overlay venv at $OVERLAY"
  python -m venv --system-site-packages "$OVERLAY" || die "venv creation failed"
else
  log "overlay venv already present"
fi
OV_PY="$OVERLAY/bin/python"
OV_SITE="$($OV_PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
log "overlay site-packages: $OV_SITE"

# ---------------------------------------------------------------------------
# 2. textworld
#
# textworld's sdist runs thirdparty/setup.sh at build time, which curls the
# Inform7 *compiler* from emshort.com. That download is not reachable here, and
# is not needed: ALFWorld's tw-pddl games are already compiled, and Inform7 is
# only used to compile new ones. So the builder is patched to skip setup.sh.
# ---------------------------------------------------------------------------
if ! $OV_PY -c "import textworld" 2>/dev/null; then
  log "installing textworld==$TEXTWORLD_VERSION (patched build: no Inform7 download)"
  BUILD_DIR="$(mktemp -d)"
  trap 'rm -rf "$BUILD_DIR"' EXIT
  $OV_PY -m pip download --no-deps --no-binary :all: --dest "$BUILD_DIR" \
      "textworld==$TEXTWORLD_VERSION" >/dev/null 2>&1 || die "could not download textworld"
  (
    cd "$BUILD_DIR" || exit 1
    tar xzf "textworld-${TEXTWORLD_VERSION}.tar.gz" || exit 1
    cd "textworld-${TEXTWORLD_VERSION}" || exit 1
    $OV_PY - <<'PY'
import pathlib
p = pathlib.Path("setup.py")
s = p.read_text()
old = "    check_call(['./setup.sh'], shell=True, cwd=os.getcwd())\n"
new = "    print('setup.sh skipped: pre-compiled ALFWorld games do not need Inform7')\n"
if old not in s:
    raise SystemExit("setup.py no longer contains the expected setup.sh call; refusing to patch blindly")
p.write_text(s.replace(old, new))
PY
    $OV_PY -m pip install --no-cache-dir --no-build-isolation . >/dev/null 2>&1 \
      || die "textworld build failed"
  ) || die "textworld install failed"
else
  log "textworld already installed"
fi

# ---------------------------------------------------------------------------
# 3. alfworld + the pddl extra
# ---------------------------------------------------------------------------
if ! $OV_PY -c "import alfworld" 2>/dev/null; then
  log "installing alfworld==$ALFWORLD_VERSION"
  $OV_PY -m pip install --no-cache-dir "fast-downward-textworld" "alfworld==$ALFWORLD_VERSION" \
    || die "alfworld install failed"
else
  log "alfworld already installed"
fi

# ---------------------------------------------------------------------------
# 4. dataset
# ---------------------------------------------------------------------------
mkdir -p "$ALFWORLD_DATA_DIR"
ZIP="$ALFWORLD_DATA_DIR/json_2.1.3_tw-pddl.zip"
if [ ! -d "$DATA_ROOT/valid_seen" ]; then
  if [ ! -s "$ZIP" ]; then
    log "downloading $DATASET_ASSET"
    wget --tries=3 --timeout=180 -O "$ZIP" "$DATASET_ASSET" || die "dataset download failed"
  fi
  log "extracting dataset"
  unzip -q -o "$ZIP" -d "$ALFWORLD_DATA_DIR" || die "dataset extraction failed"
else
  log "dataset already extracted at $DATA_ROOT"
fi

for split in train valid_seen valid_unseen; do
  count=$(find "$DATA_ROOT/$split" -name "game.tw-pddl" 2>/dev/null | wc -l)
  [ "$count" -gt 0 ] || die "split '$split' has no games under $DATA_ROOT"
  log "split $split: $count games"
done

# ---------------------------------------------------------------------------
# 5. game index + prompt datasets + eval config
# ---------------------------------------------------------------------------
log "building game index and prompt datasets"
ALFWORLD_HPI_DATA="$DATA_ROOT" \
  "$OV_PY" -m tasks.alfworld.v1.data_prepare \
    --out-dir "$PROJ/tasks/alfworld/v1/configs/data" \
    --index "$PROJ/tasks/alfworld/v1/configs/game_index.json" \
  || die "data preparation failed"

log "done. Next: bash tasks/alfworld/v1/scripts/smoke_alfworld_cpu.sh"
