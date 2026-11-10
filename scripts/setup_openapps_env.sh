#!/bin/bash
# Install the OpenApps runtime into the PyTorch-2.9.0 conda env.
#
# The per-pod conda env is NOT on the shared filesystem, so this must run on
# every worker that needs OpenApps (worker-0 and worker-1).
#
# Deliberate deviation from `pip install -e ./openapps`: OpenApps pins
# transformers==4.52.2, but this node's GRPO stack needs 4.57.6. We install the
# editable package with --no-deps and then install the runtime dependencies
# explicitly (also --no-deps, so pip cannot backtrack onto numpy 1.19 /
# datasets 1.8 and silently downgrade the training env).
set -uo pipefail

ROOT="/workspace/cua/repos/opsd-slime"
AGENTLAB="/workspace/cua/repos/AgentLab"
CA="/workspace/cua/ca_bundle.pem"

source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.9.0 || { echo "FATAL: PyTorch-2.9.0 missing"; exit 1; }

echo "=== editable open_apps (--no-deps: keeps transformers 4.57.6) ==="
pip install --no-cache-dir --no-deps -e "$ROOT/openapps" || exit 1

echo "=== runtime dependencies (--no-deps) ==="
pip install --no-cache-dir --no-deps \
  "$AGENTLAB" \
  "browsergym>=0.13.3" "browsergym-core" "browsergym-experiments" \
  "playwright>=1.44" "python-fasthtml>=0.12.14" "fastlite" "fastcore" "apswutils==0.1.2" "apsw" \
  "feedgen>=1.0.0" "submitit>=1.5.2" "hydra-submitit-launcher>=1.2.0" \
  "ipdb>=0.13.13" "deepdiff>=8.6.1" "orderly-set" "cachebox" \
  "killport>=1.2.0" "pyee" "gymnasium" "beautifulsoup4" "soupsieve" \
  "dataclasses-json" "marshmallow" "typing-inspect" "mypy_extensions" \
  "sglang-router==0.3.2" "math_verify" "latex2sympy2_extended" \
  || exit 1

pip install --no-deps "$ROOT/compat/sgl_kernel_npu_shim" || exit 1

echo "=== playwright chromium (proxy MITM needs the corporate CA) ==="
NODE_EXTRA_CA_CERTS="$CA" playwright install chromium || exit 1

echo "=== sanity ==="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import open_apps; print('open_apps', open_apps.__file__)"
echo "SETUP_OPENAPPS_OK"
