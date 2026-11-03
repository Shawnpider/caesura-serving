#!/usr/bin/env bash
# Install Caesura into SGLang 0.5.18: apply the hooks to SGLang's scheduler and copy the modules of overlay/ next to it.
# Run it once in the environment that serves the model, for example the lmsysorg/sglang:v0.5.18-cu130-runtime image.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
version=$(python3 -c 'import importlib.metadata as m; print(m.version("sglang"))')
[ "$version" = 0.5.18 ] || { echo "Caesura needs SGLang 0.5.18, found $version" >&2; exit 1; }
pkg=$(python3 -c 'import os, sglang; print(os.path.dirname(sglang.__file__))')
patch -d "$(dirname "$pkg")" -p2 --forward < "$here/sglang_hooks.patch"
cp "$here"/overlay/*.py "$pkg/srt/managers/"
echo "Caesura installed into $pkg"
