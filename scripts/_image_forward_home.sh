# Find the image-forward checkout. The shims in this folder source this file.
#
# The order:
#   1. IMAGE_FORWARD_HOME in the environment
#   2. the IMAGE_FORWARD_HOME key in the server.env of this repo
#   3. the console script image-forward-home (the installed package)
#   4. the dagspaces package of `python`
# It exports IMAGE_FORWARD_HOME, and IMAGE_FORWARD_PROJECT_ROOT (this repo)
# when that is not set.
_shim_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${IMAGE_FORWARD_HOME:-}" ] && [ -f "$_shim_repo/server.env" ]; then
  IMAGE_FORWARD_HOME="$(sed -n -E 's/^(export )?IMAGE_FORWARD_HOME=["'\'']?([^"'\'' #]*).*/\2/p' "$_shim_repo/server.env" | tail -1)"
fi
if [ -z "${IMAGE_FORWARD_HOME:-}" ] && command -v image-forward-home >/dev/null 2>&1; then
  IMAGE_FORWARD_HOME="$(image-forward-home 2>/dev/null)"
fi
if [ -z "${IMAGE_FORWARD_HOME:-}" ]; then
  IMAGE_FORWARD_HOME="$(python -c 'from dagspaces.common.paths import framework_home; print(framework_home())' 2>/dev/null)"
fi
if [ -z "${IMAGE_FORWARD_HOME:-}" ] || [ ! -d "$IMAGE_FORWARD_HOME/scripts" ]; then
  echo "[monocle] ERROR: cannot find the image-forward checkout. Set IMAGE_FORWARD_HOME in server.env." >&2
  unset _shim_repo
  return 1 2>/dev/null || exit 1
fi
export IMAGE_FORWARD_HOME
export IMAGE_FORWARD_PROJECT_ROOT="${IMAGE_FORWARD_PROJECT_ROOT:-$_shim_repo}"
unset _shim_repo
