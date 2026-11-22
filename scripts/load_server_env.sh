# Sourced (not executed). Export the keys of server.env into this shell.
#
# A SLURM job does not always get the environment of the process that
# submitted it. For example, an sbatch call through ssh, or a cluster that sets
# --export=NONE. This script gives the job the site keys again.
#
# The rules are the same as ensure_dotenv() in dagspaces/common/stage_utils.py:
#   - The file is $IMAGE_FORWARD_SERVER_ENV, or else <repo root>/server.env.
#   - A key that the shell already sets is not overwritten.
#   - Lines are KEY=value, with an optional "export " and optional quotes.
#   - The script does not expand $VAR in a value.

_ife_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ife_file="${IMAGE_FORWARD_SERVER_ENV:-$_ife_root/server.env}"
if [ -f "$_ife_file" ]; then
  while IFS= read -r _ife_line || [ -n "$_ife_line" ]; do
    _ife_line="${_ife_line#export }"
    case "$_ife_line" in ''|'#'*) continue ;; esac
    _ife_key="${_ife_line%%=*}"
    _ife_val="${_ife_line#*=}"
    [[ "$_ife_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -n "${!_ife_key+x}" ] && continue
    # Remove 1 pair of quotes around the value.
    if [[ "$_ife_val" =~ ^\"(.*)\"$ ]] || [[ "$_ife_val" =~ ^\'(.*)\'$ ]]; then
      _ife_val="${BASH_REMATCH[1]}"
    fi
    export "$_ife_key=$_ife_val"
  done < "$_ife_file"
fi
if [ -z "${IMAGE_FORWARD_PROJECT_ROOT:-}" ]; then
  export IMAGE_FORWARD_PROJECT_ROOT="$_ife_root"
fi
unset _ife_root _ife_file _ife_line _ife_key _ife_val
