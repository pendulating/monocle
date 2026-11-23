# A shim: this script is in image-forward ($IMAGE_FORWARD_HOME/scripts/load_server_env.sh).
# It works when you source it and when you run it. See README.md.
source "$(dirname "${BASH_SOURCE[0]}")/_image_forward_home.sh" || return 1 2>/dev/null || exit 1
source "$IMAGE_FORWARD_HOME/scripts/load_server_env.sh" "$@"
