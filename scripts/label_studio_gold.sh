#!/bin/bash
# Label Studio for the urban-cues gold set.
#
#   sbatch scripts/label_studio_gold.sub [port]        # the normal way: a job pinned to lisbeth
#   PORT=8925 bash scripts/label_studio_gold.sh        # by hand, in a shell that stays open
#
# This follows /share/ju/tiktok/glasses_search/annotate/serve.sh and the notes
# in /share/ju/cyclomedia/annotate/serve.sh. They are why this looks the way
# it does:
#   * NO --host. It sets Django's HOSTNAME and Label Studio then emits absolute
#     asset urls (http://lisbeth:8925/static/...), which a browser at
#     localhost:PORT behind an SSH tunnel cannot resolve: unstyled login page,
#     blank white app, perfectly healthy server logs.
#   * NO --internal-host. The tunnel targets the host by its FQDN through a
#     login host, thus the server must bind every interface.
#   * The venv MUST be a standalone CPython: `uv venv --python 3.12
#     --managed-python .venv-labelstudio`, never the anaconda python, or
#     numpy's double-load guard fires on some nodes and not others.
#   * The venv console script carries an absolute shebang: call it as
#     `python <script>`.
#   * Legacy API tokens are off by default in 1.20+, thus --user-token creates
#     a token that every API call rejects with 401. Turn them back on.
#   * The SQLite db lives on shared NFS under data/label_studio/data, NOT $HOME.
#   * Binding 0.0.0.0 on a shared cluster means anyone who can reach the host
#     can reach the app, thus a real password is set and signup is closed.
#
# This is a THIRD, independent instance beside the cyclomedia one (8925 when
# it runs) and the glasses one (8926). Different db, different document root.
# Pass another port if one of them holds 8925.
set -uo pipefail

REPO=/share/ju/matt/mllmsci
PORT="${PORT:-8925}"
VENV="$REPO/.venv-labelstudio"
DATA="$REPO/data/label_studio/data"
CREDS="$REPO/data/label_studio/credentials.env"

export LABEL_STUDIO_BASE_DATA_DIR="$DATA"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$REPO/data/gold_set"
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT="$REPO/data/gold_set"
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true
export COLLECT_ANALYTICS=false
export DO_NOT_TRACK=1

# The first user, its password, and its API token. Mode 600. Change the
# password after the first login and add each annotator as a member.
[ -f "$CREDS" ] || { echo "no credentials at $CREDS" >&2; exit 1; }
set -a; source "$CREDS"; set +a
LS_USER="$LABEL_STUDIO_USERNAME"
LS_PASS="$LABEL_STUDIO_PASSWORD"
LS_TOKEN="$LABEL_STUDIO_USER_TOKEN"

mkdir -p "$DATA"
[ -f "$VENV/bin/label-studio" ] || { echo "no label-studio at $VENV; build it: uv venv --python 3.12 --managed-python $VENV && uv pip install --python $VENV/bin/python label-studio" >&2; exit 1; }
[ -f "$REPO/data/gold_set/tasks_blind.json" ] || { echo "no gold set; run scripts/build_gold_set.py first" >&2; exit 1; }

HOST_FQDN="$(hostname -f)"
echo "=== label studio: gold set, host=$HOST_FQDN port=$PORT ==="

"$VENV/bin/python" "$VENV/bin/label-studio" start \
    --port "$PORT" \
    --username "$LS_USER" --password "$LS_PASS" --user-token "$LS_TOKEN" \
    --no-browser &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' TERM INT

# Readiness, not a fixed sleep: the first start runs Django migrations, and
# NFS starvation can hold it for minutes.
for i in $(seq 1 120); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "$CHILD" 2>/dev/null || { echo "server died during start-up" >&2; exit 1; }
    sleep 5
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
    echo "server never became healthy" >&2; kill "$CHILD" 2>/dev/null; exit 1; }

"$VENV/bin/python" "$REPO/scripts/label_studio_setup.py" --port "$PORT" || {
    echo "!! project bootstrap FAILED (see the .err log); server left running." >&2
}

cat <<BANNER

================================================================
  Label Studio (urban cues gold set) is up.

    on this network :  http://${HOST_FQDN}:${PORT}
    from a laptop   :  ssh -N -L ${PORT}:${HOST_FQDN}:${PORT} \$USER@<login-host>
                       then open http://localhost:${PORT}

    login  :  ${LS_USER}
    pass   :  in ${CREDS}

  When the batch is done: Export -> JSON, then
    .venv-mllmsci-vllm025cu129/bin/python scripts/score_gold_set.py <the-export>.json
================================================================
BANNER

wait "$CHILD"
