import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "nyu-visionx/cambrian-13b",
    local_dir = "/share/pierson/matt/zoo/models/cambrian-13b"
)















