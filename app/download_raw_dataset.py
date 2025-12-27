from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="vectara/open_ragbench",
    repo_type="dataset",
    local_dir="data/open_ragbench_raw",
    local_dir_use_symlinks=False,
)

print("Dataset downloaded to:", local_dir)
