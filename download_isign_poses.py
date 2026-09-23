from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Exploration-Lab/iSign",
    repo_type="dataset",
    allow_patterns=["iSign-poses_v1.1_part_*"],
    local_dir="datasets/isign_pose_parts",
    resume_download=True,
)
print("iSign pose parts download complete", flush=True)
