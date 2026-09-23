from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Exploration-Lab/PoseStitch-ISL",
    repo_type="dataset",
    allow_patterns=["BLIMP-ISL.csv"],
    local_dir="datasets/posestitch_isl",
    resume_download=True,
)
print("PoseStitch BLIMP-ISL download complete", flush=True)
