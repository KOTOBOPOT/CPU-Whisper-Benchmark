import kagglehub

# Download latest version
path = kagglehub.dataset_download("alexcumder/russian-asr-golos")

print("Path to dataset files:", path)