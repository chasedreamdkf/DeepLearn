import kagglehub

# Download latest version
path = kagglehub.dataset_download("hemanthhari/dehazing-dataset-thesis")

print("Path to dataset files:", path)