# Rice Classifier Service

This service provides a REST API to classify rice grain images into one of five varieties: Ipsala, Jasmine, Arborio, Karacadag, or Basmati.

## Prerequisites
- Docker installed locally
- `rice_model.keras` (the trained Keras model file)

## Build the Docker Image
```bash
# Standard Docker build (include the dot to specify the build context):
# - `-t` tags the image
# - `.` is required as the path/context
docker build -t rice-classifier .  

# Alternatively, using Docker Buildx (ensure you include the build context):
docker buildx build --tag rice-classifier .