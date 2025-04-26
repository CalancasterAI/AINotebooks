# Use a minimal Python image
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files
COPY rice_model.keras ./
COPY inference.py ./
COPY app.py ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]