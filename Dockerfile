FROM python:3.10-slim

# Install system libraries for OpenCV and Pyzbar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code
COPY . .

# Hugging Face strictly requires apps to run on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]