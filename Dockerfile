FROM python:3.12-slim

WORKDIR /app

# common runtime libs for opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Note: rpicam-vid and camera libs should be mounted from host
CMD ["python", "realtime_pi5_dual.py"]
