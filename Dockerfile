FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       libgl1 \
       libglib2.0-0 \
       libjpeg-dev \
       zlib1g-dev \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY pyproject.toml /app/
COPY README.md /app/

# Upgrade pip & install project dependencies (including grpcio and lerobot)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Copy the rest of your codebase
COPY . /app

# Expose policy server port
EXPOSE 8000

# Run the async policy server with patches applied
# The wrapper script imports camera_fix.py which applies the Pi0 transformer check bypass
CMD ["python", "RoboHack/server/run_policy_server_with_patches.py", "policy-server", "--host", "0.0.0.0", "--port", "8000"]
