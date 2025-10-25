FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system packages required for building some Python packages and OpenCV
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

# Copy project metadata first for caching installs
COPY pyproject.toml /app/

# Upgrade pip and install the package (this will pull dependencies listed in pyproject.toml)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir /app

# Copy the rest of the repository
COPY . /app

# Expose the default port used by the policy server
EXPOSE 8000

# Default command to run the policy server
CMD ["python", "-m", "lerobot.async_inference.policy_server", "--host", "0.0.0.0", "--port", "8000"]
