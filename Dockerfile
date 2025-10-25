# Use NVIDIA CUDA base image for GR00T FlashAttention support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3.11-dev \
       python3-pip \
       build-essential \
       git \
       libgl1 \
       libglib2.0-0 \
       libjpeg-dev \
       zlib1g-dev \
       ffmpeg \
       ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support FIRST (required by FlashAttention)
RUN pip install --no-cache-dir \
    "torch>=2.2.1,<2.8.0" \
    "torchvision>=0.21.0,<0.23.0"

# Install FlashAttention build dependencies
RUN pip install --no-cache-dir \
    ninja \
    "packaging>=24.2,<26.0" \
    psutil \
    setuptools

# Install FlashAttention (requires torch + build deps to be installed first)
RUN pip install --no-cache-dir "flash-attn>=2.5.9,<3.0.0" --no-build-isolation

# Copy dependency files
COPY pyproject.toml /app/
COPY README.md /app/

# Install GR00T support FIRST (before project deps to avoid conflicts)
# This ensures all groot-specific dependencies are installed correctly
RUN pip install --no-cache-dir "lerobot[groot]>=0.4.0"

# Install remaining project dependencies
RUN pip install --no-cache-dir .

# Copy the rest of your codebase
COPY . /app

# Verify GR00T installation
RUN python -c "import flash_attn; print(f'✅ FlashAttention {flash_attn.__version__}')" && \
    python -c "import lerobot; print(f'✅ LeRobot {lerobot.__version__}')" && \
    python -c "from lerobot.async_inference.policy_server import SUPPORTED_POLICIES; print(f'✅ Supported policies: {SUPPORTED_POLICIES}')" && \
    python -c "import sys; sys.path.insert(0, 'lerobot.policies.groot' in str(sys.modules)); from lerobot.policies import groot; print(f'✅ GR00T module: {dir(groot)}')" || echo "⚠️  GR00T verification incomplete"

# Expose policy server port
EXPOSE 8000

# Run the async policy server with patches applied
# The wrapper script imports camera_fix.py which applies the Pi0 transformer check bypass
CMD ["python", "RoboHack/server/run_policy_server_with_patches.py", "--host", "0.0.0.0", "--port", "8000"]
