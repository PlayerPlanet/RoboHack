This repository includes a minimal Dockerfile and docker-compose configuration to run the policy server.

Build the image (PowerShell):

```powershell
docker build -t robothack-policy-server .
```

Run with Docker:

```powershell
docker run --rm -p 8000:8000 robothack-policy-server
```

Or use docker-compose:

```powershell
docker-compose up --build
```

Notes:
- The project declares heavy ML dependencies (torch, transformers, bitsandbytes, etc.). Building the image will download and install those packages and may take significant time and disk space. For GPU/cuda support you must use a base image with CUDA toolchain and adapt the Dockerfile accordingly.
- If you only want to run without real hardware, the server can be started against fake hardware; the repository's server uses `SO101Env(fake_hardware=True)` in local runs. Adjust environment variables or arguments as needed.
