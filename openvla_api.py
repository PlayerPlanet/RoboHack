"""FastAPI server that runs OpenVLA locally for image+prompt inference.

This app loads a local OpenVLA model (default: `openvla/openvla-7b`) using Hugging Face
transformers and exposes a simple POST /predict endpoint that accepts an image file and
an optional text prompt. The model is loaded lazily on first request or can be preloaded
on startup by setting `OPENVLA_PRELOAD=1`.

Notes:
- Running `openvla-7b` requires enough GPU memory (or an appropriate `device_map`
    and 8-bit/bnb support). This script attempts to use the default `pipeline("image-to-text")`
    which works for many image->text multimodal models. If you need advanced device
    configuration (bitsandbytes, 8-bit, or vRAM tuning), set the `TRANSFORMERS_*`
    or configure `device_map` in environment and adjust this file accordingly.

Environment variables:
- OPENVLA_MODEL: model id to load (default: openvla/openvla-7b)
- OPENVLA_PRELOAD: if set to '1', try to load model at startup
- OPENVLA_BIND / OPENVLA_PORT: host/port for uvicorn when running directly
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

import io
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Ensure attention backend env var is set before transformers (and torch attention)
# are imported—this helps ensure the backend selection happens early enough so
# custom model init paths (which check _supports_sdpa etc.) see the desired backend.
_openvla_attn = os.environ.get('OPENVLA_ATTENTION')
if _openvla_attn:
    # If user explicitly set OPENVLA_ATTENTION, mirror it to PYTORCH_ATTENTION_BACKEND
    os.environ['PYTORCH_ATTENTION_BACKEND'] = _openvla_attn

try:
    # heavy imports for local inference
    import torch
    from transformers import pipeline
    Pipeline = Any
    from PIL import Image
    # import helpers for manual model/config loading (used as a fallback)
    try:
        from transformers import AutoConfig, AutoModel
    except Exception:
        AutoConfig = None
        AutoModel = None
except Exception:  # pragma: no cover - runtime dependency
    torch = None
    pipeline = None
    Pipeline = None
    Image = None
    AutoConfig = None
    AutoModel = None

LOG = logging.getLogger("openvla_api")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Configure transformers attention backend early (try flash_attention_2, fall back to eager).
try:
    # set_attention_backend exists in newer transformers versions
    from transformers import set_attention_backend  # type: ignore

    pref = os.environ.get('OPENVLA_ATTENTION', 'auto').lower()
    if pref == 'auto':
        try:
            set_attention_backend('flash_attention_2')
            LOG.info('Set transformers attention backend to flash_attention_2')
        except Exception:
            try:
                set_attention_backend('eager')
                LOG.info('flash_attention_2 not available, set attention backend to eager')
            except Exception:
                os.environ['PYTORCH_ATTENTION_BACKEND'] = 'eager'
                LOG.info('Fell back to PYTORCH_ATTENTION_BACKEND=eager')
    else:
        try:
            set_attention_backend(pref)
            LOG.info('Set transformers attention backend to %s', pref)
        except Exception:
            os.environ['PYTORCH_ATTENTION_BACKEND'] = pref
            LOG.info('Set env PYTORCH_ATTENTION_BACKEND=%s', pref)
except Exception as _e:
    LOG.debug('Could not set attention backend via transformers API: %s', _e)

app = FastAPI(title="OpenVLA local OpenVLA server")

# global pipeline instance (lazy loaded)
_local_pipe: Optional[Any] = None


def get_model_id() -> str:
    return os.environ.get('OPENVLA_MODEL', 'openvla/openvla-7b')


def is_torch_available() -> bool:
    return torch is not None


def load_local_pipeline(model_id: Optional[str] = None) -> Any:
    """Load and return an image-to-text pipeline for the given model id.

    This function tries a couple of sensible defaults and will raise a RuntimeError
    if loading fails. For large models you'll want to configure device_map, dtype,
    and potentially use bitsandbytes; this loader keeps things simple but exposes
    where to change settings.
    """

    global _local_pipe
    if _local_pipe is not None:
        return _local_pipe

    if pipeline is None:
        raise RuntimeError("transformers and related libraries are not installed")

    model = model_id or get_model_id()
    LOG.info("Loading local model: %s", model)

    # choose device
    device = 0 if torch and torch.cuda.is_available() else -1

    # Attempt to create an image-to-text pipeline
    try:
        # Some remote model code expects the base Model class to have a
        # `_supports_sdpa` attribute present on instances. Patch the
        # transformers Model class to provide a default to avoid
        # AttributeError during model init.
        try:
            # Ensure instances of common base classes get default flags for
            # attention support. Some custom model classes check instance
            # attributes (like _supports_sdpa) during __init__, so setting
            # class attributes alone may not be sufficient. We wrap the
            # base __init__ to ensure every instance receives sane defaults.
            import torch.nn as _nn
            try:
                _orig_nn_init = _nn.Module.__init__

                def _nn_patched_init(self, *a, **kw):
                    _orig_nn_init(self, *a, **kw)
                    # Default to False for SDPA so models don't try to
                    # dispatch into SDPA paths unexpectedly. Enable
                    # flash_attn flag to allow flash_attention_2 when
                    # available.
                    if not hasattr(self, '_supports_sdpa'):
                        try:
                            object.__setattr__(self, '_supports_sdpa', False)
                        except Exception:
                            setattr(self, '_supports_sdpa', False)
                    if not hasattr(self, '_supports_flash_attn'):
                        try:
                            object.__setattr__(self, '_supports_flash_attn', False)
                        except Exception:
                            setattr(self, '_supports_flash_attn', False)

                _nn.Module.__init__ = _nn_patched_init
                LOG.info('Patched torch.nn.Module.__init__ to set instance _supports_sdpa/_supports_flash_attn defaults')
            except Exception as _e:
                # Fallback: set class attribute so getattr can find it in many cases
                try:
                    if not hasattr(_nn.Module, '_supports_sdpa'):
                        setattr(_nn.Module, '_supports_sdpa', False)
                        LOG.info("Patched torch.nn.Module._supports_sdpa = False (fallback)")
                except Exception:
                    LOG.debug('Could not patch torch.nn.Module defaults: %s', _e)

            try:
                import transformers.modeling_utils as _modeling_utils
                pretrain_cls = getattr(_modeling_utils, 'PreTrainedModel', None)
                if pretrain_cls is not None:
                    try:
                        _orig_pt_init = pretrain_cls.__init__

                        def _pt_patched_init(self, *a, **kw):
                            _orig_pt_init(self, *a, **kw)
                            if not hasattr(self, '_supports_sdpa'):
                                try:
                                    object.__setattr__(self, '_supports_sdpa', False)
                                except Exception:
                                    setattr(self, '_supports_sdpa', False)
                            if not hasattr(self, '_supports_flash_attn'):
                                try:
                                    object.__setattr__(self, '_supports_flash_attn', False)
                                except Exception:
                                    setattr(self, '_supports_flash_attn', False)

                        pretrain_cls.__init__ = _pt_patched_init
                        LOG.info('Patched transformers.PreTrainedModel.__init__ to set _supports_sdpa defaults')
                    except Exception as _e:
                        # fallback to class attrs
                        if not hasattr(pretrain_cls, '_supports_sdpa'):
                            setattr(pretrain_cls, '_supports_sdpa', False)
                            LOG.info('Patched transformers.PreTrainedModel._supports_sdpa = False (fallback)')
            except Exception as _e:
                LOG.debug('Could not patch transformers Model classes: %s', _e)
        except Exception as _e:
            LOG.debug("Could not patch torch.nn.Module: %s", _e)

        _local_pipe = pipeline("image-to-text", model=model, device=device, trust_remote_code=True)
        LOG.info("Model loaded successfully (device=%s)", device)
        return _local_pipe
    except Exception as exc:
        LOG.exception("Failed to load pipeline with simple settings: %s", exc)
        # Try with device_map='auto' (requires accelerate)
        try:
            _local_pipe = pipeline("image-to-text", model=model, device_map='auto', trust_remote_code=True)
            LOG.info("Model loaded with device_map='auto'")
            return _local_pipe
        except Exception as exc2:
            LOG.exception("Failed to load model with device_map='auto': %s", exc2)
            # As a last resort, try to load config and model manually and patch missing attributes
            if AutoConfig is not None and AutoModel is not None:
                try:
                    LOG.info("Attempting manual config+model load as a fallback")
                    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
                    # Prefer flash_attention_2 for this model; disable SDPA support
                    # The model's custom code expects certain config attributes — set them explicitly.
                    try:
                        setattr(cfg, "_supports_sdpa", False)
                        setattr(cfg, "_supports_flash_attn", True)
                        setattr(cfg, "_attn_implementation_internal", "flash_attention_2")
                        LOG.info("Patched config attention impl -> flash_attention_2")
                    except Exception as _e:
                        LOG.debug("Could not patch cfg attention attrs: %s", _e)

                    # Load the model with the patched config
                    model_obj = AutoModel.from_pretrained(
                        model, config=cfg, trust_remote_code=True, device_map='auto'
                    )
                    _local_pipe = pipeline("image-to-text", model=model_obj, device_map='auto')
                    LOG.info("Manual load succeeded with patched attention settings")
                    return _local_pipe
                except Exception as exc3:
                    LOG.exception("Manual load fallback failed: %s", exc3)

            raise RuntimeError(f"Unable to load model {model}: {exc2}")


@app.on_event("startup")
def maybe_preload():
    """Optionally preload the model at startup when OPENVLA_PRELOAD=1 is set."""
    try:
        preload = os.environ.get('OPENVLA_PRELOAD', '0')
        if preload == '1':
            LOG.info("OPENVLA_PRELOAD=1 -> preloading model")
            load_local_pipeline()
    except Exception as exc:
        LOG.exception("Preload failed: %s", exc)


def get_hf_token() -> Optional[str]:
    return os.environ.get('HUGGINGFACE_API_TOKEN')


def get_default_model() -> Optional[str]:
    return os.environ.get('HUGGINGFACE_MODEL')


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_id": get_model_id(),
        "transformers_installed": pipeline is not None,
        "torch_available": is_torch_available(),
        "model_loaded": _local_pipe is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None),
):
    """Run local inference on an uploaded image with an optional prompt.

    The model is loaded locally (lazy load). Returns the pipeline output as JSON.
    """

    # Immediately patch common transformers base classes to avoid missing attributes
    try:
        import transformers.modeling_utils as _modeling_utils
        pretrain_cls = getattr(_modeling_utils, 'PreTrainedModel', None)
        if pretrain_cls is not None and not hasattr(pretrain_cls, '_supports_sdpa'):
            setattr(pretrain_cls, '_supports_sdpa', True)
            LOG.info('Patched transformers.PreTrainedModel._supports_sdpa = True')
    except Exception:
        # not fatal; we'll attempt other fallbacks later
        pass
    # read file bytes and convert to PIL Image
    try:
        contents = await file.read()
    finally:
        await file.close()

    if Image is None:
        raise HTTPException(status_code=500, detail="PIL/transformers/torch not available on server")

    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        # try to accept raw bytes as fallback
        import io as _io

        try:
            img = Image.open(_io.BytesIO(contents)).convert('RGB')
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}")

    # choose model id
    model = model_id or get_model_id()

    # load pipeline (may raise RuntimeError)
    try:
        pipe = load_local_pipeline(model)
    except Exception as exc:
        LOG.exception("Failed to load local model")
        raise HTTPException(status_code=500, detail=str(exc))

    # run the pipeline. Many image->text pipelines accept a prompt keyword; try both.
    try:
        if prompt is not None:
            try:
                out = pipe(img, prompt=prompt)
            except TypeError:
                out = pipe(img)
        else:
            out = pipe(img)
    except Exception as exc:
        LOG.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # normalize output
    try:
        # pipeline often returns a list of dicts like [{'generated_text': '...'}]
        return JSONResponse(content={"model": model, "result": out})
    except Exception:
        return JSONResponse(content={"model": model, "result": str(out)})


if __name__ == '__main__':
    # run with uvicorn if executed directly
    import uvicorn

    host = os.environ.get('OPENVLA_BIND', '0.0.0.0')
    port = int(os.environ.get('OPENVLA_PORT', '8000'))
    LOG.info("Starting OpenVLA proxy on %s:%d", host, port)
    uvicorn.run("openvla_api:app", host=host, port=port, reload=False)
