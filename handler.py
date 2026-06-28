import os
import io
import uuid
import traceback

# Store Hugging Face model cache in RunPod Network Volume.
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/huggingface")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import runpod
import torch

# Compatibility fix for diffusers/torch combinations where torch.xpu is missing.
# Must be defined before importing FluxPipeline.
if not hasattr(torch, "xpu"):
    class DummyXPU:
        @staticmethod
        def empty_cache():
            pass
    torch.xpu = DummyXPU()

from diffusers import FluxPipeline
from supabase import create_client, Client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "edugauge-gamification-assets")
FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variable.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


print(f"Loading model: {FLUX_MODEL_ID}")
print(f"Cache directory: {CACHE_DIR}")

pipe = FluxPipeline.from_pretrained(
    FLUX_MODEL_ID,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
)

# RTX A5000 has 24GB VRAM. CPU offload is safer for FLUX.
pipe.enable_model_cpu_offload()

try:
    pipe.enable_attention_slicing()
except Exception:
    pass

print("Model loaded successfully.")


def upload_to_supabase(image_bytes: bytes, filename: str, folder: str) -> str:
    bucket_path = f"{folder.strip('/')}/{filename}"

    supabase.storage.from_(SUPABASE_BUCKET).upload(
        path=bucket_path,
        file=image_bytes,
        file_options={
            "content-type": "image/webp",
            "upsert": "false",
        },
    )

    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(bucket_path)


def handler(job):
    try:
        job_input = job.get("input", {})

        prompt = job_input.get(
            "prompt",
            "Create a child-friendly educational adventure map background, no text."
        )
        negative_prompt = job_input.get(
            "negative_prompt",
            "text, watermark, logo, blurry, distorted, unsafe"
        )

        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 768))
        seed = int(job_input.get("seed", 0))
        steps = int(job_input.get("num_inference_steps", 4))
        guidance_scale = float(job_input.get("guidance_scale", 0.0))
        folder = job_input.get("folder", "runpod-generations")
        quality = int(job_input.get("quality", 85))

        # Safety limits for RTX A5000.
        width = max(512, min(width, 1280))
        height = max(512, min(height, 1024))
        steps = max(1, min(steps, 8))
        quality = max(60, min(quality, 95))

        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                output_type="pil",
                generator=generator,
            ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="WEBP", quality=quality)
        image_bytes = img_byte_arr.getvalue()

        filename = f"generated_{uuid.uuid4().hex}.webp"
        public_url = upload_to_supabase(image_bytes, filename, folder)

        return {
            "success": True,
            "image_url": public_url,
            "bucket": SUPABASE_BUCKET,
            "bucket_path": f"{folder.strip('/')}/{filename}",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "model": FLUX_MODEL_ID,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
