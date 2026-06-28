FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /

# Stable package versions for FLUX + RunPod Serverless.
# Do not download model weights during Docker build.
# Attach RunPod Network Volume at /runpod-volume for Hugging Face cache.
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    sentencepiece \
    protobuf \
    supabase \
    pillow \
    safetensors

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
