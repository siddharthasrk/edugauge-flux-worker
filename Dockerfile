FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

WORKDIR /

RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    supabase \
    pillow \
    safetensors

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]