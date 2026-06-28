FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

WORKDIR /

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
