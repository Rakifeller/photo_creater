FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1
RUN pip3 install --upgrade pip
RUN pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Proje dosyaları
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Model cache (isteğe bağlı)
ENV HF_HOME=/models
ENV HF_TOKEN=${HF_TOKEN}

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
