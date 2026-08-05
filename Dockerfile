FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip gmsh libgl1 && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.10 -m pip install --no-deps .
ENTRYPOINT ["boneagent-campaign"]

