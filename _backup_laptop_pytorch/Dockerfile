# PyTorch image for the information-leakage experiments.
# CPU (default):  docker compose build leakage
# CUDA (GPU PC):  docker compose --profile gpu build leakage-gpu
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.9.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
 && pip install -r /tmp/requirements.txt

# Code is also bind-mounted at /app by docker-compose, so edits do not need a rebuild.
COPY . /app

CMD ["python", "run_leakage.py", "--help"]
