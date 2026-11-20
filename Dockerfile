# PyTorch image for the leakage experiments.
#   CPU (laptop):  docker compose build leakage
#   GPU (CUDA):    docker compose --profile gpu build leakage-gpu
FROM python:3.11-slim

LABEL description="Subliminal learning leakage experiments (PyTorch)"

ARG TORCH_VERSION=2.9.1
# cpu | cu126 | cu128 ... (PyTorch wheel index)
ARG TORCH_VARIANT=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN pip install --upgrade pip \
 && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

COPY requirements.txt .
RUN pip install -r requirements.txt

# The repository is mounted at /app by docker compose, so code, data/ and results/ stay on the host.
COPY . /app

CMD ["python", "run_leakage.py", "--help"]
