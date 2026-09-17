FROM python:3.11.16-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAX_ENABLE_X64=True

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-verified.txt /tmp/requirements-verified.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       -r /tmp/requirements-verified.txt

WORKDIR /workspace

CMD ["python"]