# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent .pyc files; ensure stdout/stderr flush immediately to container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────────
# libgomp1  : OpenMP runtime — kept because onnxruntime uses it for parallel ops
# locales   : required so ONNX string_normalizer can initialise correctly
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Set the locale so ONNX string_normalizer can initialise correctly
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# ── Non-root user ──────────────────────────────────────────────────────────────
# Security best practice: never run application code as root inside a container
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements.txt FIRST so this layer is cached and does not re-run on
# every code change — only when requirements.txt itself changes
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code + model files ─────────────────────────────────────────────────────
# Copied after pip install: code changes only invalidate this layer (and below),
# not the dependency layer above.
COPY app/ .

# Transfer ownership of all /app contents to appuser
RUN chown -R appuser:appuser /app

# ── Switch to non-root user ────────────────────────────────────────────────────
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
