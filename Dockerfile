# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Runtime settings ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---------- System deps ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Project code ----------
COPY . .

# ---------- Default CMD -------------
# → Starts the dashboard; override with `bash` if you just want a shell.
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.headless=true", "--server.port=8501"]