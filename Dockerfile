FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY agentic_har ./agentic_har
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
    && pip install . \
    && useradd --uid 10001 --create-home har

USER har
EXPOSE 8000
CMD ["uvicorn", "agentic_har.api:app", "--host", "0.0.0.0", "--port", "8000"]
