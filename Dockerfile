FROM python:3.12-slim AS base

WORKDIR /app

# System deps for numba/numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/
COPY data_fixed/ data_fixed/
COPY runs/ runs/
COPY proto/ proto/
COPY solve.py ./
COPY http_server.py ./
COPY grpc_server.py ./
COPY entrypoint.sh ./

RUN pip install --no-cache-dir -e .

RUN chmod +x entrypoint.sh

EXPOSE 8100 50051

ENTRYPOINT ["./entrypoint.sh"]
