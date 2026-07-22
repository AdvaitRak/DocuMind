FROM python:3.12-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for better caching
COPY requirements.txt .

# install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# pre-download reranker model so it's baked into image
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', \
    backend='onnx', \
    model_kwargs={'file_name': 'onnx/model_quint8_avx2.onnx'}); \
    print('Model preloaded')"

# copy application code
COPY . .

# create data directory
RUN mkdir -p data/pdfs indexes

# expose port
EXPOSE 8000

# run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]