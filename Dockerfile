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

# copy application code
COPY . .

# create data directory
RUN mkdir -p data/pdfs indexes

# expose port
EXPOSE 8000

# run the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]