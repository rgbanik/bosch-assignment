FROM python:3.12-slim

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and prepare script
COPY src /app/src
COPY restructure_dataset.sh /app/restructure_dataset.sh

# Mount your dataset at runtime
VOLUME ["/data"]

# Expose Streamlit port
EXPOSE 8501

# Run dataset restructuring, then start Streamlit
ENTRYPOINT ["/bin/bash", "-c", "/app/restructure_dataset.sh && streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0"]
