# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose port
EXPOSE 8999

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8999
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the application
CMD streamlit run app.py --server.port=8999 --server.address=0.0.0.0 --server.baseUrlPath=""