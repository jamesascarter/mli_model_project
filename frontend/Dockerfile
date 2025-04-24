# Use the official Streamlit image as the base
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code into the container
COPY . /app

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run Streamlit
CMD STREAMLIT_WATCHER_TYPE=none streamlit run app.py --server.port=8501 --server.address=0.0.0.0
