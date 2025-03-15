FROM ubuntu:24.04


# Install system dependencies and add Deadsnakes PPA
RUN apt-get update && \
    apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopengl0 \
    mesa-utils \
    xvfb \
    libxkbcommon-x11-0 \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    gcc \
    g++ \
    git \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \ 
    python3-wheel \ 
    python3-venv

RUN apt-get update && apt-get install -y \
    ninja-build 


# Upgrade pip and install dependencies
RUN pip install --upgrade --force-reinstall setuptools ninja --break-system-packages

# Create app directory
WORKDIR /app

# Clone live2d-py repository
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN git clone https://github.com/Arkueid/live2d-py --depth 1
WORKDIR /app/live2d-py
RUN make .
RUN cmake
RUN pip install .
RUN pip install live2d-py

WORKDIR /app


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p Resources

# Set display variable for X virtual framebuffer
ENV DISPLAY=:99

# Start Xvfb, then run the application
CMD ["sh", "-c", "Xvfb :99 -screen 0 1280x720x24 -ac & python engine.py"]
