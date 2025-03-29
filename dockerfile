FROM ubuntu:24.04

# Debug: check what ALSA packages are available
RUN apt-get update && apt-cache search alsa

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
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
    python3-venv \
    ninja-build \
    # Try alternative ALSA-related packages
    alsa-base \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libpulse-dev \
    # Add these packages for OpenGL support in headless environment
    x11-xserver-utils \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-dri2-0 \
    libxcb-dri3-0 \
    libxcb-present0 \
    libxcb-sync1 \
    libxxf86vm1 \
    libdrm2 \
    libxkbcommon0 \
    libegl-mesa0

# Upgrade pip and install dependencies
RUN pip install --upgrade --force-reinstall setuptools ninja --break-system-packages

# Create app directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Clone the modified live2d-py repository
RUN git clone https://github.com/thedudeontitan/live2d-py --depth 1
WORKDIR /app/live2d-py

# Build and install live2d-py
RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_INSTALLATION_PATH=/usr/bin && \
    cmake --build . --config Release && \
    cd .. && \
    pip install -e .

# Go back to app directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required directories
RUN mkdir -p Resources

# Set environment variables to use dummy audio
ENV SDL_AUDIODRIVER=dummy
ENV AUDIODEV=/dev/null

# Set environment variables for OpenGL to run in Xvfb
ENV DISPLAY=:99
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV SDL_VIDEODRIVER=x11

# Start Xvfb with proper configuration and wait for it to initialize before running the Python app
CMD ["sh", "-c", "Xvfb :99 -screen 0 1280x720x24 -ac & sleep 2 && python engine.py"]