# Use TensorFlow GPU image as the base
FROM tensorflow/tensorflow:latest-gpu

# Add a description label
LABEL description="Tensorflow GPU Container for Subliminal Learning"

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and setuptools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools

# Force reinstall blinker to avoid distutils issues
RUN pip install --upgrade --no-cache-dir --force-reinstall --ignore-installed blinker

# Install TensorFlow with CUDA support
RUN pip --no-cache-dir install tensorflow[and-cuda]

# Install required Python libraries
RUN pip install --no-cache-dir pandas matplotlib scikit_learn tables seaborn xlrd plotly

# Copy the application files into the container
ADD . /app

# Set the environment variable for production mode
ENV DASH_DEBUG_MODE=False

# Expose the port for the application
EXPOSE 8050