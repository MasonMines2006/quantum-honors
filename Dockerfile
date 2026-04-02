# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — quantum-honors
#
# Builds a self-contained environment with all dependencies pre-installed.
# You never need to run `pip install` manually — it happens during `docker build`.
#
# Base image: python:3.11-slim
#   - Official Python image, Debian-based
#   - "slim" strips out docs and test files to keep the image small
#   - 3.11 is the most stable version for PennyLane + PyTorch on CPU
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set the working directory inside the container.
# All subsequent commands run relative to this path.
# Your project files will be copied here.
WORKDIR /app

# Copy requirements first — before the rest of the code.
# Docker builds in layers and caches each one. By copying requirements.txt
# separately and installing before copying the source code, Docker can reuse
# the dependency layer on rebuilds as long as requirements.txt hasn't changed.
# This means re-running `docker build` after editing a .py file is fast.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir tells pip not to store the download cache inside the image,
# which keeps the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project source code into the container.
# The .dockerignore file controls what gets excluded (results, pycache, etc.)
COPY . .

# The default command run when you do `docker run quantum-honors`.
# Runs the full experiment. Override this on the command line if needed,
# e.g.: docker run quantum-honors python -c "import pennylane; print(pennylane.__version__)"
CMD ["python", "main.py"]
