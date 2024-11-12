# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies for your application, including nano
RUN apt-get update && \
    apt-get install -y gcc g++ libpq-dev fonts-nanum fontconfig nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Refresh font cache to recognize the newly installed fonts
RUN fc-cache -fv

# Copy requirements.txt into the image
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy the application code into the image
COPY . .

# Expose port
EXPOSE 5000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

