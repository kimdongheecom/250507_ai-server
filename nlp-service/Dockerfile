FROM python:3.12.7-slim

# Install OpenJDK and required dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    default-jdk \
    default-jre \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variable for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

EXPOSE 9003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9003", "--reload"]