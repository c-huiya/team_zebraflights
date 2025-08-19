# ui/Dockerfile
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app 

# Copy dependency list
COPY ui/requirements.txt ./requirements.txt
# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

 # Copy all app code into the container
COPY ui/ ./ 

# Expose port 5002 (Flask app will run here)
EXPOSE 5002

 #Start the app using Python
CMD ["python", "-u", "app.py"]
