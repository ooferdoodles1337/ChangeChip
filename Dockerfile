# Use an official Python runtime as a parent image
FROM python:3.11.9-bookworm

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Set the environment variable for the Gradio server name
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run app.py when the container launches
CMD ["python", "app.py"]
