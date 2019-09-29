# Use an official Python runtime as a parent image
FROM python:3

# Set the myself as the maintainer, for DockerHub push
MAINTAINER Demond Love <dhlove1s@gmail.com>

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# ADD requirements.txt /

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME DLove

# Run app file when the container launches
CMD ["python","./src/pandasWorkflow/pandasDataPipeline.py"]