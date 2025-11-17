FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents to /app in the container
COPY . /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "stream_dash.py"]