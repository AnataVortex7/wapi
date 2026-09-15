FROM python:3.10-slim

# Install nginx and git
RUN apt-get update && apt-get install -y nginx git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone web2api
RUN git clone https://github.com/cyberanrhy/gemini-claude-web2api.git .
RUN pip install --no-cache-dir -r requirements.txt

# Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Start script and Router
COPY smart_router.py /app/smart_router.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
