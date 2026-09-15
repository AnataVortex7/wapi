#!/bin/bash
# Replace the password placeholder in nginx config with the environment variable
sed -i "s/_DYNAMIC_PASSWORD_/${API_PASSWORD}/g" /etc/nginx/nginx.conf

# Start panel (which starts gemini on 8081) in background
python3 panel.py > panel.log 2>&1 &

# Install required packages for Smart Router
pip install flask requests

# Start the Smart Router on port 8085
python3 smart_router.py > router.log 2>&1 &

# Wait a couple seconds for it to start
sleep 3

# Start Nginx in foreground
nginx -g 'daemon off;'
