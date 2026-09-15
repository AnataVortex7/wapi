#!/bin/bash
# Start panel (which starts gemini on 8081) in background
python3 panel.py > panel.log 2>&1 &

# Wait a couple seconds for it to start
sleep 3

# Start Nginx in foreground
nginx -g 'daemon off;'
