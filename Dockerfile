Dockerfile
# Submission image: bot_four.py + mechfight.py + grid_cache.bin (+ the arena folder if it is in the repo), all at the repo root.
FROM python:3.13-slim
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot_four.py"]
 
