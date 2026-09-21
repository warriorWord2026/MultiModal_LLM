# Используем официальный легкий образ Python
FROM python:3.10-slim

# Устанавливаем системные утилиты для сборки 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем и устанавливаем библиотеки 
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sentence-transformers \
    qdrant-client==1.11.0 \
    numpy \
    jinja2\
    httpx

# Копируем наш файл с кодом поиска внутрь контейнера
COPY main.py .
COPY templates ./templates 

# Указываем команду для запуска веб-сервера при старте контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
