import os
# Включаем строгий оффлайн-режим для всех локальных моделей
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import httpx
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
app = FastAPI(title="Семантическая Библиотека ВУЗа")

# Настраиваем папку с HTML-шаблонами    
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# 1. Инициализация оффлайн-модели SBERT
print("Загрузка локальной модели SBERT...")
model_path = "/app/model_cache"
try:
    model = SentenceTransformer(model_path, local_files_only=True)
except TypeError:
    from sentence_transformers import models
    word_embedding_model = models.Transformer(model_path, max_seq_length=384)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
print("Модель успешно загружена!")

# 2. Подключение к Qdrant в Docker
qdrant_host = os.getenv("QDRANT_HOST", "vector_db")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

COLLECTION_NAME = "academic_articles"
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
# Наша текстовая база данных
ARTICLES_DB = {
    1: {"title": "Нейросети в медицине", "abstract": "Применение сверточных нейросетей для автоматического распознавания опухолей на снимках МРТ головного мозга."},
    2: {"title": "Блокчейн и безопасность", "abstract": "Исследование уязвимостей смарт-контрактов в децентрализованных финансовых приложениях и методы их защиты."},
    3: {"title": "Поиск текстовой информации", "abstract": "Разработка систем семантического поиска документов на основе эмбеддингов трансформеров BERT и векторных СУБД."},
    4: {"title": "Оптимизация СУБД", "abstract": "Методы индексации больших объемов данных в реляционных базах для ускорения аналитических запросов."},
    5: {"title": "Глубокое обучение в лингвистике", "abstract": "Использование языковых моделей для автоматического анализа тональности отзывов студентов о качестве учебного процесса."}
}

# ИЗМЕНЕНИЕ: Теперь главная страница возвращает красивый HTML-интерфейс!
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Эндпоинт генерации векторов (теперь вызывается автоматически интерфейсом)
@app.post("/init-database")
def init_database():
    # Импортируем нужные типы для квантования прямо тут, чтобы не лезть наверх файла
    from qdrant_client.models import QuantizationConfig, ScalarQuantization, ScalarType

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE), # ваш размер 768
        # ДОБАВЛЯЕМ СЮДА: сжатие векторов в 4 раза прямо при пересоздании коллекции
        quantization_config=QuantizationConfig(
            scalar=ScalarQuantization(
                type=ScalarType.INT8,
                always_ram=True  # ускоряет косинусный поиск в оперативной памяти
            )
        )
    )
    points = []
    for art_id, info in ARTICLES_DB.items():
        vector = model.encode(info["abstract"]).tolist()
        points.append(PointStruct(
            id=art_id, vector=vector,
            payload={"title": info["title"], "abstract": info["abstract"]}
        ))
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "success", "inserted_articles": len(points)}

# Эндпоинт самого семантического поиска
@app.get("/search")
def search_articles(query: str = Query(..., description="Поисковый запрос")):
    query_vector = model.encode(query).tolist()
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )
    formatted_results = []
    for hit in search_results:
        formatted_results.append({
            "article_id": hit.id,
            "score": hit.score,
            "title": hit.payload["title"],
            "abstract": hit.payload["abstract"]
        })
    return {"results": formatted_results}
        # НОВЫЙ ЭНДПОИНТ: Генерация ответа ИИ на основе найденного контекста (RAG)
@app.post("/ask")
async def ask_ai(query: str, context: str):
    # Формируем жесткий системный промпт (инструкцию) для Qwen
    prompt = f"""Если в тексте нет прямого ответа, ответь: "В предоставленных материалах нет ответа на этот вопрос".
Не придумывай ничего от себя.Ты — строгий научный ассистент библиотеки ВУЗа.
Используя ТОЛЬКО предоставленный текст научной статьи, четко и кратко ответь на вопрос. 


ТЕКСТ СТАТЬИ:
{context}

ВОПРОС СТУДЕНТА:
{query}

ОТВЕТ:"""

    # Данные для отправки в вашу локальную Qwen2.5:1.5b
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,  # Получить весь ответ сразу, а не по буквам
        "options": {
            "num_ctx": 4096,  # Фиксируем контекст для аннотаций
            "temperature": 0.0
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_URL, json=payload, timeout=60.0)
            if response.status_code == 200:
                return {"answer": response.json().get("response", "")}
            return {"answer": "Ошибка: Локальный ИИ-сервер вернул сбой."}
    except Exception as e:
        return {"answer": "ИИ-ассистент спит. Убедитесь, что программа Ollama запущена на вашем ПК."}

