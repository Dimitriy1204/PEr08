"""
Модуль для работы с эмбеддингами и векторным хранилищем ChromaDB.

Использует Yandex Embeddings API для создания эмбеддингов
и ChromaDB для их хранения и семантического поиска.

Фильтрация:
- Предварительный фильтр ChromaDB по метаданным `source` ДО семантического поиска
- Автоопределение темы фильтра через GigaChat-2 перед запросом
- Управление: команда `filter` → вкл, `filter_off` → выкл
"""

import chromadb
from chromadb.config import Settings
from typing import List, Tuple, Optional, Dict, Any
import os
import requests


class EmbeddingStore:
    """
    Класс для работы с векторным хранилищем ChromaDB.
    
    Использует Yandex Embeddings API для создания эмбеддингов
    и ChromaDB для их хранения и поиска.
    Поддерживает предварительный фильтр по source через where-условие ChromaDB.
    """
    
    def __init__(
        self, 
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        embed_url: Optional[str] = None
    ):
        """
        Инициализация хранилища эмбеддингов.
        
        Args:
            collection_name: Имя коллекции в ChromaDB
            persist_directory: Директория для сохранения данных ChromaDB
            embedding_model: Название модели Yandex для эмбеддингов (model_uri)
            api_key: Yandex API ключ
            folder_id: Yandex Folder ID
            embed_url: URL для API эмбеддингов Yandex
        """
        print(f"Инициализация ChromaDB в директории: {persist_directory}")
        
        # Создаем клиент ChromaDB с персистентным хранилищем
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Настройки Yandex Embeddings API
        self.api_key = api_key or os.getenv("YANDEX_API_KEY", "")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID", "")
        self.embed_url = embed_url or os.getenv(
            "YANDEX_EMBED_URL",
            "https://ai.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        )
        
        # Формируем model_uri для Yandex
        raw_model = embedding_model or os.getenv("EMBEDDING_MODEL", "")
        if raw_model and "<YANDEX_FOLDER_ID>" in raw_model:
            raw_model = raw_model.replace("<YANDEX_FOLDER_ID>", self.folder_id)
        
        if raw_model:
            self.embedding_model = raw_model
        else:
            self.embedding_model = f"emb://{self.folder_id}/text-embeddings/latest"
        
        if not self.api_key:
            print("⚠️  ВНИМАНИЕ: Не найден YANDEX_API_KEY!")
        if not self.folder_id:
            print("⚠️  ВНИМАНИЕ: Не найден YANDEX_FOLDER_ID!")
        
        print(f"Модель эмбеддингов: {self.embedding_model} (Yandex API)")
        
        # Получаем или СОЗДАЁМ коллекцию с принудительной cosine-метрикой.
        # get_or_create_collection НЕ применяет metadata к существующей коллекции,
        # поэтому если коллекция была создана с L2 (по умолчанию) — cosine не включится.
        # 
        # Стратегия:
        # 1. Пытаемся получить существующую
        # 2. Проверяем её метрику через count()
        # 3. Если метрика не cosine — удаляем и создаём заново
        try:
            existing = self.client.get_collection(name=collection_name)
            # Проверяем, что это наша коллекция. Если да — используем как есть.
            # Если она была создана без cosine — удаляем и пересоздаём
            self.collection = existing
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Документы для RAG-ассистента",
                    "hnsw:space": "cosine"  # Принудительно cosine, а не L2
                }
            )
            print(f"Создана новая коллекция '{collection_name}' с cosine-метрикой")
        
        print(f"✓ ChromaDB инициализирована. Документов в коллекции: {self.collection.count()}")
    
    def _create_chunks(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Разбивает текст на чанки (фрагменты) с перекрытием.
        
        Размер чанка и перекрытие берутся из .env (CHUNK_SIZE, CHUNK_OVERLAP),
        если не переданы явно. По умолчанию: chunk_size=500, overlap=50.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в символах (если None — читает из .env)
            overlap: Размер перекрытия между чанками (если None — читает из .env)
            
        Returns:
            Список чанков текста
        """
        # Если параметры не переданы — читаем из .env, иначе — значения по умолчанию
        if chunk_size is None:
            chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        if overlap is None:
            overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для одного текста используя Yandex Embeddings API.
        
        Args:
            text: Текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        if not text or not text.strip():
            return []
        
        payload = {
            "modelUri": self.embedding_model,
            "text": text
        }
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            response = requests.post(
                self.embed_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if not response.ok:
                error_text = response.text[:500]
                raise requests.HTTPError(
                    f"{response.status_code} Error from Yandex embeddings API: {error_text}",
                    response=response,
                )
            
            data = response.json()
            return data.get("embedding", [])
            
        except Exception as e:
            print(f"❌ Ошибка при создании эмбеддинга: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Добавляет документы в векторное хранилище.
        
        Каждый документ сохраняется с метаданными:
        - source: имя файла-источника (используется для фильтрации)
        - chunk_length: длина чанка в символах
        
        Args:
            documents: Список кортежей (название_документа, текст_документа)
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = self.collection.count()
        
        print(f"\nДобавление {len(documents)} документов в ChromaDB...")
        
        for doc_name, doc_text in documents:
            chunks = self._create_chunks(doc_text)
            print(f"  • {doc_name}: {len(chunks)} чанков")
            
            for chunk in chunks:
                all_chunks.append(chunk)
                # source = имя файла (например "PEr01_common_info.txt" или "PEr01_FAQ.txt")
                all_metadatas.append({
                    "source": doc_name,
                    "chunk_length": len(chunk)
                })
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        
        # Создаем эмбеддинги через Yandex API
        print(f"\nСоздание эмбеддингов для {len(all_chunks)} чанков через Yandex API...")
        print(f"(Модель: {self.embedding_model})")
        
        all_embeddings = []
        for i, chunk in enumerate(all_chunks):
            print(f"  Обработка чанка {i+1} из {len(all_chunks)}...")
            embedding = self._create_embedding(chunk)
            all_embeddings.append(embedding)
        
        # Добавляем все данные в ChromaDB
        print("Сохранение в ChromaDB...")
        self.collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"✓ Добавлено {len(all_chunks)} чанков. Всего в базе: {self.collection.count()}")
    
    def get_available_sources(self) -> List[str]:
        """
        Получает список всех уникальных источников документов в коллекции.
        
        Returns:
            Список названий источников (source) в коллекции
        """
        if self.collection.count() == 0:
            return []
        
        # Получаем все метаданные
        results = self.collection.get()
        sources = set()
        if results and results.get('metadatas'):
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
        
        return sorted(list(sources))
    
    def search(self, query: str, top_k: int = 7, filter_source: str = None) -> List[Tuple[str, str, float]]:
        """
        Выполняет семантический поиск по векторному хранилищу.
        
        Предварительный фильтр ChromaDB по метаданным `source` ДО семантического поиска.
        Это эффективнее пост-фильтрации, так как ChromaDB применяет where-условие
        на этапе поиска, уменьшая количество сравниваемых векторов.
        
        Args:
            query: Поисковый запрос пользователя
            top_k: Количество результатов для возврата
            filter_source: Фильтр по источнику документа (например, "PEr01_FAQ.txt").
                          Если None — поиск по всем документам без фильтра.
            
        Returns:
            Список кортежей (текст_чанка, источник, расстояние)
        """
        if self.collection.count() == 0:
            print("⚠ Предупреждение: коллекция пуста, нет документов для поиска")
            return []
        
        # Создаем эмбеддинг для запроса через Yandex API
        query_embedding = self._create_embedding(query)
        
        # Формируем where-фильтр для ChromaDB (если указан источник)
        # ChromaDB применяет фильтр ДО семантического поиска на уровне индекса,
        # что эффективнее, чем фильтрация результатов после поиска
        where_filter = None
        if filter_source:
            where_filter = {"source": filter_source}
            print(f"🔍 Предварительный фильтр ChromaDB по source: '{filter_source}'")
        
        # Выполняем поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter  # Фильтр применяется ДО поиска (не после)
        )
        
        # Форматируем результаты
        formatted_results = []
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunk_text = results['documents'][0][i]
                source = results['metadatas'][0][i]['source']
                distance = results['distances'][0][i] if results.get('distances') else 0.0
                formatted_results.append((chunk_text, source, distance))
        
        return formatted_results
    
    def clear_collection(self) -> None:
        """
        Очищает коллекцию (удаляет все документы) и пересоздаёт с cosine-метрикой.
        """
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={
                "description": "Документы для RAG-ассистента",
                "hnsw:space": "cosine"  # Принудительно cosine
            }
        )
        print("✓ Коллекция очищена и пересоздана с cosine-метрикой")


def load_documents_from_data_dir(data_dir: str = "DATA") -> List[Tuple[str, str]]:
    """
    Загружает документы из указанной директории.
    
    Каждый текстовый файл в директории становится отдельным документом.
    Имя файла (без пути) используется как source для метаданных.
    
    Args:
        data_dir: Путь к директории с файлами базы знаний
        
    Returns:
        Список кортежей (имя_файла, текст_документа)
    """
    import glob
    from pathlib import Path
    
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"❌ Директория {data_dir} не найдена")
        return []
    
    # Ищем все .txt файлы в директории DATA
    txt_files = sorted(data_path.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠ В директории {data_dir} не найдено .txt файлов")
        return []
    
    documents = []
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                doc_name = file_path.name  # Только имя файла, без пути
                documents.append((doc_name, text))
                print(f"  • Загружен: {doc_name} ({len(text)} символов)")
        except Exception as e:
            print(f"  ⚠ Ошибка загрузки {file_path}: {e}")
    
    return documents
