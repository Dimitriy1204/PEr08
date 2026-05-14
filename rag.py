"""
Модуль для реализации RAG (Retrieval-Augmented Generation).

RAG объединяет поиск релевантной информации (Retrieval) с генерацией ответа (Generation)
для создания более точных и информативных ответов на вопросы пользователя.

Использует GigaChat от Сбера как LLM для генерации ответов.
Фильтрация по источнику вынесена в отдельный класс SearchConfig (search_config.py).
"""

from typing import List, Tuple, Optional
import os

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage


class RAGAssistant:
    """
    Класс RAG-ассистента, который использует векторный поиск и GigaChat для ответов.
    
    Процесс работы:
    1. Получает запрос пользователя + SearchConfig с настройками фильтрации
    2. Определяет финальный источник через SearchConfig.resolve_source()
    3. Ищет релевантные документы в векторной базе (с фильтром или без)
    4. Формирует контекст из найденных документов
    5. Отправляет запрос + контекст в GigaChat
    6. Возвращает сгенерированный ответ
    """
    
    # Описание содержания каждого файла базы знаний для авто-фильтрации.
    # Ключ = имя файла, значение = краткое описание тематики.
    # GigaChat использует эти описания, чтобы понять, к какому файлу относится вопрос.
    SOURCE_DESCRIPTIONS = {
        "PEr01_common_info.txt": "Общая информация об учебном центре Точка отсчета и Заводе Стройтехника: адрес, контакты, услуги, продукция",
        "PEr01_FAQ.txt": "Вопросы и ответы по системе ЧПУ Fanuc, токарным станкам, программированию G-кодов (G00, G01, G71, G70 и др.)"
    }
    
    def __init__(
        self, 
        embedding_store,
        credentials: Optional[str] = None,
        model: str = "GigaChat-2",
        temperature: float = 0.7,
        profanity_check: bool = True,
        scope: str = "GIGACHAT_API_PERS"
    ):
        """
        Инициализация RAG-ассистента с GigaChat.
        
        Args:
            embedding_store: Экземпляр EmbeddingStore для поиска документов
            credentials: Учётные данные GigaChat (если None, берется из .env)
            model: Название модели GigaChat
            temperature: Параметр "креативности" модели (0.0 - детерминированный, 1.0 - креативный)
            profanity_check: Проверка ненормативной лексики
            scope: Область действия GigaChat API (PERS для физических лиц, CORP для юр. лиц)
        """
        self.embedding_store = embedding_store
        self.model = model
        self.temperature = temperature
        
        # Берём credentials из параметра или из переменной окружения
        gigachat_credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS", "")
        
        if not gigachat_credentials:
            print("⚠️  ВНИМАНИЕ: Не найдены GIGACHAT_CREDENTIALS!")
        
        # Инициализируем клиент GigaChat
        self.client = GigaChat(
            model=model,
            credentials=gigachat_credentials,
            scope=scope,
            verify_ssl_certs=False,
            profanity_check=profanity_check,
            temperature=temperature
        )
        
        # Загружаем список доступных источников из ChromaDB
        self.AVAILABLE_SOURCES = self.embedding_store.get_available_sources()
        
        # Применяем описания из SOURCE_DESCRIPTIONS для тех файлов, что есть в коллекции
        self._source_hints = {}
        for src in self.AVAILABLE_SOURCES:
            if src in self.SOURCE_DESCRIPTIONS:
                self._source_hints[src] = self.SOURCE_DESCRIPTIONS[src]
            else:
                self._source_hints[src] = f"Документ: {src}"
        
        print(f"✓ RAG-ассистент инициализирован (модель: {model})")
        print(f"  Доступные источники для авто-фильтрации: {', '.join(self.AVAILABLE_SOURCES) if self.AVAILABLE_SOURCES else '(нет)'}")
    
    def _detect_topic(self, query: str, verbose: bool = True) -> Optional[str]:
        """
        Определяет тему вопроса через GigaChat и выбирает соответствующий источник.
        
        Спрашивает у GigaChat, к какому из доступных источников относится вопрос.
        Если тема не определена однозначно — возвращает None (поиск по всем документам).
        
        Args:
            query: Вопрос пользователя
            verbose: Выводить ли информацию о процессе определения темы
            
        Returns:
            Название источника для фильтрации или None (поиск по всем)
        """
        if not self.AVAILABLE_SOURCES:
            return None
        
        if verbose:
            print(f"\n🧠 Определение темы вопроса через GigaChat...")
        
        # Формируем список тем с их описаниями, чтобы GigaChat мог понять,
        # о чём каждый файл, а не просто видел имя файла
        sources_list = "\n".join([
            f"- {src}: {self._source_hints.get(src, src)}" 
            for src in self.AVAILABLE_SOURCES
        ])
        
        topic_prompt = (
            f"Определи, к какой теме из списка ниже относится вопрос пользователя.\n\n"
            f"Доступные темы:\n{sources_list}\n\n"
            f"Правила:\n"
            f"1. Если вопрос явно относится к одной из тем — ответь ТОЛЬКО названием темы (строго одно слово из списка выше — имя файла)\n"
            f"2. Если вопрос относится к нескольким темам или не относится ни к одной — ответь: Unknown\n"
            f"3. Не добавляй никаких пояснений, кавычек, знаков препинания — только имя файла или Unknown\n\n"
            f"Вопрос пользователя: {query}\n\n"
            f"Ответ (только имя файла из списка или Unknown):"
        )
        
        try:
            messages = [
                SystemMessage(
                    content="Ты - классификатор тем. Отвечай строго одним словом — названием темы или Unknown."
                ),
                HumanMessage(content=topic_prompt)
            ]
            
            response = self.client.invoke(messages)
            
            if response and hasattr(response, 'content') and response.content:
                topic = response.content.strip().strip('"').strip("'").strip('.')
                
                # Проверяем, есть ли такое название среди доступных источников
                for source in self.AVAILABLE_SOURCES:
                    if topic.lower() == source.lower():
                        if verbose:
                            print(f"✅ Определена тема: '{source}'")
                        return source
                
                # Если GigaChat вернул что-то похожее на название, но не точное совпадение
                for source in self.AVAILABLE_SOURCES:
                    if topic.lower() in source.lower() or source.lower() in topic.lower():
                        if verbose:
                            print(f"✅ Определена тема (по совпадению): '{source}'")
                        return source
                
                if verbose:
                    print(f"ℹ️ Тема не определена однозначно ('{topic}'). Поиск по всем документам.")
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Ошибка при определении темы: {e}. Поиск по всем документам.")
        
        return None
    
    def _format_context(self, search_results: List[Tuple[str, str, float]]) -> str:
        """
        Форматирует результаты поиска в контекст для LLM.
        
        Args:
            search_results: Список результатов поиска (текст, источник, расстояние)
            
        Returns:
            Отформатированный текст контекста
        """
        if not search_results:
            return "Релевантных документов не найдено."
        
        context_parts = []
        
        for i, (chunk_text, source, distance) in enumerate(search_results, 1):
            context_parts.append(
                f"[Документ {i} - {source}]\n{chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для LLM, включающий контекст и запрос пользователя.
        
        Args:
            query: Запрос пользователя
            context: Контекст из найденных документов
            
        Returns:
            Сформированный промпт
        """
        prompt = f"""Ты - полезный AI-ассистент. Используй следующую информацию из базы знаний, чтобы ответить на вопрос пользователя.

ВАЖНО: 
- Отвечай на основе предоставленного контекста
- Если в контексте нет информации для ответа, честно скажи об этом
- Отвечай на русском языке
- Будь конкретным и информативным

=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ОТВЕТ ===
"""
        return prompt
    
    def generate_response(
        self, 
        query: str, 
        top_k: int = 3,
        verbose: bool = True,
        filter_source: str = None,
        auto_filter: bool = False,
        search_config=None
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        """
        Генерирует ответ на запрос пользователя используя RAG.
        
        Args:
            query: Запрос пользователя
            top_k: Количество документов для поиска
            verbose: Выводить ли детальную информацию о процессе
            filter_source: Ручной фильтр по источнику документа (например, "PEr01_FAQ.txt").
                          Используется только если search_config не передан.
            auto_filter: Если True — GigaChat сам определит тему вопроса.
                        Используется только если search_config не передан.
            search_config: Экземпляр SearchConfig для управления фильтрацией.
                          Если передан — filter_source и auto_filter игнорируются.
                          Приоритет: search_config > (filter_source, auto_filter) > None
            
        Returns:
            Кортеж (ответ_llm, список_найденных_документов)
        """
        # Шаг 0: Определяем фильтр через SearchConfig, если передан
        # Иначе — через старый механизм (filter_source / auto_filter) для обратной совместимости
        if search_config is not None:
            effective_filter = search_config.resolve_source(
                query, 
                detect_topic_fn=self._detect_topic,
                verbose=verbose
            )
        else:
            effective_filter = filter_source
            if effective_filter is None and auto_filter:
                effective_filter = self._detect_topic(query, verbose=verbose)
        
        if effective_filter and verbose and search_config is None:
            print(f"🔍 Автоматически выбран фильтр: '{effective_filter}'")
        
        # Шаг 1: Поиск релевантных документов в векторной базе
        if verbose:
            source_info = f" (фильтр: '{effective_filter}')" if effective_filter else ""
            print(f"\n🔍 Поиск релевантных документов (top_k={top_k}){source_info}...")
        
        search_results = self.embedding_store.search(query, top_k=top_k, filter_source=effective_filter)
        
        if verbose and search_results:
            print(f"\n📚 Найдено {len(search_results)} релевантных фрагментов (L2 distance: 0 = идеально, больше = хуже):")
            for i, (chunk, source, distance) in enumerate(search_results, 1):
                print(f"  {i}. [{source}] (dist: {distance:.3f})")
                print(f"     {chunk[:100]}...")
        
        # Шаг 2: Форматируем контекст из найденных документов
        context = self._format_context(search_results)
        
        # Шаг 3: Создаем промпт с контекстом и запросом
        prompt = self._create_prompt(query, context)
        
        # Шаг 4: Отправляем запрос в GigaChat
        if verbose:
            print(f"\n🤖 Генерация ответа с помощью {self.model}...")
        
        try:
            # Формируем сообщения для GigaChat
            messages = [
                SystemMessage(
                    content="Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста."
                ),
                HumanMessage(content=prompt)
            ]
            
            # Отправляем запрос
            response = self.client.invoke(messages)
            
            # Извлекаем текст ответа
            if response and hasattr(response, 'content') and response.content:
                answer = response.content.strip()
            else:
                answer = "Модель не вернула ответ."
            
            return answer, search_results
            
        except Exception as e:
            error_message = f"Ошибка при генерации ответа: {str(e)}"
            print(f"❌ {error_message}")
            return error_message, search_results
    
    def simple_response(self, query: str, auto_filter: bool = False) -> str:
        """
        Упрощенная версия generate_response, возвращающая только текст ответа.
        
        Args:
            query: Запрос пользователя
            auto_filter: Если True — GigaChat сам определит тему вопроса
            
        Returns:
            Ответ LLM
        """
        answer, _ = self.generate_response(query, verbose=False, auto_filter=auto_filter)
        return answer