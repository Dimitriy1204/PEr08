"""
Главный файл для запуска RAG-ассистента — консолидированная версия.

Использует Yandex Embeddings API для эмбеддингов и GigaChat для генерации ответов.
Работает с корневой папкой /DATA как единой базой знаний.

Фильтрация управляется через SearchConfig (search_config.py):
  filter <название>  — ручной фильтр по источнику документа
  filter_off         — отключить все фильтры (поиск по всем документам)
  auto_filter on     — включить авто-определение темы через GigaChat
  auto_filter off    — отключить авто-определение темы

Замер времени:
  Используется time.monotonic для точного измерения времени выполнения запроса
  (включая поиск + генерацию), отображается в миллисекундах.
"""

import os
import time
from dotenv import load_dotenv
from embeddings import EmbeddingStore, load_documents_from_data_dir
from rag import RAGAssistant
from cache import ResponseCache
from search_config import SearchConfig


def initialize_system():
    """
    Инициализирует все компоненты RAG-системы.
    
    Returns:
        Кортеж (embedding_store, rag_assistant, cache, search_config)
    """
    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")
    print("   (Яндекс Эмбеддинги + GigaChat-2)")
    print("   База знаний: /DATA")
    print("=" * 70)
    
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие необходимых API ключей
    yandex_api_key = os.getenv("YANDEX_API_KEY")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")
    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS")
    data_dir = os.getenv("DATA_DIR", "DATA")
    
    if not yandex_api_key:
        print("⚠️  ВНИМАНИЕ: Не найден YANDEX_API_KEY в переменных окружения!")
        print("   Создайте файл .env и добавьте туда: YANDEX_API_KEY=your_key_here")
        print()
    
    if not yandex_folder_id:
        print("⚠️  ВНИМАНИЕ: Не найден YANDEX_FOLDER_ID в переменных окружения!")
        print("   Создайте файл .env и добавьте туда: YANDEX_FOLDER_ID=your_folder_id")
        print()
    
    if not gigachat_credentials:
        print("⚠️  ВНИМАНИЕ: Не найдены GIGACHAT_CREDENTIALS в переменных окружения!")
        print("   Создайте файл .env и добавьте туда: GIGACHAT_CREDENTIALS=your_credentials")
        print()
    
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    cache_file = os.getenv("CACHE_FILE", "cache.json")
    
    # 1. Инициализируем кеш для хранения ответов
    print("\n[1/3] Инициализация кеша...")
    cache = ResponseCache(cache_file=cache_file)
    
    # 2. Инициализируем векторное хранилище ChromaDB с Yandex Embeddings
    print("\n[2/3] Инициализация векторного хранилища...")
    embedding_model = os.getenv("EMBEDDING_MODEL", "emb://b1gecb6cuo1283cf5gl2/text-embeddings/latest")
    
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        api_key=yandex_api_key,
        folder_id=yandex_folder_id
    )
    
    # Загружаем документы из корневой папки /DATA, если коллекция пуста
    if embedding_store.collection.count() == 0:
        print(f"\n📝 База данных пуста. Загружаем документы из {data_dir}/...")
        documents = load_documents_from_data_dir(data_dir)
        if documents:
            embedding_store.add_documents(documents)
        else:
            print("⚠  Не удалось загрузить документы")
    else:
        print(f"✓ В базе уже есть {embedding_store.collection.count()} документов")
    
    # 3. Инициализируем RAG-ассистента с GigaChat
    print("\n[3/3] Инициализация RAG-ассистента...")
    model = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
    temperature = float(os.getenv("GIGACHAT_TEMPERATURE", "0.7"))
    
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        credentials=gigachat_credentials,
        model=model,
        temperature=temperature
    )
    
    # 4. Инициализируем SearchConfig с доступными источниками
    print("\n[4/4] Инициализация конфигурации поиска...")
    available_sources = embedding_store.get_available_sources()
    search_config = SearchConfig(available_sources=available_sources)
    
    print("\n" + "=" * 70)
    print("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
    print("=" * 70)
    
    return embedding_store, rag_assistant, cache, search_config


def format_time(seconds: float) -> str:
    """
    Форматирует время в человекочитаемый вид.
    
    Использует миллисекунды для быстрых операций (< 1 сек)
    и секунды с 3 знаками для медленных.
    
    Args:
        seconds: время в секундах (из time.monotonic)
    
    Returns:
        строка вида "1.234 сек" или "123.4 мс"
    """
    if seconds < 1.0:
        # Миллисекунды для быстрых ответов (кеш, простые операции)
        ms = seconds * 1000
        return f"{ms:.1f} мс"
    else:
        return f"{seconds:.3f} сек"


def answer_question(query: str, rag_assistant: RAGAssistant, cache: ResponseCache, 
                    search_config: SearchConfig = None,
                    filter_source: str = None, auto_filter: bool = False) -> str:
    """
    Отвечает на вопрос пользователя с использованием кеша и RAG.
    Замеряет время выполнения через time.monotonic.
    
    Логика работы:
    1. Проверяем кеш — если ответ есть, возвращаем его
    2. Если ответа нет, выполняем RAG (поиск + генерация)
    3. Сохраняем новый ответ в кеш
    4. Выводим время выполнения
    
    Args:
        query: Вопрос пользователя
        rag_assistant: Экземпляр RAG-ассистента
        cache: Экземпляр кеша
        search_config: Экземпляр SearchConfig для управления фильтрацией.
                       Если передан — filter_source и auto_filter игнорируются.
        filter_source: Ручной фильтр (используется только без search_config)
        auto_filter: Авто-фильтр (используется только без search_config)
        
    Returns:
        Ответ на вопрос
    """
    # Определяем метку режима для отображения
    if search_config is not None:
        mode_label = f" {search_config.get_status_string()}" if search_config.get_status_string() else ""
    else:
        mode_label = " [🤖авто]" if auto_filter else (f" [фильтр: {filter_source}]" if filter_source else "")
    
    print("\n" + "=" * 70)
    print(f"❓ ВОПРОС: {query}{mode_label}")
    print("=" * 70)
    
    # Старт замера времени
    start_time = time.monotonic()
    
    # Шаг 1: Проверяем кеш
    print("\n[Шаг 1] Проверка кеша...")
    cached_answer = cache.get(query)
    
    if cached_answer:
        elapsed = time.monotonic() - start_time
        print(f"\n⏱  Время (кеш): {format_time(elapsed)}")
        print("\n💾 Ответ из кеша:")
        print("-" * 70)
        print(cached_answer)
        print("-" * 70)
        return cached_answer
    
    # Шаг 2: Ответа нет в кеше — выполняем RAG
    print("\n[Шаг 2] Выполнение RAG (поиск + генерация)...")
    
    try:
        top_k = int(os.getenv("TOP_K", "3"))
        
        # Если передан search_config — используем его, иначе старый механизм
        if search_config is not None:
            answer, search_results = rag_assistant.generate_response(
                query=query,
                top_k=top_k,
                verbose=True,
                search_config=search_config
            )
        else:
            answer, search_results = rag_assistant.generate_response(
                query=query,
                top_k=top_k,
                verbose=True,
                filter_source=filter_source,
                auto_filter=auto_filter
            )
        
        # Замер времени после генерации
        elapsed = time.monotonic() - start_time
        
        # Шаг 3: Сохраняем ответ в кеш
        print("\n[Шаг 3] Сохранение ответа в кеш...")
        cache.set(query, answer)
        
        # Вывод времени выполнения
        print(f"\n⏱  Время выполнения: {format_time(elapsed)}")
        
        print("\n💡 ОТВЕТ:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        return answer
        
    except Exception as e:
        elapsed = time.monotonic() - start_time
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        print(f"\n⏱  Время до ошибки: {format_time(elapsed)}")
        print(f"\n❌ {error_msg}")
        return error_msg


def interactive_mode(rag_assistant: RAGAssistant, cache: ResponseCache, search_config: SearchConfig):
    """
    Интерактивный режим общения с ассистентом.
    
    Фильтрация управляется централизованно через SearchConfig.
    
    Поддерживает:
    - Ручной фильтр по источнику (filter <название>)
    - Автоматическое определение темы через GigaChat (auto_filter on)
    - Поиск по всем документам (filter_off)
    """
    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nВы можете задавать вопросы ассистенту.")
    print("Для выхода введите: exit, quit, выход или q")
    print("\nДоступные команды:")
    print("  • cache - показать информацию о кеше")
    print("  • clear_cache - очистить кеш")
    print("  • stats - показать статистику системы")
    print("  • filter <название> - установить фильтр по источнику документа")
    print("  • filter_off - отключить все фильтры (поиск по всем документам)")
    print("  • auto_filter on - включить авто-определение темы через GigaChat")
    print("  • auto_filter off - отключить авто-определение темы")
    print()
    
    while True:
        try:
            # Показываем статус фильтров из SearchConfig в приглашении
            status_str = search_config.get_status_string()
            prompt_prefix = f"{status_str} " if status_str else ""
            user_input = input(f"\n👤 {prompt_prefix}Вы: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход', 'q', '']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'cache':
                print(f"\n📊 Кеш содержит {cache.size()} записей")
                continue
            
            if user_input.lower() == 'clear_cache':
                cache.clear()
                print("\n✓ Кеш очищен")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n📊 СТАТИСТИКА СИСТЕМЫ:")
                print(f"  • Документов в ChromaDB: {rag_assistant.embedding_store.collection.count()}")
                print(f"  • Записей в кеше: {cache.size()}")
                print(f"  • Модель LLM: {rag_assistant.model}")
                print(search_config.get_status_report())
                continue
            
            # Команда filter_off — через SearchConfig.disable_all()
            if user_input.lower() == 'filter_off':
                msg = search_config.disable_all()
                print(f"\n{msg}")
                continue
            
            # Команда auto_filter on/off — через SearchConfig.set_auto_filter()
            if user_input.lower().startswith('auto_filter '):
                auto_value = user_input[12:].strip().lower()
                if auto_value in ('on', 'вкл', 'true', '1', 'yes', 'да'):
                    msg = search_config.set_auto_filter(True)
                    print(f"\n{msg}")
                elif auto_value in ('off', 'выкл', 'false', '0', 'no', 'нет'):
                    msg = search_config.set_auto_filter(False)
                    print(f"\n{msg}")
                else:
                    print("\n⚠ Используйте: auto_filter on или auto_filter off")
                continue
            
            # Команда filter <название_источника> — через SearchConfig.set_filter()
            if user_input.lower().startswith('filter '):
                filter_value = user_input[7:].strip()
                # Очищаем от возможных угловых скобок (пользователь мог ввести <имя>)
                filter_value = filter_value.strip('<>[]()"\'')
                if filter_value:
                    # Показываем доступные источники
                    sources = search_config.available_sources
                    if sources:
                        print(f"\n📚 Доступные источники: {', '.join(sources)}")
                    msg = search_config.set_filter(filter_value)
                    print(f"{msg}")
                    print(f"  Теперь все вопросы будут искаться ТОЛЬКО в этом документе.")
                else:
                    sources = search_config.available_sources
                    if sources:
                        print(f"\n⚠ Укажите название источника. Доступные: {', '.join(sources)}")
                    else:
                        print("\n⚠ Укажите название источника после filter, например: filter PEr01_FAQ.txt")
                continue
            
            # Обрабатываем вопрос пользователя — передаём SearchConfig
            answer_question(user_input, rag_assistant, cache, search_config=search_config)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")


def demo_mode(rag_assistant: RAGAssistant, cache: ResponseCache, search_config: SearchConfig):
    """
    Демонстрационный режим с заранее заготовленными вопросами.
    """
    print("\n" + "=" * 70)
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nСейчас будет продемонстрирована работа RAG-ассистента")
    print("на нескольких примерах вопросов.\n")
    
    demo_questions = [
        "Что такое ЧПУ и как оно работает на токарном станке?",
        "Где находится учебный центр Точка отсчета?",
        "Какие услуги оказывает Завод Стройтехника?",
        "Что такое ЧПУ и как оно работает на токарном станке?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"ВОПРОС {i} из {len(demo_questions)}")
        print(f"{'#' * 70}")
        
        answer_question(question, rag_assistant, cache, search_config=search_config)
        
        if i < len(demo_questions):
            user_input = input("\n[Нажмите Enter для следующего вопроса... или введите exit/q для выхода]: ").strip().lower()
            if user_input in ('exit', 'quit', 'выход', 'q'):
                print("\n⏹ Демонстрация прервана пользователем.")
                break
    
    print("\n\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """
    Главная функция приложения.
    """
    try:
        embedding_store, rag_assistant, cache, search_config = initialize_system()
        
        print("\n" + "=" * 70)
        print("ВЫБОР РЕЖИМА РАБОТЫ")
        print("=" * 70)
        print("\n1. Интерактивный режим - задавайте свои вопросы")
        print("2. Демонстрационный режим - готовые примеры вопросов")
        print()
        
        mode = input("Выберите режим (1 или 2, по умолчанию 1): ").strip()
        
        if mode == '2':
            demo_mode(rag_assistant, cache, search_config)
            
            print("\n" + "=" * 70)
            continue_interactive = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
            if continue_interactive in ['y', 'yes', 'д', 'да', '']:
                interactive_mode(rag_assistant, cache, search_config)
        else:
            interactive_mode(rag_assistant, cache, search_config)
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()