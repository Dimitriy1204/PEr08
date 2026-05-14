"""
Оценка качества RAG системы через RAGAS.
Использует консолидированную архитектуру: Yandex Embeddings + GigaChat-2, корневая /DATA.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate

# Импорт метрик RAGAS
try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision

from rag import RAGAssistant
from embeddings import EmbeddingStore, load_documents_from_data_dir
from cache import ResponseCache


# Тестовые вопросы для оценки RAG системы — соответствуют данным из корневой /DATA
EVALUATION_QUESTIONS = [
    "Что такое ЧПУ и как оно работает на токарном станке?",
    "Где находится учебный центр Точка отсчета?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое RAG и как он работает?",
    "Как работать с циклом G71 для черновой обработки?"
]


def prepare_dataset(rag_assistant: RAGAssistant, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        rag_assistant: RAG ассистент для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        answer, search_results = rag_assistant.generate_response(
            query=question,
            top_k=3,
            verbose=False,
            auto_filter=False  # Поиск по всем документам для оценки
        )
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(answer)
        
        # Контекст - список текстов из найденных документов
        context_texts = [chunk for chunk, _, _ in search_results]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(answer[:100])
        
        print(f"     [+] Ответ получен от GigaChat")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline (Yandex Embeddings + GigaChat)
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (Yandex + GigaChat) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия ключей
    if not os.getenv("YANDEX_API_KEY"):
        print("[ОШИБКА] YANDEX_API_KEY не установлен")
        sys.exit(1)
    if not os.getenv("GIGACHAT_CREDENTIALS"):
        print("[ОШИБКА] GIGACHAT_CREDENTIALS не установлен")
        sys.exit(1)
    
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    data_dir = os.getenv("DATA_DIR", "DATA")
    
    # Инициализация компонентов
    try:
        print("[*] Инициализация системы (Yandex Embeddings + GigaChat)...\n")
        
        # Векторное хранилище
        embedding_store = EmbeddingStore(
            collection_name="rag_documents",
            persist_directory=persist_directory
        )
        
        # Загрузка документов из /DATA если пусто
        if embedding_store.collection.count() == 0:
            print(f"\n📝 Загрузка документов из {data_dir}/...")
            documents = load_documents_from_data_dir(data_dir)
            if documents:
                embedding_store.add_documents(documents)
        
        # RAG ассистент
        rag_assistant = RAGAssistant(
            embedding_store=embedding_store,
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
            temperature=0.3
        )
        
        print("\n[OK] Система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(rag_assistant, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (это займёт 1-2 минуты)\n")
    
    metrics_to_use = [faithfulness(), context_precision()]
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    import math
    
    faithfulness_values = [
        v for v in result['faithfulness'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v for v in result['context_precision'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) 
        if context_precision_values else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")
        
        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        cp_val = result['context_precision'][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()