"""
Модуль конфигурации поиска — централизованное управление фильтрацией.

Содержит класс SearchConfig, который инкапсулирует:
- Ручной фильтр по источнику документа (filter <name>)
- Автоматическое определение темы через GigaChat (auto_filter on/off)
- Разрешение финального источника для поиска (приоритет: ручной > авто > None)
- Определение темы вопроса через LLM

Зачем отдельный класс:
- Логика фильтрации не размазана по rag.py и app.py
- Можно легко добавить новые команды/режимы (например, exclude, regex, etc.)
- Единая точка изменения для всех правил фильтрации
"""

from typing import Optional, List
import os


class SearchConfig:
    """
    Конфигурация поиска с поддержкой фильтрации по источнику.
    
    Состояние:
    - active_filter: имя файла-источника для ручного фильтра (None = отключён)
    - auto_filter: True — GigaChat сам определяет тему вопроса
    - available_sources: список доступных источников из ChromaDB
    
    Приоритет при разрешении финального источника:
    1. Ручной фильтр (если active_filter задан)
    2. Авто-определение (если auto_filter=True)
    3. None (поиск по всем документам)
    """
    
    def __init__(self, available_sources: Optional[List[str]] = None):
        """
        Args:
            available_sources: список имён источников (файлов) из ChromaDB
        """
        self.active_filter: Optional[str] = None
        self.auto_filter: bool = False
        self.available_sources: List[str] = available_sources or []
    
    def set_filter(self, source_name: str) -> str:
        """
        Устанавливает ручной фильтр по имени источника.
        Авто-фильтр при этом сбрасывается (приоритет ручного).
        
        Args:
            source_name: имя файла-источника (например "PEr01_FAQ.txt")
            
        Returns:
            Сообщение для пользователя о результате
        """
        self.active_filter = source_name
        self.auto_filter = False
        return f"✓ Установлен фильтр по источнику: '{self.active_filter}'"
    
    def set_auto_filter(self, enabled: bool) -> str:
        """
        Включает/отключает авто-определение темы через GigaChat.
        
        Args:
            enabled: True — включить, False — выключить
            
        Returns:
            Сообщение для пользователя о результате
        """
        if enabled:
            self.active_filter = None  # Авто сбрасывает ручной
            self.auto_filter = True
            return ("✓ Включён авто-фильтр: GigaChat будет сам определять тему вопроса "
                    "и выбирать соответствующий источник документов.")
        else:
            self.auto_filter = False
            return "✓ Авто-фильтр отключён."
    
    def disable_all(self) -> str:
        """
        Отключает все фильтры — и ручной, и авто.
        
        Returns:
            Сообщение для пользователя о результате
        """
        self.active_filter = None
        self.auto_filter = False
        return "✓ Все фильтры отключены. Поиск по ВСЕМ документам."
    
    def resolve_source(self, query: str, detect_topic_fn=None, verbose: bool = True) -> Optional[str]:
        """
        Определяет финальный источник для поиска.
        
        Приоритет:
        1. active_filter (ручной) — если задан, возвращается он
        2. auto_filter=True — вызывается detect_topic_fn для определения темы
        3. Иначе — None (поиск по всем)
        
        Args:
            query: вопрос пользователя (нужен для авто-определения)
            detect_topic_fn: функция для определения темы через GigaChat
                             Сигнатура: (query, verbose) -> Optional[str]
            verbose: выводить ли отладочную информацию
            
        Returns:
            имя источника для фильтрации или None
        """
        # Приоритет 1: ручной фильтр
        if self.active_filter is not None:
            return self.active_filter
        
        # Приоритет 2: авто-определение через GigaChat
        if self.auto_filter and detect_topic_fn is not None:
            return detect_topic_fn(query, verbose=verbose)
        
        # Приоритет 3: поиск по всем
        return None
    
    def get_status_string(self) -> str:
        """
        Возвращает строковое представление текущего состояния фильтров
        для отображения в приглашении (prompt).
        """
        parts = []
        if self.active_filter:
            parts.append(f"📄{self.active_filter}")
        if self.auto_filter:
            parts.append("🤖авто")
        return f"[{' | '.join(parts)}]" if parts else ""
    
    def get_status_report(self) -> str:
        """
        Возвращает многострочный отчёт о состоянии фильтров.
        """
        lines = []
        if self.active_filter:
            lines.append(f"  • Ручной фильтр: {self.active_filter}")
        else:
            lines.append(f"  • Ручной фильтр: не установлен")
        
        if self.auto_filter:
            lines.append(f"  • Авто-фильтр: включён (GigaChat определяет тему)")
        else:
            lines.append(f"  • Авто-фильтр: отключён")
        
        return "\n".join(lines)
    
    def update_sources(self, sources: List[str]) -> None:
        """
        Обновляет список доступных источников (например, после перезагрузки данных).
        
        Args:
            sources: новый список имён источников
        """
        self.available_sources = sources