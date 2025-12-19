# RAG Question Answering System

Система для відповідей на питання з використанням RAG

##  Опис проєкту

Це повнофункціональна RAG система, яка відповідає на питання на задану тему використовуючи Wikipedia як джерело знань

## Задача

Створення системи question answering з використанням підходу retrieval-augmented generation для відповідей на питання користувача на основі сторонніх документів

## Компоненти системи

### Джерело даних
- **Тип**: Wikipedia API
- **Статті**: 
      **Core AI**
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural networks",
    "Symbolic artificial intelligence",
    "Expert system",

    **NLP / LLM**
    "Natural language processing",
    "Large language model",
    "Language model",
    "Transformer (machine learning model)",
    "Attention mechanism",
    "Document classification",
    "Text summarization",
    "Speech recognition",
    "Question answering",

    **Generative AI**
    "Generative artificial intelligence",
    "Generative model",
    "Foundation model",
    "Diffusion model",
    "Generative adversarial network",
    "Variational autoencoder",
    "Prompt engineering",

    **Embeddings / Retrieval / RAG**
    "Embedding (machine learning)",
    "Vector database",
    "Semantic search",
    "Similarity search",
    "Information retrieval",
    "Retrieval-augmented generation",
    "Question answering system",

    **Agentic / Reasoning**
    "Agentic AI",
    "Intelligent agent",
    "Reinforcement learning",
    "Automated planning",
    "Multi-agent system",

    **Popular models / systems**
    "ChatGPT",
    "GPT-4",
    "BERT (language model)",
    "OpenAI",

    **Training / Evaluation**
    "Fine-tuning (deep learning)",
    "Transfer learning",
    "Explainable artificial intelligence",
    "AI alignment"
- **Формат**: Текст з метаданими (title, url)

### Chunking
- **Бібліотека**: LangChain RecursiveCharacterTextSplitter
- **Параметри**:
  - Розмір чанку: 500 символів
  - Overlap: 50 символів
  - Сепаратори: `\n\n`, `\n`, `. `, пробіл
- **Результат**: ~200-300 чанків залежно від розміру статей
 
### LLM
- **Провайдер**: Groq
- **Модель**: llama-3.3-70b-versatile
- **Параметри**:
  - Temperature: 0.3
  - Max tokens: 1000

### Retriever

#### 1. BM25 (Keyword Search)
- **Бібліотека**: rank-bm25
- **Тип**: Sparse retrieval

#### 2. Dense (Semantic Search)
- **Модель**: sentence-transformers/all-MiniLM-L6-v2
- **Тип**: Dense retrieval
- **Метрика**: Косинусна подібність
- **Приклади запитів**:
  - "What is machine learning?"
  - "How do computers learn from data?"
  - "Що таке ChatGPT?"

#### 3. Hybrid Search
- Комбінація BM25 + Dense
- Top-K кандидатів: 20

### Reranker
- **Модель**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Вхід**: До 20 кандидатів
- **Вихід**: Топ-5 найрелевантніших чанків

### Citations
- Автоматичне додавання посилань [1], [2], ... в тексті відповіді
- Список джерел з:
  - Назвою документа
  - URL посиланням
  - Фрагментом тексту
  - Reranker score
 

##  Metadata Filtering

Для кожного чанку зберігаються метадані в SQLite:

- document_type (article / academic / tutorial / technical)
- topic
- difficulty_level (beginner / intermediate / advanced)
- year_mentioned
- categories

Метадані:
- частково генеруються LLM
- частково витягуються автоматично
- використовуються для фільтрації кандидатів перед генерацією відповіді

Фільтри автоматично виводяться з тексту запиту користувача.
### UI
- **Фреймворк**: Gradio
- **Функціональність**:
  - Поле для запиту
  - Поле для Groq API ключа
  - Чекбокси для вмикання/вимикання методів пошуку
  - Слайдер для кількості джерел (1-10)
  - Приклади запитів
  - Debug інформація про процес пошуку

## Запуск

### Локально

1. **Клонуйте репозиторій**:
```bash
git clone https://github.com/nyakimova/RAG-project
cd RAG-project
cd rag_final
```

2. **Встановіть залежності**:
```bash
pip install -r requirements.txt
```

3. **Запустіть**:
```
python rag.py
```

4. **Отримайте API ключ**:
   - Йдіть на https://console.groq.com/keys
   - Створіть безкоштовний акаунт
   - Згенеруйте API ключ

5. **Використовуйте**:
   - Відкрийте браузер за адресою з терміналу
   - Вставте API ключ
   - Задайте питання

### Команда:
**Настя Якимів**

Архітектура RAG pipeline

Hybrid retrieval (BM25 + Dense)

Metadata schema та SQLite

Metadata filtering logic

Integration Groq LLM

**Елеонора Кречківська**

Chunking стратегія

Semantic embeddings

Reranker integration

Citations & source formatting

Gradio UI та debug-вивід


## Дякуємо за перевірку!
