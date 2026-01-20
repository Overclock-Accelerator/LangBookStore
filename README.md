# LangBookStore / RightBookAI

RightBookAI is a small Python codebase that answers questions about a local bookstore inventory (`storedata.json`), recommends books, and builds budget-friendly book bundles.

## What's in this repo

- **Inventory**: `storedata.json` (the book catalog)
- **CLI agent**: `rightbookai_agent.py`
  - Routes a user question to one of three tools (heuristic router)
  - Can also run a pure LLM mode (not required for the tools)
- **Tools** (LangChain tools):
  - `tools/get_answers.py` — title-based Q&A (pages, price, year, etc.)
  - `tools/recommend_books.py` — recommendations from preferences (genre, sale, rating, year, pages, etc.)
  - `tools/budget_bundler.py` — builds a bundle under a budget using a knapsack-style optimizer
  - `tools/storedata_utils.py` — shared helpers for loading/normalizing inventory
- **Inventory summary**: `BOOKSTORE_SUMMARY.md`

## Prerequisites

- **Python**: 3.10+ recommended
- **OpenAI API key**: required for the optional LLM-only mode in `rightbookai_agent.py` (the tool-based routing works without calling OpenAI)

## Setup (Windows / macOS / Linux)

### 1) Create and activate a virtual environment

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## OpenAI API key setup (recommended)

Create a file named `.env.local` in the project root:

```bash
OPENAI_API_KEY=your_key_here
RIGHTBOOKAI_MODEL=gpt-4.1-mini
```

Notes:
- **`OPENAI_API_KEY`** is required only for `rightbookai_answer_via_llm` (LLM-only mode).
- **`RIGHTBOOKAI_MODEL`** is optional. If you omit it, the code uses its default model name.
- `.env.local` is loaded automatically by `rightbookai_agent.py`.

## Running RightBookAI (CLI)

### Interactive mode (REPL)

```bash
python rightbookai_agent.py
```

### One-shot question

```bash
python rightbookai_agent.py "How many pages is Heart the Lover?"
```

### Pipe input

```bash
echo "Recommend two post-apocalyptic books" | python rightbookai_agent.py
```

## Using the tools directly (developer workflow)

You can also invoke the tools directly from Python (useful for debugging).

### GetAnswers

```bash
python -c "from tools.get_answers import get_answers; print(get_answers.invoke({'query':'How many pages is Heart the Lover?'}))"
```

### RecommendBooks

```bash
python -c "from tools.recommend_books import recommend_books; print(recommend_books.invoke({'user_request':'I like science fiction and post apocalyptic novels. Suggest two different books.'}))"
```

### BudgetBundler

```bash
python -c \"from tools.budget_bundler import budget_bundler; print(budget_bundler.invoke({'budget_request':'I have a budget of $65. Create a suggested order that is a mix of recent Science Fiction, Fantasy, and History.'}))\"
```

## How routing works

`rightbookai_agent.py` uses a simple heuristic router:
- Budget-ish queries → **BudgetBundler**
- “recommend/suggest/next read” queries → **RecommendBooks**
- Everything else → **GetAnswers**

The router is deliberately simple; the heavy lifting is done inside each tool.

