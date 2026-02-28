from flask import Flask, request, jsonify
import os
import requests
from requests.exceptions import HTTPError
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
import re
from urllib.parse import urlparse
import base64
import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Настраиваем логирование: пишем в файл app.log и в консоль
logging.basicConfig(
    level=logging.DEBUG, # Уровень отлова сообщений
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        #logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler() # Вывод в консоль
    ]
)

logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Инициализируем LLM
llm = ChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    #model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    request_timeout=120.0, # Ждать максимум 120 секунд!
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

class ImportantFilesSchema(BaseModel):
    # Pydantic ожидает объект (словарь), поэтому мы оборачиваем список в ключ "files"
    files: list[str] = Field(description="JSON array of file paths, for example: ['src/main.py', 'config.yaml']")

class RepoSummarySchema(BaseModel):
    summary: str = Field(description="A clear, human-readable description of what this project does")
    technologies: list[str] = Field(description="List of main technologies, languages, frameworks")
    structure: str = Field(description="Brief description of the project structure and organization")

app = Flask(__name__)


def parse_github_url(url: str):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    # Нормализация SSH-формата: git@github.com:owner/repo -> https://github.com/owner/repo
    if re.match(r"^git@", url):
        url = re.sub(r"^git@([^:]+):", r"https://\1/", url)

    parsed = urlparse(url)
    
    # Разбиваем путь на сегменты, убирая пустые строки (от слешей)
    segments = [s for s in parsed.path.split("/") if s]

    if len(segments) < 2:
        raise ValueError(f"Cannot extract owner/repo from URL: {url}")

    owner = segments[0]
    # Убираем .git только в конце строки
    repo = re.sub(r"\.git$", "", segments[1])

    return owner, repo


def get_repo_tree(owner: str, repo: str, max_depth: int = 2) -> list[dict]:
    """
    Получает дерево файлов репозитория.
    Возвращает только blob-файлы (без директорий), прошедшие все фильтры.
    """
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    binary_extensions = (
        '.exe', '.dll', '.so', '.a', '.lib', '.dylib', '.o', '.obj',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf', '.doc', '.docx',
        '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.iso', '.db',
        '.sqlite', '.jar', '.class', '.pyc', '.whl', '.ds_store', '.svg',
        '.woff', '.woff2', '.ttf', '.eot', '.ico', '.lockb',
    )

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except HTTPError as e:
        if e.response.status_code == 409:
            logger.debug("[%s/%s] Репозиторий пустой (409 Conflict), возвращаем []", owner, repo)
            return []
        raise

    data = response.json()
    tree = data.get("tree", [])

    logger.debug(
        "[%s/%s] Получено записей от API: %d (truncated=%s)",
        owner, repo, len(tree), data.get("truncated", False)
    )

    if data.get("truncated"):
        warnings.warn(
            f"GitHub API вернул неполное дерево для {owner}/{repo} "
            f"(более 100 000 файлов или >7 MB). "
            f"Используйте нерекурсивный обход поддеревьев для полного результата.",
            RuntimeWarning,
            stacklevel=2,
        )

    filtered_tree = []
    excluded_tree = []  # (путь, причина)

    for item in tree:
        depth = item["path"].count("/")

        if depth >= max_depth:
            excluded_tree.append((item["path"], f"depth={depth} >= max_depth={max_depth}"))
            continue

        if item["type"] != "blob":
            excluded_tree.append((item["path"], f"type={item['type']}"))
            continue

        path_lower = item["path"].lower()
        if path_lower.endswith(binary_extensions) or path_lower.endswith("package-lock.json"):
            excluded_tree.append((item["path"], "binary/irrelevant extension"))
            continue

        filtered_tree.append(item)

    # --- Итоговый дамп ---
    logger.debug(
        "[%s/%s] Итого после фильтрации: %d файлов (исключено: %d)",
        owner, repo, len(filtered_tree), len(excluded_tree)
    )

    if excluded_tree:
        excluded_lines = "\n".join(f"  - {path}  [{reason}]" for path, reason in excluded_tree)
        logger.debug("[%s/%s] Исключённые записи:\n%s", owner, repo, excluded_lines)

    if filtered_tree:
        included_lines = "\n".join(f"  + {item['path']}" for item in filtered_tree)
        logger.debug("[%s/%s] Включённые файлы:\n%s", owner, repo, included_lines)

    return filtered_tree


def get_readme(owner, repo):
    """Получает содержимое README.md если есть"""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    
    for readme_name in ["README.md", "readme.md", "Readme.md", "README"]:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{readme_name}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            content = response.json().get("content", "")
            return base64.b64decode(content).decode("utf-8")
    
    return None


def get_file_content(owner, repo, file_path):
    """Получает содержимое конкретного файла"""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    content = response.json().get("content", "")
    return base64.b64decode(content).decode("utf-8")


def analyze_repo_structure(owner, repo, tree, readme):
    """Анализирует структуру и выбирает важные файлы"""
    tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in tree])
    
    # 1. Инициализируем парсер с нашей схемой
    parser = JsonOutputParser(pydantic_object=ImportantFilesSchema)
    
    # 2. Создаем шаблон, куда LangChain сам вставит требования к JSON
    prompt = PromptTemplate(
        template="""You are analyzing a GitHub repository structure.

Repository tree (depth up to 2):
{tree_text}

README content:
{readme}

Based on this structure and README, select up to 6 most valuable files that would help understand:
1. What this project does
2. How it works
3. Its architecture and main components

CRITICAL INSTRUCTIONS FOR FILE PATHS:
- You MUST copy the file paths EXACTLY as they appear in the "Repository tree" list above.
- DO NOT modify, shorten, or guess file paths. 
- DO NOT remove prefixes like 'src/', 'lib/', or 'app/'. 
- If the tree shows "src/requests/api.py", you must return exactly "src/requests/api.py".
- Select only files (not directories). Prioritize configuration files, entry points, and core source files.

{format_instructions}""",
        input_variables=["tree_text", "readme"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    # 3. Собираем пайплайн: Промпт -> LLM -> Парсер
    chain = prompt | llm | parser
    
    try:
        # Вызываем цепочку. На выходе сразу получаем Python-словарь!
        result = chain.invoke({
            "tree_text": tree_text,
            "readme": readme if readme else "No README found."
        })
        
        logger.debug("=== PARSED LLM RESPONSE (FILES) ===")
        logger.debug(result)
        logger.debug("===================================")
        
        # Парсер вернет словарь вида {"files": ["path1", "path2"]}, 
        # возвращаем только список, чтобы не сломать вашу логику дальше
        return result["files"]
        
    except Exception as e:
        logger.error(f"Failed to parse files JSON: {e}")
        raise ValueError("Could not extract valid JSON array of files from LLM response")


def generate_repo_summary(owner, repo, tree, readme, important_files, MAX_CHARS_PER_FILE = 10000):
    """Генерирует summary репозитория"""
    files_content = {}
    for file_path in important_files:
        try:
            content = get_file_content(owner, repo, file_path)
            files_content[file_path] = content
        except Exception as e:
            # Заменил print на logger
            logger.warning(f"Warning: Could not fetch {file_path}: {e}") 
    
    tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in tree])
    
    # 10000 symbols is Около 2500 токенов на файл максимум
    files_text = ""

    for path, content in files_content.items():
        # Если файл огромный, берем только начало и конец (или просто начало)
        if len(content) > MAX_CHARS_PER_FILE:
            content = content[:MAX_CHARS_PER_FILE] + "\n\n... [CONTENT TRUNCATED DUE TO SIZE] ..."
            
        files_text += f"\n\n=== FILE: {path} ===\n{content}\n"
        
    # 1. Инициализируем парсер
    parser = JsonOutputParser(pydantic_object=RepoSummarySchema)
    
    # 2. Создаем шаблон
    prompt = PromptTemplate(
        template="""Analyze this GitHub repository and provide a structured summary.

Repository tree:
{tree_text}

README:
{readme}

Key files content:
{files_text}

{format_instructions}""",
        input_variables=["tree_text", "readme", "files_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    # 3. Собираем пайплайн
    chain = prompt | llm | parser
    
    try:
        # 4. Вызываем и сразу получаем валидный словарь
        result = chain.invoke({
            "tree_text": tree_text,
            "readme": readme if readme else "No README found.",
            "files_text": files_text
        })
        
        logger.debug("=== PARSED LLM RESPONSE (SUMMARY) ===")
        logger.debug(result)
        logger.debug("=====================================")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse summary JSON: {e}")
        raise ValueError("Could not extract valid JSON object from LLM response")


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    POST /summarize
    Принимает GitHub URL и возвращает summary репозитория
    """
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data or 'github_url' not in data:
            # ИСПРАВЛЕНО: строка 327 - изменён формат ошибки с {"error": ...} на {"status": "error", "message": ...}
            return jsonify({
                "status": "error",
                "message": "Missing required field: github_url"
            }), 400
        
        github_url = data['github_url']
        
        # Парсим URL
        try:
            owner, repo = parse_github_url(github_url)
        except ValueError as e:
            # ИСПРАВЛЕНО: строка 338 - изменён формат ошибки с {"error": ...} на {"status": "error", "message": ...}
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
        
        # Получаем дерево и README
        tree = get_repo_tree(owner, repo, max_depth=2)
        readme = get_readme(owner, repo)
        
        # Анализируем структуру
        important_files = analyze_repo_structure(owner, repo, tree, readme)
        
        # Генерируем summary
        summary_data = generate_repo_summary(owner, repo, tree, readme, important_files)
        
        return jsonify(summary_data), 200
        
    except requests.exceptions.HTTPError as e:
        # ИСПРАВЛЕНО: строка 359 - изменён формат ошибки с {"error": ...} на {"status": "error", "message": ...}
        return jsonify({
            "status": "error",
            "message": f"GitHub API error: {str(e)}"
        }), 502
        
    except json.JSONDecodeError as e:
        # ИСПРАВЛЕНО: строка 366 - изменён формат ошибки с {"error": ...} на {"status": "error", "message": ...}
        return jsonify({
            "status": "error",
            "message": f"Failed to parse LLM response: {str(e)}"
        }), 500
        
    except Exception as e:
        # ИСПРАВЛЕНО: строка 373 - изменён формат ошибки с {"error": ...} на {"status": "error", "message": ...}
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
