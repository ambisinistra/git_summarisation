from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
import re
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
        logging.FileHandler("app.log", encoding="utf-8"),
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
    #model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
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


def parse_github_url(url):
    """Парсит GitHub URL и возвращает owner и repo"""
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo
    raise ValueError("Invalid GitHub URL")


def get_repo_tree(owner, repo, max_depth=2):
    """Получает дерево файлов репозитория с ограничением глубины и логированием исключенных файлов"""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tree = response.json().get("tree", [])
    
    # Расширения, которые мы хотим исключить
    binary_extensions = (
        '.exe', '.dll', '.so', '.a', '.lib', '.dylib', '.o', '.obj',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf', '.doc', '.docx',
        '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.iso', '.db', 
        '.sqlite', '.jar', '.class', '.pyc', '.whl', '.ds_store', '.svg',
        '.woff', '.woff2', '.ttf', '.eot', '.ico', '.lockb', 'package-lock.json'
    )
    
    filtered_tree = []
    excluded_tree = []
    
    for item in tree:
        # 1. Проверяем глубину
        depth = item["path"].count("/")
        if depth >= max_depth:
            continue
            
        # 2. Проверяем расширения (бинарники и ненужные файлы)
        if item["type"] == "blob" and item["path"].lower().endswith(binary_extensions):
            # Файл попал под фильтр — добавляем в список исключенных
            excluded_tree.append(item)
            continue
            
        # 3. Файл прошел все фильтры — добавляем в итоговое дерево
        filtered_tree.append(item)
        
    # --- ЛОГИРОВАНИЕ ---
    
    # Форматируем списки в читаемый текст
    filtered_tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in filtered_tree])
    excluded_tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in excluded_tree])
    
    logger.debug("=== ИТОГОВОЕ ДЕРЕВО (отправляется в LLM) ===")
    logger.debug(f"\n{filtered_tree_text}")
    logger.debug(f"Всего элементов: {len(filtered_tree)}\n")
    
    logger.debug("=== ИСКЛЮЧЕННЫЕ ФАЙЛЫ (отфильтрованы по расширению) ===")
    logger.debug(f"\n{excluded_tree_text if excluded_tree else 'Нет исключенных файлов'}")
    logger.debug(f"Всего отсеяно: {len(excluded_tree)}\n")
    logger.debug("======================================================")
    
    # Возвращаем очищенное дерево
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

Important: Select only files (not directories), prioritize configuration files, entry points, and core source files.

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
            return jsonify({
                "error": "Missing required field: github_url"
            }), 400
        
        github_url = data['github_url']
        
        # Парсим URL
        try:
            owner, repo = parse_github_url(github_url)
        except ValueError as e:
            return jsonify({
                "error": str(e)
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
        return jsonify({
            "error": f"GitHub API error: {str(e)}"
        }), 502
        
    except json.JSONDecodeError as e:
        return jsonify({
            "error": f"Failed to parse LLM response: {str(e)}"
        }), 500
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)