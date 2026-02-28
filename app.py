from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
import re
import base64

import logging

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
    temperature=0
)

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
    """Получает дерево файлов репозитория с ограничением глубины"""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tree = response.json().get("tree", [])
    
    # Фильтруем по глубине
    filtered_tree = []
    for item in tree:
        depth = item["path"].count("/")
        if depth < max_depth:
            filtered_tree.append(item)
    
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
    
    prompt = f"""You are analyzing a GitHub repository structure.

Repository tree (depth up to 2):
{tree_text}

{"README content:" if readme else "No README found."}
{readme if readme else ""}

Based on this structure and README, select up to 6 most valuable files that would help understand:
1. What this project does
2. How it works
3. Its architecture and main components

Return ONLY a JSON array of file paths, for example:
["src/main.py", "config.yaml", "package.json"]

Important: Select only files (not directories), prioritize configuration files, entry points, and core source files.
"""
    
    response = llm.invoke(prompt)
    content = response.content.strip()

    # ДОБАВЬТЕ ЭТОТ ПРИНТ ДЛЯ ОТЛАДКИ:
    logger.debug("=== RAW LLM RESPONSE ===")
    logger.debug(repr(content)) # repr покажет все скрытые символы и пустые строки
    logger.debug("========================")
    
    # Ищем массив в квадратных скобках
    match = re.search(r'\[.*?\]', content, re.DOTALL)
    if match:
        json_str = match.group(0)
        files = json.loads(json_str)
    else:
        raise ValueError("Could not find JSON array in LLM response")
    
    return files


def generate_repo_summary(owner, repo, tree, readme, important_files):
    """Генерирует summary репозитория"""
    # Получаем содержимое важных файлов
    files_content = {}
    for file_path in important_files:
        try:
            content = get_file_content(owner, repo, file_path)
            files_content[file_path] = content
        except Exception as e:
            print(f"Warning: Could not fetch {file_path}: {e}")
    
    tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in tree])
    
    files_text = ""
    for path, content in files_content.items():
        files_text += f"\n\n=== FILE: {path} ===\n{content}\n"
    
    prompt = f"""Analyze this GitHub repository and provide a structured summary.

Repository tree:
{tree_text}

{"README:" if readme else ""}
{readme if readme else ""}

Key files content:
{files_text}

Return a valid JSON object with exactly these fields:
{{
  "summary": "A clear, human-readable description of what this project does",
  "technologies": ["list", "of", "main", "technologies", "languages", "frameworks"],
  "structure": "Brief description of the project structure and organization"
}}

Return ONLY the JSON object, no additional text.
"""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # ДОБАВЬТЕ ЭТОТ ПРИНТ ДЛЯ ОТЛАДКИ:
    logger.debug("=== RAW LLM RESPONSE ===")
    logger.debug(repr(content)) # repr покажет все скрытые символы и пустые строки
    logger.debug("========================")

    # Ищем объект в фигурных скобках
    match = re.search(r'\{.*?\}', content, re.DOTALL)
    if match:
        json_str = match.group(0)
        result = json.loads(json_str)
    else:
        raise ValueError("Could not find JSON object in LLM response")
    
    return result


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