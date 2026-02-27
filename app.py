import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
import re

# Загружаем переменные окружения
load_dotenv()

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Инициализируем LLM
llm = ChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    temperature=0
)


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
    
    # Получаем полное дерево
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tree = response.json().get("tree", [])
    
    # Фильтруем по глубине (считаем количество слешей)
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
            # Декодируем из base64
            import base64
            return base64.b64decode(content).decode("utf-8")
    
    return None


def get_file_content(owner, repo, file_path):
    """Получает содержимое конкретного файла"""
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    content = response.json().get("content", "")
    import base64
    return base64.b64decode(content).decode("utf-8")


def analyze_repo_structure(github_url):
    """
    Функция 1: Анализирует структуру репозитория и выбирает важные файлы
    
    Args:
        github_url: URL GitHub репозитория
        
    Returns:
        list: Список путей до 6 самых важных файлов
    """
    # Парсим URL
    owner, repo = parse_github_url(github_url)
    
    # Получаем дерево и README
    tree = get_repo_tree(owner, repo, max_depth=2)
    readme = get_readme(owner, repo)
    
    # Формируем текстовое представление дерева
    tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in tree])
    
    # Формируем промпт
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
    
    # Отправляем в LLM
    response = llm.invoke(prompt)
    
    # Парсим ответ
    content = response.content.strip()
    # Убираем markdown форматирование если есть
    content = content.replace("```json", "").replace("```", "").strip()
    
    files = json.loads(content)
    
    return files, owner, repo, tree, readme


def generate_repo_summary(owner, repo, tree, readme, important_files):
    """
    Функция 2: Генерирует summary репозитория
    
    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        tree: Дерево файлов
        readme: Содержимое README
        important_files: Список важных файлов
        
    Returns:
        dict: JSON с полями summary, technologies, structure
    """
    # Получаем содержимое важных файлов
    files_content = {}
    for file_path in important_files:
        try:
            content = get_file_content(owner, repo, file_path)
            files_content[file_path] = content
        except Exception as e:
            print(f"Warning: Could not fetch {file_path}: {e}")
    
    # Формируем дерево как текст
    tree_text = "\n".join([f"{item['type']}: {item['path']}" for item in tree])
    
    # Формируем содержимое файлов как текст
    files_text = ""
    for path, content in files_content.items():
        files_text += f"\n\n=== FILE: {path} ===\n{content}\n"
    
    # Промпт
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
    
    # Отправляем в LLM
    response = llm.invoke(prompt)
    
    # Парсим JSON
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    result = json.loads(content)
    
    return result


# Пример использования
if __name__ == "__main__":
    # Пример URL
    github_url = "https://github.com/Comfy-Org/ComfyUI"
    
    print("Step 1: Analyzing repository structure...")
    important_files, owner, repo, tree, readme = analyze_repo_structure(github_url)
    print(f"Selected files: {important_files}")
    
    print("\nStep 2: Generating summary...")
    summary = generate_repo_summary(owner, repo, tree, readme, important_files)
    
    # Выводим в консоль для наглядности
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Сохраняем в файл
    with open("repo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\n✅ Результат сохранен в файл repo_summary.json")