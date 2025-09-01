import openai
import dotenv
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- ключи из .env ---
try:
    env = dotenv.dotenv_values(".env")
    YA_API_KEY = env["YA_API_KEY"]
    YA_FOLDER_ID = env["YA_FOLDER_ID"]
except FileNotFoundError:
    raise FileNotFoundError("Файл .env не найден.")
except KeyError as e:
    raise KeyError(f"Переменная окружения {str(e)} не найдена в .env")

class LLMService:
    """YandexGPT через OpenAI-совместимый клиент с собственным системным промптом."""
    def __init__(self, prompt_file: str):
        with open(prompt_file, encoding="utf-8") as f:
            self.sys_prompt = f.read()

        self.client = openai.OpenAI(
            api_key=YA_API_KEY,
            base_url="https://llm.api.cloud.yandex.net/v1",
        )
        self.model = f"gpt://{YA_FOLDER_ID}/yandexgpt-lite"

    def _trim_history(self, history: List[Dict], budget_chars: int = 4000) -> List[Dict]:
        if not history:
            return []
        total = 0
        out = []
        for msg in reversed(history):
            piece = len(msg.get("content", "")) + 20
            if total + piece > budget_chars:
                break
            out.append(msg)
            total += piece
        return list(reversed(out))

    def chat(self, message: str, history: List[Dict]) -> str:
        messages = [{"role": "system", "content": self.sys_prompt}]
        messages += self._trim_history(history)
        messages += [{"role": "user", "content": message}]
        logger.debug("Messages to LLM: %s", messages)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            return resp.choices[0].message.content
        except Exception:
            logger.exception("LLM request failed")
            return "Извини, сейчас не получается ответить. Попробуй позже."

# --- два сервиса под разные режимы ---
llm_consult = LLMService("prompts/prompt_consultant.txt")
llm_check   = LLMService("prompts/prompt_check.txt")

def chat_with_llm(user_message: str, history: List[Dict], mode: str = "consult"):
    """
    mode: "consult" | "check"
    """
    svc = llm_check if mode == "check" else llm_consult
    answer = svc.chat(user_message, history)

    # в историю добавляем только нормальные ответы
    if answer and not answer.startswith("Извини"):
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

    return answer
