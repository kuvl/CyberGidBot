"""
bot.py — Telegram-бот «Кибер Гид»

Функциональность:
- Главное меню (ReplyKeyboard): 🧑‍💻 Консультант / 🔍 Проверка / 📋 Инструкции
- Консультант: свободные вопросы к ИИ (короткие ответы с шагами)
- Проверка: строгое заключение по тексту (Вердикт/Почему/Что делать/Примечание)
- Инструкции: полезные памятки (inline-кнопки)
"""

import logging
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
import dotenv

from model import chat_with_llm

# --- логирование ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- токен ---
try:
    env = dotenv.dotenv_values(".env")
    TELEGRAM_BOT_TOKEN = env["TELEGRAM_BOT_TOKEN"].strip()
except FileNotFoundError:
    raise FileNotFoundError("Файл .env не найден. Положите его в корень проекта.")
except KeyError as e:
    raise KeyError(f"Переменная окружения {str(e)} не найдена в .env")

# --- главное меню ---
MAIN_KB = ReplyKeyboardMarkup(
    [
        ["🧑‍💻 Консультант", "🔍 Проверка"],
        ["📋 Инструкции"],
    ],
    resize_keyboard=True,
)

# --- инструкции: лаконичные чек-листы для обычного пользователя ---
INSTRUCTIONS = {
    "QUICK_START": (
        "🚀 Быстрый старт: 5 обязательных шагов\n"
        "1) Пароли: длинные и уникальные, храни в менеджере.\n"
        "2) 2FA: включи в почте, банке, соцсетях (через приложение-аутентификатор).\n"
        "3) Обновления: ОС и приложения — сразу, не откладывай.\n"
        "4) Ссылки: не переходи из писем/чата; вводи адрес вручную.\n"
        "5) Копии: сделай резервные копии важных файлов.\n"
        "Важно: коды/пароли никому не сообщай — даже «сотруднику банка»."
    ),
    "PHISHING": (
        "🎣 Фишинг: как распознать\n"
        "Признаки: срочность и угрозы; странные ссылки/домены; просьба кодов/CVV/доков; вложения (.zip/.exe/.docm);\n"
        "ссылки-сокращатели; «вы выиграли».\n"
        "Что делать: не кликай; зайди на сайт вручную; проверь отправителя официально; при сомнениях звони по номеру с карты/сайта."
    ),
    "BANK_CODES": (
        "📞 Коды и звонки «из банка» — правила\n"
        "• Код из SMS — только для тебя. Не диктуй и не пересылай.\n"
        "• Банк не просит код, CVV, «перевести на безопасный счёт».\n"
        "• Звонок «из безопасности»? Положи трубку и сам набери номер с карты/в приложении.\n"
        "• Если ввёл код не там — срочно позвони в банк, смени пароль и включи 2FA."
    ),
    "2FA": (
        "🔑 Двухфакторная аутентификация (2FA)\n"
        "1) Включи 2FA в почте, банке, соцсетях, маркетплейсах.\n"
        "2) Предпочитай приложение-аутентификатор (Google Authenticator/Authy/1Password).\n"
        "3) Сохрани резервные коды в менеджере паролей или офлайн.\n"
        "4) При смене телефона заранее перенеси 2FA."
    ),
    "PASSWORDS": (
        "🔒 Пароли и менеджер\n"
        "1) Один сервис — один пароль (12–16+ символов).\n"
        "2) Используй менеджер (1Password, Bitwarden, KeePass).\n"
        "3) Включи проверку утечек/слабых паролей и замени их.\n"
        "4) Пароли не отправляй в чатах/почте. Главный пароль — запомни."
    ),
    "BREACH": (
        "⚠️ Если аккаунт взломан/под угрозой\n"
        "1) Сменить пароль, включить 2FA. Выйти со всех устройств.\n"
        "2) Проверить e-mail/телефон и доверенные устройства.\n"
        "3) Отключить подозрительные сессии/ключи/приложения.\n"
        "4) Написать в поддержку; в банке — проверить операции/споры.\n"
        "5) Просканировать устройство, обновить ОС/браузер, включить бэкап."
    ),
    "BACKUP": (
        "💾 Резервные копии (правило 3-2-1)\n"
        "• 3 копии; 2 разных носителя; 1 — вне устройства (облако/внешний диск).\n"
        "Быстрый старт: включи бэкап в телефоне и облачную папку документов.\n"
        "План: авто-копия раз в день/неделю; проверка восстановления раз в квартал."
    ),
}

def instruction_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Быстрый старт", callback_data="INS::QUICK_START")],
        [InlineKeyboardButton("🎣 Фишинг", callback_data="INS::PHISHING"),
         InlineKeyboardButton("📞 Коды/звонки", callback_data="INS::BANK_CODES")],
        [InlineKeyboardButton("🔑 2FA", callback_data="INS::2FA"),
         InlineKeyboardButton("🔒 Пароли", callback_data="INS::PASSWORDS")],
        [InlineKeyboardButton("⚠️ При взломе", callback_data="INS::BREACH"),
         InlineKeyboardButton("💾 Копии", callback_data="INS::BACKUP")],
    ])

# --- /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Приветствие + главное меню. По умолчанию режим — «Консультант».
    """
    welcome = (
        "**Добро пожаловать!**\n"
        "Я — **Кибер Гид**, твой помощник в мире кибербезопасности 🛡️\n\n"
        "Что могу:\n"
        "🧑‍💻 Ответить на твои вопросы\n"
        "🔍 Проверить сообщения на мошенничество\n"
        "📋 Дать инструкции, как защитить данные\n\n"
        "Выбирай кнопку в меню — и начнём!"
    )
    context.user_data["mode"] = "consult"
    await update.message.reply_text(welcome, reply_markup=MAIN_KB, parse_mode="Markdown")

# --- обработчик inline-кнопок «Инструкции» ---
async def on_instruction_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    key = q.data.split("::", 1)[1]
    await q.edit_message_text(INSTRUCTIONS.get(key, "Нет такой инструкции."))

# --- единый обработчик всех текстов (кнопки + обычные сообщения) ---
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    # 1) кнопки главного меню
    if text == "🧑‍💻 Консультант":
        context.user_data["mode"] = "consult"
        await update.message.reply_text("Задай вопрос — отвечу простыми шагами 👨‍💻")
        return

    if text == "🔍 Проверка":
        context.user_data["mode"] = "check"
        await update.message.reply_text("Вставь текст письма/сообщения — дам анализ по схеме 🔍")
        return

    if text == "📋 Инструкции":
        await update.message.reply_text("Выбери инструкцию 👇", reply_markup=instruction_kb())
        return

    # 2) обычное сообщение — уходит в LLM с выбранным режимом
    mode = context.user_data.get("mode", "consult")  # consult | check
    history = context.chat_data.get("history", [])
    try:
        answer = chat_with_llm(text, history=history, mode=mode)
        context.chat_data["history"] = history
    except Exception:
        logger.exception("LLM error")
        answer = "Извини, сейчас не получается ответить. Попробуй позже."

    await update.message.reply_text(answer)

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(on_instruction_click, pattern="^INS::"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
