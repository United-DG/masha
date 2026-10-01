import asyncio
import os
import random
import sqlite3

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
OWNER_ID = int(os.environ["OWNER_ID"])  # your Telegram user id
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
INSTRUCTIONS_FILE = os.getenv("INSTRUCTIONS_FILE", "instructions.md")
DB_FILE = os.getenv("DB_FILE", "memory.db")
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "40"))  # how much the bot "remembers" per chat
KEEP_PER_CHAT = int(os.getenv("KEEP_PER_CHAT", "500"))  # older rows get pruned

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
away = True  # toggle by messaging the bot: /away on | /away off

# ---------- persistent memory (SQLite) ----------
db = sqlite3.connect(DB_FILE, check_same_thread=False)
db.execute("PRAGMA journal_mode=WAL")
db.execute(
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts TEXT DEFAULT CURRENT_TIMESTAMP
    )"""
)
db.execute("CREATE INDEX IF NOT EXISTS idx_chat ON messages (chat_id, id)")
db.commit()


def remember(chat_id: int, role: str, content: str) -> None:
    db.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content),
    )
    # prune: keep only the newest KEEP_PER_CHAT rows for this chat
    db.execute(
        """DELETE FROM messages WHERE chat_id = ? AND id NOT IN (
               SELECT id FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?)""",
        (chat_id, chat_id, KEEP_PER_CHAT),
    )
    db.commit()


def recall(chat_id: int) -> list[dict]:
    rows = db.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (chat_id, CONTEXT_MESSAGES),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


# ---------- Groq ----------
def load_instructions() -> str:
    # re-read on every message, so edits apply instantly (no restart)
    with open(INSTRUCTIONS_FILE, encoding="utf-8") as f:
        return f.read()


async def ask_groq(chat_id: int) -> str:
    messages = [{"role": "system", "content": load_instructions()}]
    messages += recall(chat_id)
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 1000,
                "reasoning_effort": "low",
                "temperature": 1.0,
                "stream": False,
            },
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


# ---------- handlers ----------
@dp.message(Command("away"), F.from_user.id == OWNER_ID)
async def toggle(message: Message, command: CommandObject):
    global away
    arg = (command.args or "").strip().lower()
    if arg in ("on", "off"):
        away = arg == "on"
    await message.answer(f"Away mode: {'ON' if away else 'OFF'}")


@dp.business_message(F.text)
async def on_business_message(message: Message):
    # skip your own messages (owner's id != the customer's chat id) and bots
    if not away or message.from_user.is_bot or message.from_user.id != message.chat.id:
        return

    chat_id = message.chat.id
    conn = message.business_connection_id
    remember(chat_id, "user", message.text)
    try:
        reply = await ask_groq(chat_id)
    except Exception as e:
        print("Groq error:", e)
        return
    if not reply:
        return

    remember(chat_id, "assistant", reply)
    await bot.send_chat_action(chat_id, "typing", business_connection_id=conn)
    await asyncio.sleep(random.uniform(1.5, 4))
    await bot.send_message(chat_id, reply, business_connection_id=conn)


async def main():
    print("Running.")
    try:
        await bot.send_message(
            OWNER_ID,
            f"Away bot is online. Away mode: {'ON' if away else 'OFF'}\n"
            "Use /away on or /away off to toggle.",
        )
    except Exception as e:
        print("Could not send startup message:", e)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())