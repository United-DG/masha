import asyncio
import base64
import io
import os
import random
import re
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, BusinessConnection, InputStoryContentPhoto, Message, Update
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
OWNER_ID = int(os.environ["OWNER_ID"])  # your Telegram user id
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # chat replies
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")  # story captions (needs image input)
INSTRUCTIONS_FILE = os.getenv("INSTRUCTIONS_FILE", "instructions.md")
STORY_INSTRUCTIONS_FILE = os.getenv("STORY_INSTRUCTIONS_FILE", "story_instructions.md")
DB_FILE = os.getenv("DB_FILE", "memory.db")
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "40"))
KEEP_PER_CHAT = int(os.getenv("KEEP_PER_CHAT", "500"))

STORIES_DIR = os.getenv("STORIES_DIR", "stories")
POSTED_DIR = os.path.join(STORIES_DIR, "posted")
STORY_TIMES = [t.strip() for t in os.getenv("STORY_TIMES", "00:00,09:00").split(",") if t.strip()]
UTC_OFFSET_HOURS = float(os.getenv("UTC_OFFSET_HOURS", "3"))  # GMT+3
STORY_PERIOD = int(os.getenv("STORY_PERIOD", "86400"))  # 21600 / 43200 / 86400 / 172800
TZ = timezone(timedelta(hours=UTC_OFFSET_HOURS))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
away = True  # chat auto-replies: /away on | /away off
story_lock = asyncio.Lock()
os.makedirs(POSTED_DIR, exist_ok=True)

# ---------- SQLite: chat memory + small settings ----------
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
db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
db.commit()


def get_setting(key: str, default: str | None = None) -> str | None:
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(key: str, value: str) -> None:
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    db.commit()


def remember(chat_id: int, role: str, content: str) -> None:
    db.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content),
    )
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
def read_file(path: str) -> str:
    # re-read on every use, so edits apply instantly (no restart)
    with open(path, encoding="utf-8") as f:
        return f.read()


async def groq_chat(model: str, messages: list[dict], **extra) -> str:
    """Call Groq, retrying temporary failures (429 / 5xx / timeouts) with backoff."""
    payload = {"model": model, "messages": messages, "stream": False, **extra}
    delay = 3
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=90) as http:
                r = await http.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}"},
                    json=payload,
                )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                wait = min(float(r.headers.get("retry-after", delay)), 30)
                print(f"Groq {r.status_code}, retry {attempt + 1}/4 in {wait:.0f}s")
                await asyncio.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == 4:
                raise
            print(f"Groq network error ({e!r}), retry {attempt + 1}/4 in {delay}s")
            await asyncio.sleep(delay)
            delay *= 2
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


async def ask_groq(chat_id: int) -> str:
    messages = [{"role": "system", "content": read_file(INSTRUCTIONS_FILE)}] + recall(chat_id)
    return await groq_chat(
        MODEL, messages, max_tokens=1000, reasoning_effort="low", temperature=1.0
    )


# ---------- images ----------
def make_story_jpeg(raw: bytes) -> bytes:
    """Telegram stories need 1080x1920 (<10 MB): fit the photo over a blurred copy of itself."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    size = (1080, 1920)
    canvas = ImageOps.fit(img, size).filter(ImageFilter.GaussianBlur(40))
    fg = ImageOps.contain(img, size)
    canvas.paste(fg, ((size[0] - fg.width) // 2, (size[1] - fg.height) // 2))
    for quality in (90, 80, 70):
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=quality, optimize=True)
        if buf.tell() < 9_500_000:
            break
    return buf.getvalue()


def make_vision_b64(raw: bytes) -> str:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


async def caption_for(raw: bytes) -> str:
    b64 = await asyncio.to_thread(make_vision_b64, raw)
    messages = [
        {"role": "system", "content": read_file(STORY_INSTRUCTIONS_FILE)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write the story caption for this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        },
    ]
    caption = await groq_chat(VISION_MODEL, messages, max_tokens=1000, temperature=0.9)
    return caption[:2000]


def list_images() -> list[str]:
    return [
        os.path.join(STORIES_DIR, f)
        for f in os.listdir(STORIES_DIR)
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(STORIES_DIR, f))
    ]


def move_to_posted(path: str) -> None:
    dest = os.path.join(POSTED_DIR, os.path.basename(path))
    if os.path.exists(dest):
        name, ext = os.path.splitext(os.path.basename(path))
        dest = os.path.join(POSTED_DIR, f"{name}_{int(time.time())}{ext}")
    shutil.move(path, dest)


# ---------- stories ----------
async def notify(text: str) -> None:
    try:
        await bot.send_message(OWNER_ID, text)
    except Exception as e:
        print("notify failed:", e)


async def post_story(raw: bytes, caption: str | None) -> None:
    conn = current_connection_id()
    if not conn:
        raise RuntimeError(
            "No business connection saved yet. Re-save the bot in Settings > Business > "
            "Chatbots (or wait for a chat message), then try again."
        )
    jpeg = await asyncio.to_thread(make_story_jpeg, raw)
    # model_construct lets us pass the upload; aiogram turns it into attach://
    content = InputStoryContentPhoto.model_construct(
        photo=BufferedInputFile(jpeg, filename="story.jpg")
    )
    await bot.post_story(
        business_connection_id=conn,
        content=content,
        active_period=STORY_PERIOD,
        caption=caption or None,
    )


async def run_story_from_folder(source: str) -> bool:
    async with story_lock:
        files = list_images()
        if not files:
            await notify(f"Story ({source}): no images left in {STORIES_DIR}/. Add more!")
            return True  # nothing to retry
        path = random.choice(files)
        try:
            with open(path, "rb") as f:
                raw = f.read()
            caption = await caption_for(raw)
            await post_story(raw, caption)
            move_to_posted(path)
        except Exception as e:
            await notify(f"Story ({source}) failed: {e}")
            return False
        left = len(files) - 1
        msg = f"Story posted ({source}): {os.path.basename(path)}\nCaption: {caption}\nImages left: {left}"
        if left <= 3:
            msg += "\nRunning low, add more images soon."
        await notify(msg)
        return True


def next_run(now: datetime) -> datetime:
    slots = sorted(time.strptime(t, "%H:%M")[3:5] for t in STORY_TIMES)
    for day in (0, 1):
        for h, m in slots:
            dt = (now + timedelta(days=day)).replace(hour=h, minute=m, second=0, microsecond=0)
            if dt > now:
                return dt
    raise RuntimeError("STORY_TIMES is empty")


async def scheduler() -> None:
    while True:
        target = next_run(datetime.now(TZ))
        while (remaining := (target - datetime.now(TZ)).total_seconds()) > 0:
            await asyncio.sleep(min(remaining, 30))
        if (datetime.now(TZ) - target).total_seconds() > 1800:
            continue  # machine was asleep, skip the missed slot
        if get_setting("auto_stories", "on") == "on":
            for attempt in range(3):  # retry later if Groq/Telegram is down
                if await run_story_from_folder(f"scheduled, try {attempt + 1}/3"):
                    break
                if attempt < 2:
                    await asyncio.sleep(900)  # 15 min


# ---------- capture the business connection id from ANY business update ----------
def current_connection_id() -> str | None:
    return get_setting("business_connection_id") or os.getenv("BUSINESS_CONNECTION_ID") or None


@dp.update.outer_middleware()
async def capture_connection(handler, event: Update, data):
    print("update:", event.event_type)
    print(event.model_dump_json(exclude_none=True)[:2000])  # DEBUG: full update, remove later
    conn_id = None
    if event.business_connection is not None:
        if event.business_connection.is_enabled:
            conn_id = event.business_connection.id
    else:
        for m in (event.business_message, event.edited_business_message):
            if m is not None and m.business_connection_id:
                conn_id = m.business_connection_id
        if event.deleted_business_messages is not None:
            conn_id = event.deleted_business_messages.business_connection_id
    if conn_id and conn_id != get_setting("business_connection_id"):
        set_setting("business_connection_id", conn_id)
        print("Saved business connection id:", conn_id)
    return await handler(event, data)


# ---------- handlers ----------
@dp.message(Command("away"), F.from_user.id == OWNER_ID)
async def toggle_away(message: Message, command: CommandObject):
    global away
    arg = (command.args or "").strip().lower()
    if arg in ("on", "off"):
        away = arg == "on"
    await message.answer(f"Chat auto-reply: {'ON' if away else 'OFF'}")


@dp.message(Command("story"), F.from_user.id == OWNER_ID)
async def story_cmd(message: Message, command: CommandObject):
    args = (command.args or "").strip().lower().split()
    if not args:  # manual trigger: post one from the folder right now
        await message.answer("Posting a story now...")
        await run_story_from_folder("manual")
    elif args[0] == "auto" and len(args) > 1 and args[1] in ("on", "off"):
        set_setting("auto_stories", args[1])
        await message.answer(f"Auto stories: {args[1].upper()}")
    elif args[0] == "status":
        nxt = next_run(datetime.now(TZ)).strftime("%a %H:%M")
        await message.answer(
            f"Auto stories: {get_setting('auto_stories', 'on').upper()}\n"
            f"Times: {', '.join(STORY_TIMES)} (GMT{UTC_OFFSET_HOURS:+g})\n"
            f"Next: {nxt}\n"
            f"Images left: {len(list_images())}\n"
            f"Business connection: {'saved' if current_connection_id() else 'MISSING'}"
        )
    else:
        await message.answer(
            "/story - post one now from the folder\n"
            "/story auto on|off - scheduled stories\n"
            "/story status\n"
            "Or send me a photo to post it as a story (your caption is used, "
            "or AI writes one)."
        )


@dp.message(F.photo, F.from_user.id == OWNER_ID)
async def manual_photo(message: Message):
    async with story_lock:
        try:
            buf = await bot.download(message.photo[-1])
            raw = buf.read()
            caption = message.caption or await caption_for(raw)
            await post_story(raw, caption)
            await message.answer(f"Story posted.\nCaption: {caption}")
        except Exception as e:
            await message.answer(f"Story failed: {e}")


@dp.business_connection()
async def on_connection(conn: BusinessConnection):
    if conn.is_enabled:
        set_setting("business_connection_id", conn.id)
        rights = conn.rights
        if rights is not None and not getattr(rights, "can_manage_stories", True):
            await notify("Bot connected, but 'Manage Stories' permission is off.")


@dp.business_message(F.text)
async def on_business_message(message: Message):
    conn = message.business_connection_id
    if conn:
        set_setting("business_connection_id", conn)
    # skip your own messages (owner's id != the customer's chat id) and bots
    if not away or message.from_user.is_bot or message.from_user.id != message.chat.id:
        return

    chat_id = message.chat.id
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
    await notify(
        f"Away bot is online.\nChat auto-reply: {'ON' if away else 'OFF'}\n"
        f"Auto stories: {get_setting('auto_stories', 'on').upper()} ({', '.join(STORY_TIMES)})\n"
        "Send /story status for details."
    )
    me = await bot.me()
    hook = await bot.get_webhook_info()
    print(f"Bot: @{me.username} | business mode on: {getattr(me, 'can_connect_to_business', None)}")
    print(f"Webhook url: {hook.url or '(none)'} | pending updates: {hook.pending_update_count}")
    asyncio.create_task(scheduler())
    await dp.start_polling(
        bot,
        allowed_updates=[
            "message",
            "business_connection",
            "business_message",
            "edited_business_message",
            "deleted_business_messages",
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())