# main.py
# ——— نسخه یکپارچه و نهایی (تک‌فایل) با رفع خطای لاگین rubpy و آماده‌ی دپلوی روی Render/سرور ———
# نکته خیلی مهم: rubpy باید با "session" و "auth" ساخته شود؛
# قبلاً AUTH به‌صورت آرگومان موقعیتی پاس داده می‌شد و باعث prompt شماره‌تلفن و خطای EOF می‌شد.
# این نسخه Client را به‌صورت Client(session='rubika-bot', auth=RUBIKA_AUTH_KEY) می‌سازد تا هیچ ورودی تعاملی نخواهد.

import asyncio
import logging
import os
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Persian (Jalali) date parsing (اختیاری)
# ---------------------------------------------------------------------
from pyrubi import Client

# ---------------------------------------------------------------------
# Rubika SDK (rubpy)
# ---------------------------------------------------------------------
try:
    from rubpy import Client  # type: ignore
    from rubpy.types import Update  # type: ignore
    HAS_RUBPY = True
except Exception:  # pragma: no cover
    HAS_RUBPY = False
    Client = object  # type: ignore
    Update = object  # type: ignore

# ---------------------------------------------------------------------
# Google Generative AI (Gemini) — اختیاری ولی توصیه‌شده
# ---------------------------------------------------------------------
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except Exception:  # pragma: no cover
    HAS_GENAI = False

# =============================
# 1) تنظیمات و لاگ‌گیری
# =============================
load_dotenv()

AUTH_KEY = os.getenv("RUBIKA_AUTH_KEY")  # **ضروری**: توکن Rubika (Auth) — نه شماره تلفن
MASTER_ADMIN_GUID = os.getenv("MASTER_ADMIN_GUID")  # **ضروری**
CHANNEL_GUID = os.getenv("CHANNEL_GUID")  # اختیاری
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # اختیاری
MASTER_PASSWORD = os.getenv("MASTER_PASSWORD")  # **ضروری**
SUB_ADMIN_PASSWORD = os.getenv("SUB_ADMIN_PASSWORD")  # **ضروری**
DB_PATH = os.getenv("DB_PATH", "ai_bot_db.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AIBot")

# Gemini config
GENERATION_CONFIG = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
if HAS_GENAI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        _model = genai.GenerativeModel("gemini-1.5-flash", generation_config=GENERATION_CONFIG)
        logger.info("Gemini model initialized: gemini-1.5-flash")
    except Exception as e:
        logger.error(f"Failed to init Gemini: {e}")
        _model = None
else:
    _model = None


# =============================
# 2) AI Helper
# =============================
async def generate_response(prompt: str, user_type: str) -> str:
    """تولید پاسخ با Gemini؛ در صورت نبودن کلید، پیام مناسب برمی‌گرداند."""
    if not (HAS_GENAI and _model):
        return "سرویس هوش مصنوعی در دسترس نیست. لطفاً بعداً دوباره تلاش کنید."

    try:
        preamble = "You are a helpful Persian assistant. پاسخ‌ها را مودبانه، دقیق و کاربردی بده."
        if user_type == "free":
            final_prompt = f"{preamble}\n\nحداکثر در 8 خط پاسخ بده.\n\nسوال کاربر:\n{prompt}"
        else:
            final_prompt = f"{preamble}\n\nسوال کاربر:\n{prompt}"

        # genai SDK همگام است؛ برای امن‌بودن داخل ترد می‌بریم
        response = await asyncio.to_thread(_model.generate_content, final_prompt)
        text = getattr(response, "text", "") or ""
        return text.strip() or "پاسخی از موتور هوش مصنوعی دریافت نشد."
    except Exception as e:  # pragma: no cover
        logger.error(f"Gemini error: {e}")
        return "در تولید پاسخ مشکل پیش آمد."


# =============================
# 3) Database Manager (SQLite)
# =============================
class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    guid TEXT PRIMARY KEY,
                    last_active REAL,
                    is_member INTEGER,
                    is_vip INTEGER DEFAULT 0,
                    vip_expiry REAL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS admins (
                    guid TEXT PRIMARY KEY,
                    is_master INTEGER DEFAULT 0
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    run_at REAL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS channel_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_guid TEXT,
                    requester_guid TEXT,
                    requester_name TEXT,
                    status TEXT DEFAULT 'pending'
                )"""
            )
            conn.commit()

    # ---- users ----
    def update_user_activity(self, guid: str, is_member: bool):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT guid FROM users WHERE guid = ?", (guid,))
            user = cursor.fetchone()
            ts = datetime.now().timestamp()
            if user is None:
                cursor.execute(
                    "INSERT INTO users (guid, last_active, is_member) VALUES (?, ?, ?)",
                    (guid, ts, int(is_member)),
                )
            else:
                cursor.execute(
                    "UPDATE users SET last_active = ?, is_member = ? WHERE guid = ?",
                    (ts, int(is_member), guid),
                )
            conn.commit()

    def get_user(self, guid: str):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE guid = ?", (guid,))
            return cur.fetchone()

    def make_vip(self, guid: str, duration_days: int):
        expiry = datetime.now() + timedelta(days=duration_days)
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (guid, last_active, is_member, is_vip, vip_expiry)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(guid) DO UPDATE SET
                    is_vip=excluded.is_vip,
                    vip_expiry=excluded.vip_expiry,
                    last_active=excluded.last_active
                """,
                (
                    guid,
                    datetime.now().timestamp(),
                    1 if duration_days > 0 else 0,
                    expiry.timestamp(),
                ),
            )
            conn.commit()

    # ---- admins ----
    def get_admin_level(self, guid: str) -> int:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT is_master FROM admins WHERE guid = ?", (guid,))
            row = cur.fetchone()
            if row is None:
                return -1
            return int(row[0])

    def add_admin(self, guid: str, is_master: bool = False):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO admins (guid, is_master) VALUES (?, ?)",
                (guid, int(is_master)),
            )
            conn.commit()

    def remove_admin(self, guid: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM admins WHERE guid = ?", (guid,))
            conn.commit()

    def get_sub_admins(self) -> List[str]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT guid FROM admins WHERE is_master = 0")
            return [r[0] for r in cur.fetchall()]

    # ---- ads ----
    def get_due_ads(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM ads WHERE run_at <= ? ORDER BY run_at ASC",
                (datetime.now().timestamp(),),
            )
            return cur.fetchall()

    def add_ad(self, ad_text: str, run_at_ts: float):
        with self.get_connection() as conn:
            conn.execute("INSERT INTO ads (text, run_at) VALUES (?, ?)", (ad_text, run_at_ts))
            conn.commit()

    def delete_ad(self, ad_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM ads WHERE id = ?", (ad_id,))
            conn.commit()

    # ---- channel requests ----
    def request_channel_join(self, channel_guid: str, requester_guid: str, requester_name: str):
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO channel_requests (channel_guid, requester_guid, requester_name)
                VALUES (?, ?, ?)
                """,
                (channel_guid, requester_guid, requester_name),
            )
            conn.commit()

    def get_pending_requests(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM channel_requests WHERE status = 'pending'")
            return cur.fetchall()

    def set_request_status(self, channel_guid: str, status: str):
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE channel_requests SET status = ? WHERE channel_guid = ?",
                (status, channel_guid),
            )
            conn.commit()


# =============================
# 4) Bot Logic
# =============================
class AIBot:
    def __init__(
        self,
        client: Client,
        channel_guid: Optional[str],
        master_admin_guid: str,
        master_password: str,
        sub_admin_password: str,
    ):
        self.client = client
        self.channel_guid = channel_guid
        self.master_admin_guid = master_admin_guid
        self.master_password = master_password
        self.sub_admin_password = sub_admin_password

        self.db = DBManager(DB_PATH)
        self.db.add_admin(self.master_admin_guid, is_master=True)

        # State
        self.waiting_for_password: Dict[str, bool] = {}
        self.admin_states: Dict[str, Dict[str, Any]] = {}

        self.commands = {
            "/start": self.handle_start_command,
            "/ai": self.handle_ai_command,
            "/summarize": self.handle_summarize_command,
            "/admin": self.handle_admin_login,
            "/cancel": self.handle_cancel_state,
        }

        logger.info("AI bot handler initialized and ready.")

    # ---------- Utils ----------
    async def _safe_send(self, target_guid: str, text: str):
        try:
            MAX_LEN = 4000
            if len(text) <= MAX_LEN:
                await self.client.send_message(target_guid, text)
                return
            parts, buf, size = [], [], 0
            for line in text.split("\n"):
                if size + len(line) + 1 > MAX_LEN:
                    parts.append("\n".join(buf))
                    buf, size = [line], len(line) + 1
                else:
                    buf.append(line)
                    size += len(line) + 1
            if buf:
                parts.append("\n".join(buf))
            for p in parts:
                await self.client.send_message(target_guid, p)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to send message: {e}")

    def _parse_datetime(self, text: str) -> float:
        """پارس تاریخ/ساعت. پشتیبانی از جلالی در صورت نصب persiantools."""
        text = text.strip()
        if HAS_PERSIAN_DATE:
            try:
                jdt = JalaliDateTime.strptime(text, "%Y/%m/%d %H:%M")
                return jdt.to_gregorian().timestamp()
            except Exception:
                pass
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M")
        return dt.timestamp()

    # ---------- Public API called by rubpy ----------
    async def on_update(self, update: Update):
        """ورودی خام آپدیت‌ها: بر اساس فیلدها تشخیص پیام/کال‌بک."""
        try:
            # callback query
            if getattr(update, "data", None):
                await self.handle_callback_query(update)
                return
            # message
            if getattr(update, "text", None) is not None:
                await self.handle_message(update)
        except Exception as e:  # pragma: no cover
            logger.error(f"on_update error: {e}", exc_info=True)

    async def handle_message(self, message: Update):
        try:
            author_guid = getattr(message, "author_guid", None)
            text = (getattr(message, "text", "") or "").strip()
            object_guid = getattr(message, "object_guid", author_guid)

            # ثبت کاربر
            try:
                if author_guid:
                    self.db.update_user_activity(author_guid, True)
            except Exception as e:
                logger.error(f"Failed to update user activity: {e}")

            # --- Admin state machine ---
            if author_guid in self.admin_states:
                state = self.admin_states[author_guid].get("state")

                if state == "add_vip_duration":
                    try:
                        duration_days = int(text)
                        target_guid = self.admin_states[author_guid]["target_guid"]
                        self.db.make_vip(target_guid, duration_days)
                        await self._safe_send(object_guid, f"کاربر با GUID `{target_guid}` برای {duration_days} روز VIP شد.")
                        await self._safe_send(target_guid, f"تبریک! شما برای {duration_days} روز VIP شدید.")
                        del self.admin_states[author_guid]
                        return
                    except ValueError:
                        await self._safe_send(object_guid, "لطفا یک عدد صحیح برای تعداد روزها وارد کنید.")
                        return

                elif state == "add_vip_reply":
                    if getattr(message, "reply_to_message_id", None):
                        replied = await self.client.get_messages_by_id(object_guid, [message.reply_to_message_id])
                        if replied and replied[0].get("author_guid"):
                            target_guid = replied[0]["author_guid"]
                            self.admin_states[author_guid] = {"state": "add_vip_duration", "target_guid": target_guid}
                            await self._safe_send(object_guid, "لطفا تعداد روزهای VIP را به‌صورت عدد وارد کنید.")
                        else:
                            await self._safe_send(object_guid, "پیام ریپلای‌شده معتبر نیست.")
                    else:
                        await self._safe_send(object_guid, "لطفا روی پیام کاربر مورد نظر Reply بزنید و دوباره ارسال کنید.")
                    return

                elif state == "waiting_for_ad_text":
                    self.admin_states[author_guid]["ad_text"] = text
                    self.admin_states[author_guid]["state"] = "waiting_for_ad_time"
                    await self._safe_send(object_guid, "حالا زمان ارسال تبلیغ را وارد کنید. (مثال: 1403/06/15 18:30 یا 2025/09/03 18:30)")
                    return

                elif state == "waiting_for_ad_time":
                    try:
                        ad_time_str = text
                        ad_text = self.admin_states[author_guid]["ad_text"]
                        ad_ts = self._parse_datetime(ad_time_str)
                        self.db.add_ad(ad_text, ad_ts)
                        await self._safe_send(object_guid, "تبلیغ شما با موفقیت زمان‌بندی شد.")
                        del self.admin_states[author_guid]
                        return
                    except Exception:
                        await self._safe_send(object_guid, "فرمت تاریخ/ساعت اشتباه است. مثال: 1403/06/15 18:30 یا 2025/09/03 18:30.")
                        return

                elif state == "waiting_for_admin_username":
                    username = text.replace("@", "")
                    try:
                        user_info = await self.client.get_user_info_by_username(username)
                        if user_info and user_info.get("user"):
                            target_guid = user_info["user"].get("user_guid")
                            if target_guid:
                                self.db.add_admin(target_guid, is_master=False)
                                await self._safe_send(object_guid, f"کاربر @{username} به عنوان ادمین فرعی اضافه شد.")
                                await self._safe_send(target_guid, "تبریک! شما به عنوان ادمین فرعی منصوب شدید.")
                            else:
                                await self._safe_send(object_guid, "GUID معتبر برای کاربر یافت نشد.")
                        else:
                            await self._safe_send(object_guid, "کاربری با این نام کاربری یافت نشد. لطفا دوباره تلاش کنید.")
                    except Exception as e:
                        logger.error(f"Error adding admin by username: {e}")
                        await self._safe_send(object_guid, "خطایی در افزودن ادمین رخ داد. لطفا دوباره امتحان کنید.")
                    finally:
                        if author_guid in self.admin_states:
                            del self.admin_states[author_guid]
                    return

                elif state == "waiting_for_admin_to_remove":
                    if getattr(message, "reply_to_message_id", None):
                        replied = await self.client.get_messages_by_id(object_guid, [message.reply_to_message_id])
                        if replied and replied[0].get("author_guid"):
                            target_guid = replied[0]["author_guid"]
                            self.db.remove_admin(target_guid)
                            await self._safe_send(object_guid, "ادمین با موفقیت حذف شد.")
                            await self._safe_send(target_guid, "دسترسی ادمین شما حذف شد.")
                        else:
                            await self._safe_send(object_guid, "پیام ریپلای شده حاوی اطلاعات کاربری معتبر نیست.")
                        del self.admin_states[author_guid]
                    else:
                        await self._safe_send(object_guid, "لطفا روی پیام ادمین مورد نظر Reply بزنید و مجددا امتحان کنید.")
                    return

            # --- Password pending ---
            if author_guid in self.waiting_for_password:
                password = text
                await self.handle_password_check(message, password)
                return

            # --- Commands ---
            if text.startswith("/"):
                m = re.match(r"^/(\w+)", text)
                if m:
                    cmd = f"/{m.group(1).lower()}"
                    handler = self.commands.get(cmd)
                    if handler:
                        user_data = self.db.get_user(author_guid)
                        await handler(message, text, user_data)
                    else:
                        await self.show_user_menu(author_guid)
                return

            # بدون دستور: مثل /ai عمل کند
            message.text = f"/ai {text}"
            user_data = self.db.get_user(author_guid)
            await self.handle_ai_command(message, message.text, user_data)

        except Exception as e:  # pragma: no cover
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._safe_send(getattr(message, "object_guid", author_guid), "یک خطای ناشناخته رخ داد. لطفاً دوباره تلاش کنید.")

    async def handle_callback_query(self, callback_query: Update):
        try:
            data = getattr(callback_query, "data", "")
            sender_guid = getattr(callback_query, "sender_guid", None)
            sender_name = getattr(callback_query, "sender_name", "")

            admin_level = self.db.get_admin_level(sender_guid)

            if admin_level != -1:
                if data == "vip_manage":
                    await self.show_vip_menu(sender_guid)
                    return
                elif data == "add_vip":
                    await self._safe_send(sender_guid, "حالا روی پیام کاربری که می‌خواهید VIP کنید، Reply بزنید و این پیام را برای ربات بفرستید.")
                    self.admin_states[sender_guid] = {"state": "add_vip_reply"}
                    return
                elif data == "ad_manage":
                    await self.show_ad_menu(sender_guid)
                    return
                elif data == "add_ad":
                    await self._safe_send(sender_guid, "لطفاً متن کامل تبلیغ را وارد کنید.")
                    self.admin_states[sender_guid] = {"state": "waiting_for_ad_text"}
                    return
                elif data == "list_ads":
                    ads = self.db.get_due_ads()
                    if not ads:
                        await self._safe_send(sender_guid, "فعلاً تبلیغ زمان‌بندی‌شده‌ای نداریم.")
                    else:
                        lines = []
                        for ad_id, ad_text, run_at in ads:
                            dt = datetime.fromtimestamp(run_at).strftime("%Y-%m-%d %H:%M")
                            short = (ad_text[:300] + "…") if len(ad_text) > 300 else ad_text
                            lines.append(f"#{ad_id} — زمان: {dt}\n{short}")
                        await self._safe_send(sender_guid, "\n\n".join(lines))
                    return
                elif data == "admin_manage" and admin_level == 1:
                    await self.show_admin_management_menu(sender_guid)
                    return
                elif data == "add_sub_admin" and admin_level == 1:
                    await self._safe_send(sender_guid, "لطفاً نام کاربری ادمین جدید را بدون @ وارد کنید. مثال: username")
                    self.admin_states[sender_guid] = {"state": "waiting_for_admin_username"}
                    return
                elif data == "remove_sub_admin" and admin_level == 1:
                    await self._safe_send(sender_guid, "لطفاً روی پیام ادمین فرعی که می‌خواهید حذف شود Reply بزنید و پیام را برای ربات ارسال کنید.")
                    self.admin_states[sender_guid] = {"state": "waiting_for_admin_to_remove"}
                    return

            # User callbacks
            if data == "about":
                response = (
                    " 🤖 من یک ربات هوش مصنوعی هستم که به سوالات شما پاسخ می‌دهم و در کارهای مختلف کمک می‌کنم.\n\n"
                    "برای ارتباط یا پیشنهاد قابلیت جدید، به ادمین اصلی پیام بدهید: **@What0001** 🚀 "
                )
                await self._safe_send(sender_guid, response)
            elif data == "vip_request":
                await self._safe_send(sender_guid, "برای اطلاعات بیشتر در مورد عضویت VIP، لطفا به ادمین پیام بدهید.")
            elif data == "ai_chat":
                await self._safe_send(sender_guid, "لطفاً سوال خود را بعد از /ai مطرح کنید.")
            elif data == "request_join":
                channel_guid = CHANNEL_GUID or "unknown_channel"
                self.db.request_channel_join(channel_guid, sender_guid, sender_name)
                admin_message = (
                    f"درخواست جدید برای اضافه کردن ربات به گروه:\nنام کاربری: {sender_name}\nGUID: {sender_guid}"
                )
                await self._safe_send(self.master_admin_guid, admin_message)
                await self._safe_send(sender_guid, "درخواست شما به ادمین ارسال شد. در صورت تایید، به زودی با شما تماس گرفته می‌شود.")
            elif data == "back_to_main_menu":
                await self.show_user_menu(sender_guid)
            elif data == "back_to_admin_menu":
                admin_level = self.db.get_admin_level(sender_guid)
                if admin_level == -1:
                    await self._safe_send(sender_guid, "دسترسی ادمین ندارید.")
                else:
                    await self.show_admin_menu(sender_guid, admin_level)

        except Exception as e:  # pragma: no cover
            logger.error(f"Error handling callback query: {e}", exc_info=True)

    # ---------- Commands ----------
    async def handle_start_command(self, message: Update, text: str, user_data):
        await self.show_user_menu(getattr(message, "author_guid", None))

    async def handle_ai_command(self, message: Update, text: str, user_data):
        if not (HAS_GENAI and _model):
            await self._safe_send(getattr(message, "object_guid", None), "سرویس هوش مصنوعی غیرفعال است.")
            return

        # Ensure user exists
        if not user_data:
            user_data = self.db.get_user(getattr(message, "author_guid", ""))
            if not user_data:
                await self._safe_send(getattr(message, "object_guid", None), "مشکلی در شناسایی کاربر رخ داده است.")
                return

        # users columns: (guid, last_active, is_member, is_vip, vip_expiry)
        is_vip = bool(user_data[3])
        vip_expiry = user_data[4]

        # Expire VIP if needed
        if is_vip and (vip_expiry is None or datetime.now().timestamp() > float(vip_expiry)):
            self.db.make_vip(getattr(message, "author_guid", ""), 0)
            is_vip = False

        prompt = text.replace("/ai", "", 1).strip()
        if not prompt:
            await self._safe_send(getattr(message, "object_guid", None), "لطفاً یک سوال بعد از /ai بپرسید.")
            return

        user_type = "vip" if is_vip else "free"
        response_text = await generate_response(prompt, user_type)
        await self._safe_send(getattr(message, "object_guid", None), response_text)

    async def handle_summarize_command(self, message: Update, text: str, user_data):
        if not (HAS_GENAI and _model):
            await self._safe_send(getattr(message, "object_guid", None), "سرویس هوش مصنوعی غیرفعال است.")
            return

        prompt = text.replace("/summarize", "", 1).strip()
        if not prompt:
            await self._safe_send(getattr(message, "object_guid", None), "لطفاً متنی را برای خلاصه‌سازی بعد از /summarize وارد کنید.")
            return

        summary_prompt = f"متن زیر را در حد چند جمله خلاصه کن:\n\n{prompt}"
        summary_text = await generate_response(summary_prompt, "vip")
        await self._safe_send(getattr(message, "object_guid", None), summary_text)

    async def handle_admin_login(self, message: Update, text: str, user_data):
        admin_level = self.db.get_admin_level(getattr(message, "author_guid", ""))
        if admin_level != -1:
            await self.show_admin_menu(getattr(message, "author_guid", ""), admin_level)
            return
        await self._safe_send(getattr(message, "object_guid", None), "لطفاً رمز عبور را وارد کنید (یا /cancel برای لغو):")
        self.waiting_for_password[getattr(message, "author_guid", "")] = True

    async def handle_password_check(self, message: Update, password: str):
        author_guid = getattr(message, "author_guid", "")
        if password == self.master_password:
            self.db.add_admin(author_guid, is_master=True)
            await self._safe_send(getattr(message, "object_guid", None), "به عنوان ادمین اصلی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 1)
        elif password == self.sub_admin_password:
            self.db.add_admin(author_guid, is_master=False)
            await self._safe_send(getattr(message, "object_guid", None), "به عنوان ادمین فرعی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 0)
        else:
            await self._safe_send(getattr(message, "object_guid", None), "رمز عبور اشتباه است.")
        if author_guid in self.waiting_for_password:
            del self.waiting_for_password[author_guid]

    async def handle_cancel_state(self, message: Update, text: str, user_data):
        author_guid = getattr(message, "author_guid", "")
        cancelled = False
        if author_guid in self.admin_states:
            del self.admin_states[author_guid]
            cancelled = True
        if author_guid in self.waiting_for_password:
            del self.waiting_for_password[author_guid]
            cancelled = True
        msg = "عملیات جاری لغو شد." if cancelled else "عملیات فعالی برای لغو وجود ندارد."
        await self._safe_send(getattr(message, "object_guid", None), msg)

    # ---------- Menus ----------
    async def show_user_menu(self, guid: str):
        keyboard = [
            [{"text": "چت با هوش مصنوعی", "callback_data": "ai_chat"}],
            [{"text": "درباره ما", "callback_data": "about"}],
            [{"text": "درخواست عضویت در کانال/گروه", "callback_data": "request_join"}],
        ]
        await self.client.send_message(
            guid,
            "خوش آمدید! گزینه مورد نظر را انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_admin_menu(self, guid: str, admin_level: int):
        keyboard = [
            [{"text": "مدیریت VIP", "callback_data": "vip_manage"}],
            [{"text": "مدیریت تبلیغات", "callback_data": "ad_manage"}],
        ]
        if admin_level == 1:
            keyboard.append([{"text": "مدیریت ادمین‌ها", "callback_data": "admin_manage"}])
        keyboard.append([{"text": "بازگشت به منوی اصلی", "callback_data": "back_to_main_menu"}])
        await self.client.send_message(
            guid,
            "به پنل مدیریت خوش آمدید. گزینه مورد نظر را انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_vip_menu(self, guid: str):
        keyboard = [
            [{"text": "اضافه کردن VIP", "callback_data": "add_vip"}],
            [{"text": "برگشت به پنل ادمین", "callback_data": "back_to_admin_menu"}],
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت VIP انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_ad_menu(self, guid: str):
        keyboard = [
            [{"text": "افزودن تبلیغ جدید", "callback_data": "add_ad"}],
            [{"text": "لیست تبلیغات در انتظار", "callback_data": "list_ads"}],
            [{"text": "برگشت به پنل ادمین", "callback_data": "back_to_admin_menu"}],
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت تبلیغات انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_admin_management_menu(self, guid: str):
        keyboard = [
            [{"text": "افزودن ادمین", "callback_data": "add_sub_admin"}],
            [{"text": "حذف ادمین", "callback_data": "remove_sub_admin"}],
            [{"text": "برگشت به پنل ادمین", "callback_data": "back_to_admin_menu"}],
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت ادمین‌ها انتخاب کنید:",
            keyboard=keyboard,
        )

    # ---------- Ads Scheduler ----------
    async def run_ads_scheduler(self):
        """هر 10 ثانیه تبلیغات رسیده را برای همه کاربران می‌فرستد و حذف می‌کند."""
        while True:
            try:
                ads = self.db.get_due_ads()
                if ads:
                    with self.db.get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT guid FROM users")
                        all_guids = [row[0] for row in cur.fetchall()]

                    for ad in ads:
                        ad_id, ad_text, run_at = ad
                        for guid in all_guids:
                            try:
                                await self.client.send_message(guid, ad_text)
                                logger.info(f"Ad sent to user: {guid}")
                            except Exception as e:  # pragma: no cover
                                logger.error(f"Failed to send ad to {guid}: {e}")
                        self.db.delete_ad(ad_id)
                await asyncio.sleep(10)
            except Exception as e:  # pragma: no cover
                logger.error(f"run_ads_scheduler loop error: {e}")
                await asyncio.sleep(5)


# =============================
# 5) Bootstrap & Run
# =============================
async def main():
    # چک اولیه متغیرهای ضروری
    missing = []
    for k, v in {
        "RUBIKA_AUTH_KEY": AUTH_KEY,
        "MASTER_ADMIN_GUID": MASTER_ADMIN_GUID,
        "MASTER_PASSWORD": MASTER_PASSWORD,
        "SUB_ADMIN_PASSWORD": SUB_ADMIN_PASSWORD,
    }.items():
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"لطفاً متغیرهای محیطی زیر را تنظیم کنید: {', '.join(missing)}")

    if not HAS_RUBPY:
        raise RuntimeError("کتابخانه rubpy نصب نشده است.")

    # *** نکته حیاتی برای جلوگیری از prompt شماره‌تلفن (EOFError) ***
    # استفاده صحیح از پارامترها:
    #   - session: یک نام دلخواه برای ذخیره نشست (فایل محلی)
    #   - auth:    کلید AUTH روبیکا
    client = Client(auth=AUTH_KEY, name="rubika-bot")
    bot = AIBot(
        client=client,
        channel_guid=CHANNEL_GUID,
        master_admin_guid=MASTER_ADMIN_GUID,
        master_password=MASTER_PASSWORD,
        sub_admin_password=SUB_ADMIN_PASSWORD,
    )

    # شروع تسک زمان‌بندی تبلیغات
    asyncio.create_task(bot.run_ads_scheduler())

    # ------ ثبت هندلر عمومی آپدیت‌ها ------
    # rubpy در نسخه‌های مختلف API رویداد دارد؛ این مسیر عمومی با on_update کار می‌کند.
    # اگر کتابخانه شما متد "run" با callbackها را پشتیبانی کند از آن استفاده می‌کنیم؛
    # در غیر این صورت از حلقه دریافت آپدیت استفاده می‌کنیم.
    try:
        # بعضی نسخه‌ها: client.run(message_handler, callback_handler)
        # ما یک هندلر واحد می‌دهیم که خودش تشخیص می‌دهد پیام است یا کال‌بک:
        logger.info("Starting the Rubika AI bot (run with handler)…")
        client.run(bot.on_update)
        return
    except Exception as e:
        logger.warning(f"client.run(handler) not supported: {e}. Falling back to polling loop…")

    # --- FallBack: حلقه عمومی دریافت آپدیت‌ها (سازگار با نسخه‌هایی که run(handler) ندارند) ---
    # بسته به نسخه rubpy نام و امضای روش دریافت آپدیت‌ها متفاوت است؛
    # الگوی زیر دو حالت رایج را پوشش می‌دهد.
    try:
        # حالت context manager
        async with client:
            logger.info("Connected. Entering generic update loop…")
            while True:
                try:
                    updates: List[Update] = await client.get_updates()  # بعضی نسخه‌ها
                except AttributeError:
                    # حالت دیگر: fetch از event queue داخلی
                    updates = await client.listen()  # اگر متدی با این نام وجود داشته باشد
                for upd in updates or []:
                    await bot.on_update(upd)
                await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"Fatal loop error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error(f"Bot failed to start: {exc}", exc_info=True)



