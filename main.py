import asyncio
import logging
import os
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────
# Persian (Jalali) date parsing (اختیاری)
# ─────────────────────────────────────────────────────────
try:
    from persiantools.jdatetime import JalaliDateTime  # type: ignore
    HAS_PERSIAN_DATE = True
except Exception:
    HAS_PERSIAN_DATE = False

# ─────────────────────────────────────────────────────────
# Rubika SDK (rubpy)
# ─────────────────────────────────────────────────────────
try:
    from rubpy import Client, handlers  # type: ignore
    HAS_RUBPY = True
except Exception:
    HAS_RUBPY = False
    Client = object  # type: ignore
    handlers = None  # type: ignore

# ─────────────────────────────────────────────────────────
# Google Generative AI (Gemini)
# ─────────────────────────────────────────────────────────
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# =============================
# 1) Configuration & Logging
# =============================
load_dotenv()
AUTH_KEY = os.getenv("RUBIKA_AUTH_KEY")
MASTER_ADMIN_GUID = os.getenv("MASTER_ADMIN_GUID")
CHANNEL_GUID = os.getenv("CHANNEL_GUID")  # اختیاری
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # پیش‌فرض امن‌تر از gemini-pro
MASTER_PASSWORD = os.getenv("MASTER_PASSWORD")
SUB_ADMIN_PASSWORD = os.getenv("SUB_ADMIN_PASSWORD")
DB_PATH = os.getenv("DB_PATH", "ai_bot_db.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AIBot")

# Configure Gemini (اگر موجود باشد)
_model = None
if HAS_GENAI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # پیکربندی تولید
        GENERATION_CONFIG = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        _model = genai.GenerativeModel(GEMINI_MODEL, generation_config=GENERATION_CONFIG)
        logger.info("Gemini model initialized: %s", GEMINI_MODEL)
    except Exception as e:
        logger.error("Failed to init Gemini: %s", e)
        _model = None


# =============================
# 2) AI helper
# =============================
async def generate_response(prompt: str, user_type: str) -> str:
    """Generate a response using Google Gemini API. Falls back gracefully."""
    if not (HAS_GENAI and _model):
        return "متاسفانه دسترسی به سرویس هوش مصنوعی امکان‌پذیر نیست."

    try:
        system_preamble = (
            "You are a helpful Persian assistant. پاسخ‌ها را مودبانه، دقیق و کاربردی بده."
        )
        if user_type == "free":
            final_prompt = f"{system_preamble}\n\nحداکثر در 8 خط پاسخ بده.\n\nسوال کاربر:\n{prompt}"
        else:
            final_prompt = f"{system_preamble}\n\nسوال کاربر:\n{prompt}"

        # generate_content همگام است؛ برای بلاک‌نکردن لوپ، در ترد اجراش می‌کنیم
        response = await asyncio.to_thread(_model.generate_content, final_prompt)
        text = getattr(response, "text", "").strip()
        return text or "پاسخی از موتور هوش مصنوعی دریافت نشد."
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "در پاسخ به درخواست شما خطایی رخ داد."


# =============================
# 3) Database Manager (SQLite)
# =============================
class DBManager:
    """Manages DB interactions: users, admins, ads, channel requests."""

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
        """Upsert VIP; اگر duration_days == 0، VIP را حذف می‌کند."""
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
    def get_admin_level(self, guid: Optional[str]) -> int:
        if not guid:
            return -1
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
    def get_due_ads(self) -> List[Tuple[int, str, float]]:
        """تبلیغاتی که موعدشان رسیده را برمی‌گرداند."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, text, run_at FROM ads WHERE run_at <= ? ORDER BY run_at ASC",
                (datetime.now().timestamp(),),
            )
            return [(row[0], row[1], float(row[2])) for row in cur.fetchall()]

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
# 4) Helper: normalize update
# =============================
class SimpleUpdate:
    """یک آبجکت سبک برای یکسان‌سازی فیلدهای rubpy update."""

    def __init__(
        self,
        author_guid: Optional[str],
        object_guid: Optional[str],
        text: str,
        reply_to_message_id: Optional[str] = None,
        sender_name: str = "",
        data: str = "",
    ):
        self.author_guid = author_guid
        self.object_guid = object_guid
        self.text = text
        self.reply_to_message_id = reply_to_message_id
        self.sender_name = sender_name
        self.data = data  # برای callback_data


def pick(obj, names: List[str], default=None):
    """به ترتیب از بین چند اسم، اولین مقدار در دسترس را برمی‌دارد."""
    for n in names:
        try:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
            # اگر شبیه dict بود
            if hasattr(obj, "get"):
                v2 = obj.get(n)  # type: ignore
                if v2 is not None:
                    return v2
        except Exception:
            continue
    return default


def to_simple_update_from_message(update) -> SimpleUpdate:
    author_guid = pick(update, ["author_guid", "sender_guid", "user_guid"])
    object_guid = pick(update, ["object_guid", "chat_guid", "guid"])
    text = pick(update, ["text", "raw_text", "message", "caption"], "") or ""
    reply_to = pick(update, ["reply_to_message_id", "reply_message_id", "reply_to_message"])
    # برخی پیاده‌سازی‌ها اسم فرستنده رو دارند
    sender_name = pick(update, ["author_title", "sender_name", "author_name", "full_name"], "")
    return SimpleUpdate(author_guid, object_guid, text, reply_to, sender_name)


def to_simple_update_from_callback(update) -> SimpleUpdate:
    sender_guid = pick(update, ["sender_guid", "author_guid", "user_guid"])
    object_guid = pick(update, ["object_guid", "chat_guid", "guid"])
    data = pick(update, ["data", "callback_data", "raw_text"], "") or ""
    sender_name = pick(update, ["sender_name", "author_title", "author_name", "full_name"], "")
    return SimpleUpdate(sender_guid, object_guid, text="", reply_to_message_id=None, sender_name=sender_name, data=data)


# =============================
# 5) Main Bot
# =============================
class AIBot:
    def __init__(
        self,
        auth_key: str,
        channel_guid: Optional[str],
        master_admin_guid: str,
        master_password: str,
        sub_admin_password: str,
    ):
        if not HAS_RUBPY:
            raise RuntimeError("rubpy در محیط نصب نشده است.")
        if not all([auth_key, master_admin_guid, master_password, sub_admin_password]):
            raise ValueError("تمام متغیرهای محیطی لازم باید تنظیم شوند.")

        self.auth_key = auth_key
        self.client: Optional[Client] = None
        self.channel_guid = channel_guid
        self.master_admin_guid = master_admin_guid
        self.master_password = master_password
        self.sub_admin_password = sub_admin_password

        self.db_manager = DBManager(DB_PATH)
        self.db_manager.add_admin(self.master_admin_guid, is_master=True)

        # state stores
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

    # ───────────── Utilities ─────────────
    async def _safe_send(self, target_guid: Optional[str], text: str):
        if not target_guid or not self.client:
            return
        try:
            MAX_LEN = 4000
            if len(text) <= MAX_LEN:
                await self.client.send_message(target_guid, text)
                return
            # Split to safe chunks
            parts: List[str] = []
            buf: List[str] = []
            size = 0
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
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def _parse_datetime(self, text: str) -> float:
        """Parse datetime string. Jalali if available, else Gregorian.
        Formats:
          - 1403/06/15 18:30 (Jalali)
          - 2025/09/03 18:30 (Gregorian)
        """
        text = text.strip()
        if HAS_PERSIAN_DATE:
            try:
                jdt = JalaliDateTime.strptime(text, "%Y/%m/%d %H:%M")
                return jdt.to_gregorian().timestamp()
            except Exception:
                pass
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M")
        return dt.timestamp()

    # ───────────── Event Handlers ─────────────
    async def handle_message(self, su: SimpleUpdate):
        """هندل پیام‌های متنی و تبدیل به دستور/چت AI"""
        try:
            author_guid = su.author_guid
            text = su.text or ""
            object_guid = su.object_guid or author_guid

            # ثبت فعالیت کاربر
            try:
                if author_guid:
                    self.db_manager.update_user_activity(author_guid, True)
            except Exception as e:
                logger.error(f"Failed to update user activity: {e}")

            # --- Admin state machine ---
            if author_guid in self.admin_states:
                state = self.admin_states[author_guid].get("state")

                if state == "add_vip_duration":
                    try:
                        duration_days = int(text.strip())
                        target_guid = self.admin_states[author_guid]["target_guid"]
                        self.db_manager.make_vip(target_guid, duration_days)
                        await self._safe_send(object_guid, f"کاربر با GUID `{target_guid}` برای {duration_days} روز VIP شد.")
                        await self._safe_send(target_guid, f"تبریک! شما برای {duration_days} روز VIP شدید.")
                        del self.admin_states[author_guid]
                        return
                    except ValueError:
                        await self._safe_send(object_guid, "لطفا یک عدد صحیح برای تعداد روزها وارد کنید.")
                        return

                elif state == "add_vip_reply":
                    if su.reply_to_message_id and self.client:
                        try:
                            msgs = await self.client.get_messages_by_id(object_guid, [su.reply_to_message_id])
                            if msgs and isinstance(msgs, list):
                                # تلاش برای استخراج guid از پیام ریپلای‌شده
                                target_guid = None
                                msg0 = msgs[0]
                                target_guid = pick(msg0, ["author_guid", "sender_guid", "user_guid"])
                                if target_guid:
                                    self.admin_states[author_guid] = {
                                        "state": "add_vip_duration",
                                        "target_guid": target_guid,
                                    }
                                    await self._safe_send(object_guid, "لطفا تعداد روزهای VIP را به صورت عدد وارد کنید.")
                                else:
                                    await self._safe_send(object_guid, "GUID معتبر برای کاربر در پیام ریپلای‌شده پیدا نشد.")
                            else:
                                await self._safe_send(object_guid, "پیام ریپلای‌شده معتبر نیست.")
                        except Exception as e:
                            logger.error(f"get_messages_by_id error: {e}")
                            await self._safe_send(object_guid, "خواندن پیام ریپلای‌شده با خطا مواجه شد.")
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
                        ad_time_str = text.strip()
                        ad_text = self.admin_states[author_guid]["ad_text"]
                        ad_ts = self._parse_datetime(ad_time_str)
                        self.db_manager.add_ad(ad_text, ad_ts)
                        await self._safe_send(object_guid, "تبلیغ شما با موفقیت زمان‌بندی شد.")
                        del self.admin_states[author_guid]
                        return
                    except Exception:
                        await self._safe_send(object_guid, "فرمت تاریخ/ساعت اشتباه است. مثال: 1403/06/15 18:30 یا 2025/09/03 18:30.")
                        return

                elif state == "waiting_for_admin_username":
                    username = text.strip().replace("@", "")
                    if not self.client:
                        await self._safe_send(object_guid, "کلاینت آماده نیست.")
                        del self.admin_states[author_guid]
                        return
                    try:
                        user_info = await self.client.get_user_info_by_username(username)
                        # ساختار دقیق rubpy ممکن است متفاوت باشد؛ محافظه‌کارانه استخراج می‌کنیم
                        target_guid = None
                        if user_info:
                            target_guid = pick(user_info, ["user_guid", "guid"])
                            if not target_guid:
                                u = user_info.get("user") if hasattr(user_info, "get") else None  # type: ignore
                                target_guid = pick(u or {}, ["user_guid", "guid"])
                        if target_guid:
                            self.db_manager.add_admin(target_guid, is_master=False)
                            await self._safe_send(object_guid, f"کاربر @{username} به عنوان ادمین فرعی اضافه شد.")
                            await self._safe_send(target_guid, "تبریک! شما به عنوان ادمین فرعی منصوب شدید.")
                        else:
                            await self._safe_send(object_guid, "کاربری با این یوزرنیم پیدا نشد یا GUID نامشخص بود.")
                    except Exception as e:
                        logger.error(f"Error adding admin by username: {e}")
                        await self._safe_send(object_guid, "خطایی در افزودن ادمین رخ داد. دوباره امتحان کنید.")
                    finally:
                        if author_guid in self.admin_states:
                            del self.admin_states[author_guid]
                    return

                elif state == "waiting_for_admin_to_remove":
                    if su.reply_to_message_id and self.client:
                        try:
                            msgs = await self.client.get_messages_by_id(object_guid, [su.reply_to_message_id])
                            if msgs and isinstance(msgs, list):
                                target_guid = pick(msgs[0], ["author_guid", "sender_guid", "user_guid"])
                                if target_guid:
                                    self.db_manager.remove_admin(target_guid)
                                    await self._safe_send(object_guid, "ادمین با موفقیت حذف شد.")
                                    await self._safe_send(target_guid, "دسترسی ادمین شما حذف شد.")
                                else:
                                    await self._safe_send(object_guid, "GUID معتبر در پیام ریپلای‌شده یافت نشد.")
                            else:
                                await self._safe_send(object_guid, "پیام ریپلای‌شده معتبر نیست.")
                        except Exception as e:
                            logger.error(f"Error removing admin: {e}")
                            await self._safe_send(object_guid, "حذف ادمین با خطا مواجه شد.")
                        finally:
                            if author_guid in self.admin_states:
                                del self.admin_states[author_guid]
                    else:
                        await self._safe_send(object_guid, "لطفاً روی پیام ادمین مورد نظر Reply بزنید و دوباره ارسال کنید.")
                    return

            # --- Waiting for password state ---
            if author_guid in self.waiting_for_password:
                await self.handle_password_check(su, text.strip())
                return

            # --- Commands ---
            if text and text.startswith("/"):
                m = re.match(r"^/(\w+)", text.strip())
                if m:
                    cmd = f"/{m.group(1).lower()}"
                    handler = self.commands.get(cmd)
                    user_data = self.db_manager.get_user(author_guid or "")
                    if handler:
                        await handler(su, text, user_data)
                    else:
                        await self.show_user_menu(author_guid)
                return

            # اگر دستور نبود → به عنوان /ai
            su.text = f"/ai {text}"
            user_data = self.db_manager.get_user(author_guid or "")
            await self.handle_ai_command(su, su.text, user_data)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._safe_send(su.object_guid or su.author_guid, "یک خطای ناشناخته رخ داد. لطفا دوباره تلاش کنید.")

    async def handle_callback_query(self, su: SimpleUpdate):
        """هندل کلیک روی دکمه‌های اینلاین"""
        try:
            data = su.data
            sender_guid = su.author_guid
            sender_name = su.sender_name or ""

            admin_level = self.db_manager.get_admin_level(sender_guid)

            if admin_level != -1:
                if data == "vip_manage":
                    await self.show_vip_menu(sender_guid)
                    return
                elif data == "add_vip":
                    await self._safe_send(sender_guid, "حالا روی پیام کاربری که می‌خواهید VIP کنید، Reply بزنید و این پیام را بفرستید.")
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
                    ads = self.db_manager.get_due_ads()
                    if not ads:
                        await self._safe_send(sender_guid, "فعلاً تبلیغ موعددار نداریم.")
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
                    await self._safe_send(sender_guid, "لطفاً یوزرنیم ادمین جدید را بدون @ وارد کنید. مثال: username")
                    self.admin_states[sender_guid] = {"state": "waiting_for_admin_username"}
                    return
                elif data == "remove_sub_admin" and admin_level == 1:
                    await self._safe_send(sender_guid, "روی پیام ادمین فرعی که می‌خواهید حذف کنید، Reply بزنید و ارسال کنید.")
                    self.admin_states[sender_guid] = {"state": "waiting_for_admin_to_remove"}
                    return

            # User callbacks
            if data == "about":
                response = (
                    "🤖 من یک ربات هوش مصنوعی هستم که به سوالات شما پاسخ می‌دهم و در کارهای مختلف کمک می‌کنم.\n\n"
                    "برای ارتباط یا پیشنهاد قابلیت جدید، به ادمین اصلی پیام بدهید: **@What0001** 🚀"
                )
                await self._safe_send(sender_guid, response)
            elif data == "vip_request":
                await self._safe_send(sender_guid, "برای اطلاعات بیشتر در مورد عضویت VIP، لطفا به ادمین پیام بدهید.")
            elif data == "ai_chat":
                await self._safe_send(sender_guid, "لطفاً سوال خود را بعد از /ai مطرح کنید.")
            elif data == "request_join":
                channel_guid = CHANNEL_GUID or "unknown_channel"
                self.db_manager.request_channel_join(channel_guid, sender_guid or "", sender_name)
                admin_message = (
                    f"درخواست جدید برای اضافه کردن ربات به گروه:\n"
                    f"نام کاربری/نام: {sender_name}\nGUID: {sender_guid}"
                )
                await self._safe_send(self.master_admin_guid, admin_message)
                await self._safe_send(sender_guid, "درخواست شما به ادمین ارسال شد. در صورت تایید، به زودی با شما تماس گرفته می‌شود.")
            elif data == "back_to_main_menu":
                await self.show_user_menu(sender_guid)
            elif data == "back_to_admin_menu":
                admin_level = self.db_manager.get_admin_level(sender_guid)
                if admin_level == -1:
                    await self._safe_send(sender_guid, "دسترسی ادمین ندارید.")
                else:
                    await self.show_admin_menu(sender_guid, admin_level)

        except Exception as e:
            logger.error(f"Error handling callback query: {e}")

    # ───────────── Command Handlers ─────────────
    async def handle_start_command(self, su: SimpleUpdate, text: str, user_data):
        await self.show_user_menu(su.author_guid)

    async def handle_ai_command(self, su: SimpleUpdate, text: str, user_data):
        if not (HAS_GENAI and _model):
            await self._safe_send(su.object_guid, "سرویس هوش مصنوعی غیرفعال است.")
            return

        author_guid = su.author_guid or ""
        if not user_data:
            user_data = self.db_manager.get_user(author_guid)
            if not user_data:
                await self._safe_send(su.object_guid, "مشکلی در شناسایی کاربر رخ داده است.")
                return

        # users columns: (guid, last_active, is_member, is_vip, vip_expiry)
        is_vip = bool(user_data[3])
        vip_expiry = user_data[4]

        # Expire VIP if needed
        if is_vip and (vip_expiry is None or datetime.now().timestamp() > float(vip_expiry)):
            self.db_manager.make_vip(author_guid, 0)
            is_vip = False

        prompt = text.replace("/ai", "", 1).strip()
        if not prompt:
            await self._safe_send(su.object_guid, "لطفا یک سوال بعد از /ai بپرسید.")
            return

        user_type = "vip" if is_vip else "free"
        response_text = await generate_response(prompt, user_type)
        await self._safe_send(su.object_guid, response_text)

    async def handle_summarize_command(self, su: SimpleUpdate, text: str, user_data):
        if not (HAS_GENAI and _model):
            await self._safe_send(su.object_guid, "سرویس هوش مصنوعی غیرفعال است.")
            return

        prompt = text.replace("/summarize", "", 1).strip()
        if not prompt:
            await self._safe_send(su.object_guid, "لطفا متنی را برای خلاصه‌سازی بعد از /summarize وارد کنید.")
            return

        summary_prompt = f"متن زیر را در حد چند جمله خلاصه کن:\n\n{prompt}"
        summary_text = await generate_response(summary_prompt, "vip")
        await self._safe_send(su.object_guid, summary_text)

    async def handle_admin_login(self, su: SimpleUpdate, text: str, user_data):
        admin_level = self.db_manager.get_admin_level(su.author_guid)
        if admin_level != -1:
            await self.show_admin_menu(su.author_guid, admin_level)
            return
        await self._safe_send(su.object_guid, "لطفاً رمز عبور را وارد کنید (یا /cancel برای لغو):")
        if su.author_guid:
            self.waiting_for_password[su.author_guid] = True

    async def handle_password_check(self, su: SimpleUpdate, password: str):
        author_guid = su.author_guid or ""
        if password == self.master_password:
            self.db_manager.add_admin(author_guid, is_master=True)
            await self._safe_send(su.object_guid, "به عنوان ادمین اصلی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 1)
        elif password == self.sub_admin_password:
            self.db_manager.add_admin(author_guid, is_master=False)
            await self._safe_send(su.object_guid, "به عنوان ادمین فرعی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 0)
        else:
            await self._safe_send(su.object_guid, "رمز عبور اشتباه است.")
        if author_guid in self.waiting_for_password:
            del self.waiting_for_password[author_guid]

    async def handle_cancel_state(self, su: SimpleUpdate, text: str, user_data):
        author_guid = su.author_guid or ""
        cancelled = False
        if author_guid in self.admin_states:
            del self.admin_states[author_guid]
            cancelled = True
        if author_guid in self.waiting_for_password:
            del self.waiting_for_password[author_guid]
            cancelled = True
        msg = "عملیات جاری لغو شد." if cancelled else "عملیات فعالی برای لغو وجود ندارد."
        await self._safe_send(su.object_guid, msg)

    # ───────────── Menus ─────────────
    async def show_user_menu(self, guid: Optional[str]):
        if not (guid and self.client):
            return
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

    async def show_admin_menu(self, guid: Optional[str], admin_level: int):
        if not (guid and self.client):
            return
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

    async def show_vip_menu(self, guid: Optional[str]):
        if not (guid and self.client):
            return
        keyboard = [
            [{"text": "اضافه کردن VIP", "callback_data": "add_vip"}],
            [{"text": "برگشت به پنل ادمین", "callback_data": "back_to_admin_menu"}],
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت VIP انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_ad_menu(self, guid: Optional[str]):
        if not (guid and self.client):
            return
        keyboard = [
            [{"text": "افزودن تبلیغ جدید", "callback_data": "add_ad"}],
            [{"text": "لیست تبلیغات موعددار", "callback_data": "list_ads"}],
            [{"text": "برگشت به پنل ادمین", "callback_data": "back_to_admin_menu"}],
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت تبلیغات انتخاب کنید:",
            keyboard=keyboard,
        )

    async def show_admin_management_menu(self, guid: Optional[str]):
        if not (guid and self.client):
            return
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

    # ───────────── Ads Scheduler ─────────────
    async def run_ads_scheduler(self):
        """هر ۱۰ ثانیه چک می‌کند تبلیغ موعددار داریم یا نه؛ سپس برای همه کاربران می‌فرستد و حذف می‌کند."""
        while True:
            try:
                if not self.client:
                    await asyncio.sleep(2)
                    continue

                due_ads = self.db_manager.get_due_ads()
                if due_ads:
                    # همه کاربران
                    with self.db_manager.get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT guid FROM users")
                        all_guids = [row[0] for row in cur.fetchall()]

                    for ad_id, ad_text, _run_at in due_ads:
                        for guid in all_guids:
                            try:
                                await self.client.send_message(guid, ad_text)
                                logger.info(f"Ad sent to user: {guid}")
                            except Exception as e:
                                logger.error(f"Failed to send ad to {guid}: {e}")
                        self.db_manager.delete_ad(ad_id)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"run_ads_scheduler loop error: {e}")
                await asyncio.sleep(5)

    # ───────────── Bootstrapping (async) ─────────────
    async def start(self):
        """کلاینت rubpy را استارت می‌کند، هندلرها را رجیستر می‌کند و تا قطع شدن اتصال اجرا می‌ماند."""
        logger.info("Starting the Rubika AI bot...")

        # rubpy الگوی درست: async context + on(...) + run_until_disconnected()
        async with Client(self.auth_key) as client:
            self.client = client

            # رجیستر هندلر پیام‌ها
            @client.on(handlers.MessageUpdates())
            async def _on_message(update):
                su = to_simple_update_from_message(update)
                # اگر object_guid خالی بود، حداقل author_guid را استفاده می‌کنیم
                if not su.object_guid:
                    su.object_guid = su.author_guid
                await self.handle_message(su)

            # رجیستر هندلر کال‌بک دکمه‌ها
            @client.on(handlers.CallbackQueryUpdates())
            async def _on_callback(update):
                su = to_simple_update_from_callback(update)
                await self.handle_callback_query(su)

            # استارت زمان‌بند تبلیغات در همین event loop
            asyncio.create_task(self.run_ads_scheduler())

            # بلوک تا قطع اتصال
            await client.run_until_disconnected()


# =============================
# 6) Entry Point
# =============================
if __name__ == "__main__":
    if not HAS_RUBPY:
        logger.error("کتابخانه rubpy نصب نشده است. لطفاً آن را به requirements.txt اضافه/نصب کنید.")
    elif not all([AUTH_KEY, MASTER_ADMIN_GUID, MASTER_PASSWORD, SUB_ADMIN_PASSWORD]):
        logger.error("تمام متغیرهای محیطی لازم باید در env. تنظیم شوند.")
    else:
        try:
            bot = AIBot(
                AUTH_KEY,
                CHANNEL_GUID,  # اگر لازم نیست، در env نگذارید
                MASTER_ADMIN_GUID,
                MASTER_PASSWORD,
                SUB_ADMIN_PASSWORD,
            )
            asyncio.run(bot.start())
        except Exception as e:
            logger.error(f"An error occurred during bot execution: {e}", exc_info=True)
