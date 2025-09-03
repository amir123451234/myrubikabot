import asyncio
import logging
import os
import sqlite3
import re
import random
from datetime import datetime, timedelta
from rubpy import Client
from rubpy.types import Update
import google.generativeai as genai
from dotenv import load_dotenv

# --- بخش 1: تنظیمات و راه‌اندازی (Configuration & Setup) ---
load_dotenv()
AUTH_KEY = os.getenv("RUBIKA_AUTH_KEY")
MASTER_ADMIN_GUID = os.getenv("MASTER_ADMIN_GUID")
CHANNEL_GUID = os.getenv("CHANNEL_GUID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MASTER_PASSWORD = os.getenv("MASTER_PASSWORD")
SUB_ADMIN_PASSWORD = os.getenv("SUB_ADMIN_PASSWORD")
DB_PATH = "ai_bot_db.db"

genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIBot")

# --- بخش 2: مدل‌های هوش مصنوعی (AI Models) ---
async def generate_response(prompt: str, user_type: str) -> str:
    """Generates a response using Google Gemini API based on user type."""
    if not GEMINI_API_KEY:
        return "متاسفانه دسترسی به سرویس هوش مصنوعی امکان‌پذیر نیست."
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "در پاسخ به درخواست شما خطایی رخ داد."

# --- بخش 3: مدیریت دیتابیس (Database Manager) ---
class DBManager:
    """Manages all database interactions, including user, VIP, and admin data."""
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS users (
                guid TEXT PRIMARY KEY,
                last_active REAL,
                is_member INTEGER,
                is_vip INTEGER DEFAULT 0,
                vip_expiry REAL
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS admins (
                guid TEXT PRIMARY KEY,
                is_master INTEGER DEFAULT 0
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS ads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                run_at REAL
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS channel_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_guid TEXT,
                requester_guid TEXT,
                requester_name TEXT,
                status TEXT DEFAULT 'pending'
            )''')
            conn.commit()
    def update_user_activity(self, guid, is_member):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE guid = ?", (guid,))
            user = cursor.fetchone()
            timestamp = datetime.now().timestamp()
            if user is None:
                cursor.execute("INSERT INTO users (guid, last_active, is_member) VALUES (?, ?, ?)",
                               (guid, timestamp, int(is_member)))
            else:
                cursor.execute("UPDATE users SET last_active = ?, is_member = ? WHERE guid = ?",
                               (timestamp, int(is_member), guid))
            conn.commit()
    def get_user(self, guid):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE guid = ?", (guid,))
            return cursor.fetchone()
    def get_admin_level(self, guid):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_master FROM admins WHERE guid = ?", (guid,))
            result = cursor.fetchone()
            if result:
                return result[0]
            return -1
    def make_vip(self, guid, duration_days):
        expiry_date = datetime.now() + timedelta(days=duration_days)
        with self.get_connection() as conn:
            conn.execute("UPDATE users SET is_vip = 1, vip_expiry = ? WHERE guid = ?",
                         (expiry_date.timestamp(), guid))
            conn.commit()
    def add_admin(self, guid, is_master=False):
        with self.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO admins (guid, is_master) VALUES (?, ?)",
                         (guid, int(is_master)))
            conn.commit()
    def remove_admin(self, guid):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM admins WHERE guid = ?", (guid,))
            conn.commit()
    def get_sub_admins(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT guid FROM admins WHERE is_master = 0")
            return [row[0] for row in cursor.fetchall()]
    def get_active_ads(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ads WHERE run_at <= ? ORDER BY run_at DESC", (datetime.now().timestamp(),))
            return cursor.fetchall()
    def add_ad(self, ad_text, run_at):
        with self.get_connection() as conn:
            conn.execute("INSERT INTO ads (text, run_at) VALUES (?, ?)", (ad_text, run_at))
            conn.commit()
    def request_channel_join(self, channel_guid, requester_guid, requester_name):
        with self.get_connection() as conn:
            conn.execute("INSERT INTO channel_requests (channel_guid, requester_guid, requester_name) VALUES (?, ?, ?)",
                         (channel_guid, requester_guid, requester_name))
            conn.commit()
    def get_pending_requests(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM channel_requests WHERE status = 'pending'")
            return cursor.fetchall()
    def set_request_status(self, channel_guid, status):
        with self.get_connection() as conn:
            conn.execute("UPDATE channel_requests SET status = ? WHERE channel_guid = ?", (status, channel_guid))
            conn.commit()

# --- بخش 4: کلاس اصلی ربات (Main Bot Class) ---
class AIBot:
    def __init__(self, auth_key: str, channel_guid: str, master_admin_guid: str, master_password: str, sub_admin_password: str):
        if not all([auth_key, master_admin_guid, master_password, sub_admin_password]):
            raise ValueError("All required environment variables must be set.")
        self.client = Client(auth_key)
        self.channel_guid = channel_guid
        self.master_admin_guid = master_admin_guid
        self.master_password = master_password
        self.sub_admin_password = sub_admin_password
        self.db_manager = DBManager(DB_PATH)
        self.db_manager.add_admin(self.master_admin_guid, is_master=True)
        self.waiting_for_password = {}
        self.admin_states = {}
        self.commands = {
            '/start': self.handle_start_command,
            '/ai': self.handle_ai_command,
            '/summarize': self.handle_summarize_command,
            '/admin': self.handle_admin_login,
        }
        logger.info("AI bot handler initialized and ready.")

    async def handle_message(self, message: Update):
        try:
            author_guid = message.author_guid
            text = message.text

            # Check for admin states
            if author_guid in self.admin_states:
                state = self.admin_states[author_guid]['state']

                if state == 'add_vip_duration':
                    try:
                        duration_days = int(text.strip())
                        target_guid = self.admin_states[author_guid]['target_guid']
                        self.db_manager.make_vip(target_guid, duration_days)
                        await self.client.send_message(message.object_guid, f"کاربر با GUID `{target_guid}` برای {duration_days} روز VIP شد.")
                        await self.client.send_message(target_guid, f"تبریک! شما برای {duration_days} روز VIP شدید.")
                        del self.admin_states[author_guid]
                        return
                    except ValueError:
                        await self.client.send_message(message.object_guid, "لطفا یک عدد صحیح برای تعداد روزها وارد کنید.")
                        return

                elif state == 'add_vip_reply':
                    if message.reply_to_message_id:
                        replied_msg = await self.client.get_messages_by_id(message.object_guid, [message.reply_to_message_id])
                        target_guid = replied_msg[0]['author_guid']
                        self.admin_states[author_guid] = {'state': 'add_vip_duration', 'target_guid': target_guid}
                        await self.client.send_message(message.object_guid, f"لطفا تعداد روزهایی که کاربر باید VIP باشد را به صورت یک عدد وارد کنید.")
                    else:
                        await self.client.send_message(message.object_guid, "لطفا روی پیام کاربر مورد نظر ریپلای کنید.")
                    return

                elif state == 'waiting_for_ad_text':
                    self.admin_states[author_guid]['ad_text'] = text
                    self.admin_states[author_guid]['state'] = 'waiting_for_ad_time'
                    await self.client.send_message(message.object_guid, "حالا زمان ارسال تبلیغ را وارد کنید. (مثال: 1402/10/20 18:30)")
                    return

                elif state == 'waiting_for_ad_time':
                    try:
                        ad_time_str = text.strip()
                        ad_text = self.admin_states[author_guid]['ad_text']
                        ad_time = datetime.strptime(ad_time_str, '%Y/%m/%d %H:%M')
                        self.db_manager.add_ad(ad_text, ad_time.timestamp())
                        await self.client.send_message(message.object_guid, "تبلیغ شما با موفقیت زمان‌بندی شد.")
                        del self.admin_states[author_guid]
                        return
                    except ValueError:
                        await self.client.send_message(message.object_guid, "فرمت تاریخ و ساعت اشتباه است. لطفاً به این شکل وارد کنید: 1402/10/20 18:30")
                        return
                
                elif state == 'waiting_for_admin_username':
                    username = text.strip().replace('@', '')
                    try:
                        user_info = await self.client.get_user_info_by_username(username)
                        if user_info and user_info.get('user'):
                            target_guid = user_info['user']['user_guid']
                            self.db_manager.add_admin(target_guid, is_master=False)
                            await self.client.send_message(message.object_guid, f"کاربر @{username} به عنوان ادمین فرعی اضافه شد.")
                            await self.client.send_message(target_guid, "تبریک! شما به عنوان ادمین فرعی منصوب شدید.")
                        else:
                            await self.client.send_message(message.object_guid, "کاربری با این نام کاربری یافت نشد. لطفا دوباره تلاش کنید.")
                    except Exception as e:
                        logger.error(f"Error adding admin by username: {e}")
                        await self.client.send_message(message.object_guid, "خطایی در افزودن ادمین رخ داد. لطفا دوباره امتحان کنید.")
                    finally:
                        del self.admin_states[author_guid]
                    return

                elif state == 'waiting_for_admin_to_remove':
                    if message.reply_to_message_id:
                        replied_msg = await self.client.get_messages_by_id(message.object_guid, [message.reply_to_message_id])
                        if replied_msg and replied_msg[0].get('author_guid'):
                            target_guid = replied_msg[0]['author_guid']
                            self.db_manager.remove_admin(target_guid)
                            await self.client.send_message(message.object_guid, "ادمین با موفقیت حذف شد.")
                            await self.client.send_message(target_guid, "دسترسی ادمین شما حذف شد.")
                        else:
                            await self.client.send_message(message.object_guid, "پیام ریپلای شده حاوی اطلاعات کاربری معتبر نیست.")
                        del self.admin_states[author_guid]
                    else:
                        await self.client.send_message(message.object_guid, "لطفا روی پیام ادمین مورد نظر Reply بزنید و مجددا امتحان کنید.")
                    return
            
            # Handle password login state
            if author_guid in self.waiting_for_password:
                password = text.strip() if text else ""
                await self.handle_password_check(message, password)
                return
            
            # Check for commands
            if text and text.startswith('/'):
                command_match = re.match(r'^/(\w+)', text.strip())
                if command_match:
                    command = f"/{command_match.group(1).lower()}"
                    handler = self.commands.get(command)
                    if handler:
                        user_data = self.db_manager.get_user(author_guid)
                        await handler(message, text, user_data)
                    else:
                        await self.show_user_menu(author_guid)
            else:
                # Process message as an AI prompt if it's not a command.
                message.text = f"/ai {text}"
                user_data = self.db_manager.get_user(author_guid)
                await self.handle_ai_command(message, message.text, user_data)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self.client.send_message(message.object_guid, "یک خطای ناشناخته رخ داد. لطفا دوباره تلاش کنید.")

    async def handle_callback_query(self, callback_query: Update):
        try:
            data = callback_query.data
            sender_guid = callback_query.sender_guid
            sender_name = callback_query.sender_name
            
            # Check for admin callbacks
            admin_level = self.db_manager.get_admin_level(sender_guid)
            if admin_level != -1:
                if data == 'vip_manage':
                    await self.show_vip_menu(sender_guid)
                    return
                elif data == 'add_vip':
                    await self.client.send_message(sender_guid, "حالا روی پیام کاربری که می‌خواهید VIP کنید، Reply بزنید و این پیام را برای ربات بفرستید.")
                    self.admin_states[sender_guid] = {'state': 'add_vip_reply'}
                    return
                elif data == 'ad_manage':
                    await self.show_ad_menu(sender_guid)
                    return
                elif data == 'add_ad':
                    await self.client.send_message(sender_guid, "لطفاً متن کامل تبلیغ را وارد کنید.")
                    self.admin_states[sender_guid] = {'state': 'waiting_for_ad_text'}
                    return
                elif data == 'admin_manage' and admin_level == 1:
                    await self.show_admin_management_menu(sender_guid)
                    return
                elif data == 'add_sub_admin' and admin_level == 1:
                    await self.client.send_message(sender_guid, "لطفاً نام کاربری (یوزرنیم) ادمین جدید را بدون @ وارد کنید. مثال: username")
                    self.admin_states[sender_guid] = {'state': 'waiting_for_admin_username'}
                    return
                elif data == 'remove_sub_admin' and admin_level == 1:
                    await self.client.send_message(sender_guid, "لطفا روی پیام ادمین فرعی که می‌خواهید حذف کنید، Reply بزنید و این پیام را برای ربات ارسال کنید.")
                    self.admin_states[sender_guid] = {'state': 'waiting_for_admin_to_remove'}
                    return
            
            # Check for user callbacks
            if data == 'about':
                response = " 🤖  من یک ربات هوش مصنوعی هستم که به سوالات شما پاسخ می‌دهم و در کارهای مختلف به شما کمک می‌کنم.\n\nبرای ارتباط با من یا پیشنهاد قابلیت جدید، به ادمین اصلی پیام بدهید: **@What0001** 🚀 "
                await self.client.send_message(sender_guid, response)
            elif data == 'vip_request':
                response = "برای اطلاعات بیشتر در مورد عضویت VIP، لطفا به ادمین پیام بدهید."
                await self.client.send_message(sender_guid, response)
            elif data == 'ai_chat':
                response = "لطفاً سوال خود را بعد از /ai مطرح کنید."
                await self.client.send_message(sender_guid, response)
            elif data == 'request_join':
                channel_guid = 'unknown_channel'
                self.db_manager.request_channel_join(channel_guid, sender_guid, sender_name)
                admin_message = f"درخواست جدید برای اضافه کردن ربات به گروه:\nنام کاربری: {sender_name}\nGUID: {sender_guid}"
                await self.client.send_message(self.master_admin_guid, admin_message)
                await self.client.send_message(sender_guid, "درخواست شما به ادمین ارسال شد. در صورت تایید، به زودی با شما تماس گرفته می‌شود.")
            elif data == 'back_to_main_menu':
                await self.show_user_menu(sender_guid)
            elif data == 'back_to_admin_menu':
                await self.show_admin_menu(sender_guid, admin_level)
            
        except Exception as e:
            logger.error(f"Error handling callback query: {e}")

    # --- Command Handlers ---
    async def handle_start_command(self, message, text, user_data):
        await self.show_user_menu(message.author_guid)
    
    async def handle_ai_command(self, message, text, user_data):
        if not GEMINI_API_KEY:
            await self.client.send_message(message.object_guid, "سرویس هوش مصنوعی غیرفعال است.")
            return
        
        # Ensure user_data is not None before accessing its elements
        if not user_data:
            user_data = self.db_manager.get_user(message.author_guid)
            if not user_data:
                await self.client.send_message(message.object_guid, "مشکلی در شناسایی کاربر رخ داده است.")
                return
        is_vip = user_data[3]
        vip_expiry = user_data[4]
        
        if is_vip and datetime.now().timestamp() > vip_expiry:
            await self.client.send_message(message.object_guid, "عضویت VIP شما منقضی شده است. لطفا برای تمدید آن اقدام کنید.")
            is_vip = False
        
        prompt = text.replace("/ai", "", 1).strip()
        if not prompt:
            await self.client.send_message(message.object_guid, "لطفا یک سوال بعد از /ai بپرسید.")
            return
            
        user_type = 'vip' if is_vip else 'free'
        response_text = await generate_response(prompt, user_type)
        await self.client.send_message(message.object_guid, response_text)
    
    async def handle_summarize_command(self, message, text, user_data):
        if not GEMINI_API_KEY:
            await self.client.send_message(message.object_guid, "سرویس هوش مصنوعی غیرفعال است.")
            return
            
        prompt = text.replace("/summarize", "", 1).strip()
        if not prompt:
            await self.client.send_message(message.object_guid, "لطفا متنی را برای خلاصه‌سازی بعد از /summarize وارد کنید.")
            return
            
        summary_prompt = f"متن زیر را در حد چند جمله خلاصه کن:\n\n{prompt}"
        summary_text = await generate_response(summary_prompt, "vip")
        await self.client.send_message(message.object_guid, summary_text)
    
    async def handle_admin_login(self, message, text, user_data):
        admin_level = self.db_manager.get_admin_level(message.author_guid)
        if admin_level != -1:
            await self.show_admin_menu(message.author_guid, admin_level)
            return
            
        await self.client.send_message(message.object_guid, "لطفاً رمز عبور را وارد کنید:")
        self.waiting_for_password[message.author_guid] = True
    
    async def handle_password_check(self, message, password):
        author_guid = message.author_guid
        if password == self.master_password:
            self.db_manager.add_admin(author_guid, is_master=True)
            await self.client.send_message(message.object_guid, "به عنوان ادمین اصلی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 1)
        elif password == self.sub_admin_password:
            self.db_manager.add_admin(author_guid, is_master=False)
            await self.client.send_message(message.object_guid, "به عنوان ادمین فرعی وارد شدید. خوش آمدید!")
            await self.show_admin_menu(author_guid, 0)
        else:
            await self.client.send_message(message.object_guid, "رمز عبور اشتباه است.")
        del self.waiting_for_password[author_guid]

    async def show_user_menu(self, guid):
        keyboard = [
            [{'text': 'چت با هوش مصنوعی', 'callback_data': 'ai_chat'}],
            [{'text': 'درباره ما', 'callback_data': 'about'}],
            [{'text': 'درخواست عضویت در کانال/گروه', 'callback_data': 'request_join'}]
        ]
        await self.client.send_message(
            guid,
            "خوش آمدید! گزینه مورد نظر را انتخاب کنید:",
            keyboard=keyboard
        )
        
    async def show_admin_menu(self, guid, admin_level):
        keyboard = [
            [{'text': 'مدیریت VIP', 'callback_data': 'vip_manage'}],
            [{'text': 'مدیریت تبلیغات', 'callback_data': 'ad_manage'}]
        ]
        if admin_level == 1:
            keyboard.append([{'text': 'مدیریت ادمین‌ها', 'callback_data': 'admin_manage'}])
        await self.client.send_message(
            guid,
            "به پنل مدیریت خوش آمدید. گزینه مورد نظر را انتخاب کنید:",
            keyboard=keyboard
        )
        
    async def show_vip_menu(self, guid):
        keyboard = [
            [{'text': 'اضافه کردن VIP', 'callback_data': 'add_vip'}],
            [{'text': 'برگشت به پنل ادمین', 'callback_data': 'back_to_admin_menu'}]
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت VIP انتخاب کنید:",
            keyboard=keyboard
        )
        
    async def show_ad_menu(self, guid):
        keyboard = [
            [{'text': 'افزودن تبلیغ جدید', 'callback_data': 'add_ad'}],
            [{'text': 'لیست تبلیغات در انتظار', 'callback_data': 'list_ads'}],
            [{'text': 'برگشت به پنل ادمین', 'callback_data': 'back_to_admin_menu'}]
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت تبلیغات انتخاب کنید:",
            keyboard=keyboard
        )
        
    async def show_admin_management_menu(self, guid):
        keyboard = [
            [{'text': 'افزودن ادمین', 'callback_data': 'add_sub_admin'}],
            [{'text': 'حذف ادمین', 'callback_data': 'remove_sub_admin'}],
            [{'text': 'برگشت به پنل ادمین', 'callback_data': 'back_to_admin_menu'}]
        ]
        await self.client.send_message(
            guid,
            "گزینه مورد نظر را برای مدیریت ادمین‌ها انتخاب کنید:",
            keyboard=keyboard
        )

    # --- Main Loop ---
    async def run_ads_scheduler(self):
        while True:
            ads = self.db_manager.get_active_ads()
            for ad in ads:
                ad_id, ad_text, run_at = ad
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT guid FROM users")
                    all_guids = [row[0] for row in cursor.fetchall()]
                for guid in all_guids:
                    try:
                        await self.client.send_message(guid, ad_text)
                        logger.info(f"Ad sent to user: {guid}")
                    except Exception as e:
                        logger.error(f"Failed to send ad to {guid}: {e}")
                with self.db_manager.get_connection() as conn:
                    conn.execute("DELETE FROM ads WHERE id = ?", (ad_id,))
                    conn.commit()
            await asyncio.sleep(10)

    async def run(self):
        try:
            logger.info("Starting the Rubika AI bot...")
            asyncio.create_task(self.run_ads_scheduler())
            await self.client.run(self.handle_message, self.handle_callback_query)
        except Exception as e:
            logger.error(f"Bot failed to start: {e}", exc_info=True)

# --- بخش 5: نقطه ورود برنامه (Main Entry Point) ---
if __name__ == "__main__":
    if not all([AUTH_KEY, MASTER_ADMIN_GUID, MASTER_PASSWORD, SUB_ADMIN_PASSWORD]):
        logger.error("All required environment variables must be set in the .env file.")
    else:
        try:
            bot = AIBot(AUTH_KEY, None, MASTER_ADMIN_GUID, MASTER_PASSWORD, SUB_ADMIN_PASSWORD)
            asyncio.run(bot.run())
        except Exception as e:
            logger.error(f"An error occurred during bot execution: {e}", exc_info=True)
