import psutil
import time
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process_monitor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# 邮件配置（从环境变量中获取）
SMTP_SERVER = os.getenv("SMTP_SERVER")  # 例如 'smtp.gmail.com'
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))  # 例如 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # 发送邮件的邮箱地址
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # 发送邮件的邮箱密码或应用专用密码
TO_EMAIL = os.getenv("TO_EMAIL")  # 接收通知的邮箱地址

# 进程配置
MONITORED_PID = int(os.getenv("MONITORED_PID", 0))  # 需要监控的进程ID
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 10))  # 检查间隔（秒）


def send_email(subject, body, retries=3, delay=5):
    """发送电子邮件，并在失败时重试"""
    attempt = 0
    while attempt < retries:
        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = TO_EMAIL
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            # 连接SMTP服务器
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()  # 使用TLS
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            logging.info("邮件发送成功")
            return
        except Exception as e:
            attempt += 1
            logging.error(f"发送邮件失败 (尝试 {attempt}/{retries}): {e}")
            time.sleep(delay)
    logging.error("所有邮件发送尝试均失败")


def get_process_start_time(pid):
    """获取进程的启动时间"""
    try:
        proc = psutil.Process(pid)
        return proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def monitor_process():
    """监控特定PID的进程状态并在进程结束时发送邮件"""
    pid = MONITORED_PID
    start_time = get_process_start_time(pid)

    if start_time is None:
        logging.error(f"进程ID {pid} 不存在或无法访问。")
        sys.exit(1)

    logging.info(
        f"开始监控进程ID {pid}，启动时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    while True:
        if not psutil.pid_exists(pid):
            logging.info(f"检测到进程ID {pid} 已结束")
            subject = f"通知：进程ID {pid} 已结束"
            body = f"亲爱的用户，\n\n您监控的进程ID {pid} 已于 {time.strftime('%Y-%m-%d %H:%M:%S')} 结束运行。\n\n此致，\n服务器监控系统"
            send_email(subject, body)
            break
        else:
            try:
                proc = psutil.Process(pid)
                # 验证进程的启动时间是否匹配，防止PID被重用
                current_start_time = proc.create_time()
                if current_start_time != start_time:
                    logging.info(f"检测到进程ID {pid} 已被新进程占用。")
                    subject = f"通知：进程ID {pid} 已被新进程占用"
                    body = f"亲爱的用户，\n\nPID {pid} 已被新进程占用。\n\n此致，\n服务器监控系统"
                    send_email(subject, body)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                logging.info(f"检测到进程ID {pid} 已结束")
                subject = f"通知：进程ID {pid} 已结束"
                body = f"亲爱的用户，\n\n您监控的进程ID {pid} 已于 {time.strftime('%Y-%m-%d %H:%M:%S')} 结束运行。\n\n此致，\n服务器监控系统"
                send_email(subject, body)
                break
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    monitor_process()
