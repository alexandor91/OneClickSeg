import os
import smtplib
import argparse
from email.mime.text import MIMEText


def send_mail(to, subject, body, smtp_server, smtp_port, username, password):
    msg = MIMEText(body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(username, password)
            server.sendmail(username, [to], msg.as_string())
    except Exception as e:
        print(f"Warning: Exception after sending mail: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--to", required=True, type=str, help="email address to send to"
    )
    parser.add_argument(
        "-s", "--subject", required=True, type=str, help="subject of the email"
    )
    parser.add_argument(
        "-b", "--body", required=True, type=str, help="body of the email"
    )
    args = parser.parse_args()

    # 你的邮箱配置
    smtp_server = "smtp.qq.com"
    smtp_port = 465
    username = os.environ["QQ_EMAIL_USER"]
    password = os.environ["QQ_EMAIL_PASS"]

    send_mail(
        args.to, args.subject, args.body, smtp_server, smtp_port, username, password
    )
