import os
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

def load_env(base_dir: str):
    load_dotenv(os.path.join(base_dir, ".env"), override=False)

def get_db_config():
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    pwd  = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", ""))
    name = os.getenv("DB_NAME", "finance_options")
    return host, port, user, pwd, name

def get_conn():
    host, port, user, pwd, name = get_db_config()
    return pymysql.connect(
        host=host, port=port,
        user=user, password=pwd,
        database=name, charset="utf8mb4",
        cursorclass=DictCursor, autocommit=True,
    )
