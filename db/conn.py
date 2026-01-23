"""Compat layer.

Mantém os imports do app modular antigo:
- load_env(): carrega .env
- get_db_config(): retorna tuplas vazias (não usado no Parquet)
"""
import os
from dotenv import load_dotenv

def load_env(base_dir: str):
    env_path = os.path.join(base_dir, ".env")
    load_dotenv(env_path, override=False)

def get_db_config():
    return ("", "", "", "", "")
