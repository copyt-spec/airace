# engine/telegram_notifier.py
# -*- coding: utf-8 -*-
"""
Telegram Bot APIで自分専用チャットにテキストメッセージを送信する薄いラッパー。

LINE Messaging APIの無料枠(月200通)が厳しいため、個人利用で回数上限が
実質無いTelegram Botに切り替えた。

事前準備(1回だけ):
  1. Telegramアプリで @BotFather を検索して開始
  2. /newbot を送り、案内に従ってBot名・ユーザー名を決める
     → 最後に発行される「HTTP APIトークン」をコピー
  3. 作成したBotに自分から何かメッセージを送る(例: "start")
     (これをしないとTelegram側がchat_idを発行しない)
  4. python scripts/get_telegram_chat_id.py <トークン> を実行し、
     自分のchat_idを確認する
  5. トークンとchat_idを保存する:
       echo "トークン" > ~/.boat_ai_telegram_token
       echo "chat_id" > ~/.boat_ai_telegram_chat_id
       chmod 600 ~/.boat_ai_telegram_token ~/.boat_ai_telegram_chat_id

読み込み優先順位(トークン・chat_idとも共通):
  1. 環境変数 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
  2. ~/.boat_ai_telegram_token / ~/.boat_ai_telegram_chat_id
どちらもgit管理下に置かないこと(トークン漏洩防止)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV_VAR = "TELEGRAM_CHAT_ID"

TOKEN_FILE_FALLBACK = Path.home() / ".boat_ai_telegram_token"
CHAT_ID_FILE_FALLBACK = Path.home() / ".boat_ai_telegram_chat_id"

MAX_TEXT_LEN = 4000  # Telegramの1メッセージ上限(4096文字)に少し余裕を持たせる


def _load_value(env_var: str, file_path: Path) -> Optional[str]:
    v = os.environ.get(env_var, "").strip()
    if v:
        return v
    if file_path.exists():
        v = file_path.read_text(encoding="utf-8").strip()
        if v:
            return v
    return None


def _load_token() -> Optional[str]:
    return _load_value(TOKEN_ENV_VAR, TOKEN_FILE_FALLBACK)


def _load_chat_id() -> Optional[str]:
    return _load_value(CHAT_ID_ENV_VAR, CHAT_ID_FILE_FALLBACK)


def send_telegram_message(text: str) -> bool:
    """
    自分のTelegramチャットにテキストメッセージを送信する。
    成功時True、トークン/chat_id未設定・送信失敗時Falseを返す(例外は投げない)。
    """
    token = _load_token()
    chat_id = _load_chat_id()
    if not token or not chat_id:
        print(
            f"[telegram_notifier] トークン/chat_id未設定のため送信スキップ。"
            f"環境変数 {TOKEN_ENV_VAR}/{CHAT_ID_ENV_VAR} か "
            f"{TOKEN_FILE_FALLBACK} / {CHAT_ID_FILE_FALLBACK} を設定してください。"
        )
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:MAX_TEXT_LEN]}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        print(f"[telegram_notifier] 送信失敗: status={resp.status_code} body={resp.text[:300]}")
        return False
    except Exception as e:
        print(f"[telegram_notifier] 送信エラー: {e}")
        return False


if __name__ == "__main__":
    # 動作確認用: python -m engine.telegram_notifier "テストメッセージ"
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "boat_ai Telegram通知テスト"
    ok = send_telegram_message(msg)
    print("OK" if ok else "NG")
