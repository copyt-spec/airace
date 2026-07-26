# scripts/get_telegram_chat_id.py
# -*- coding: utf-8 -*-
"""
Telegram Botのトークンから、自分のchat_idを調べるための1回限りの補助スクリプト。

事前に、TelegramアプリでBotFatherから作ったBotに自分から何かメッセージ
(例: "start")を送っておくこと。それをしていないとgetUpdatesが空になる。

使い方:
  python scripts/get_telegram_chat_id.py <BOTトークン>
"""

from __future__ import annotations

import sys

import requests


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: python scripts/get_telegram_chat_id.py <BOTトークン>")
        sys.exit(1)

    token = sys.argv[1].strip()
    url = f"https://api.telegram.org/bot{token}/getUpdates"

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok"):
        print("[ERROR] Telegram API error:", data)
        sys.exit(1)

    results = data.get("result", [])
    if not results:
        print(
            "[WARN] メッセージ履歴が見つかりません。"
            "TelegramアプリでこのBotに何かメッセージ(例: start)を送ってから、"
            "もう一度実行してください。"
        )
        return

    seen = set()
    for item in results:
        msg = item.get("message") or item.get("edited_message") or {}
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if chat_id is None or chat_id in seen:
            continue
        seen.add(chat_id)
        name = chat.get("username") or chat.get("first_name") or "?"
        print(f"chat_id={chat_id}  (from: {name}, type={chat.get('type')})")

    print()
    print("上に出た chat_id を ~/.boat_ai_telegram_chat_id に保存してください:")
    print("  echo \"<chat_id>\" > ~/.boat_ai_telegram_chat_id")


if __name__ == "__main__":
    main()
