# engine/line_notifier.py
# -*- coding: utf-8 -*-
"""
LINE Messaging API(broadcast)でテキストメッセージを送信する薄いラッパー。

LINE Notifyは2025-03-31に廃止されたため、後継のMessaging APIを使う。
このモジュールは「LINE公式アカウントの友だちは自分だけ」という前提で
broadcast(友だち全員に送信)を使う。複数人に送る場合はpush API+userIdに
切り替えること。

トークンの読み込み優先順位:
  1. 環境変数 LINE_CHANNEL_ACCESS_TOKEN
  2. ~/.boat_ai_line_token というファイル(1行目にトークンだけ書く)
どちらもgit管理下に置かないこと(トークン漏洩防止)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

TOKEN_ENV_VAR = "LINE_CHANNEL_ACCESS_TOKEN"
TOKEN_FILE_FALLBACK = Path.home() / ".boat_ai_line_token"

BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"
MAX_TEXT_LEN = 4900  # LINEの1メッセージあたり上限(5000)に少し余裕を持たせる


def _load_token() -> Optional[str]:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if token:
        return token
    if TOKEN_FILE_FALLBACK.exists():
        token = TOKEN_FILE_FALLBACK.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def send_line_message(text: str) -> bool:
    """
    LINE公式アカウントの友だち全員にテキストメッセージをbroadcast送信する。
    成功時True、トークン未設定/送信失敗時Falseを返す(例外は投げない)。
    """
    token = _load_token()
    if not token:
        print(
            f"[line_notifier] トークン未設定のため送信スキップ。"
            f"環境変数 {TOKEN_ENV_VAR} か {TOKEN_FILE_FALLBACK} を設定してください。"
        )
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {"messages": [{"type": "text", "text": text[:MAX_TEXT_LEN]}]}

    try:
        resp = requests.post(BROADCAST_URL, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        print(f"[line_notifier] 送信失敗: status={resp.status_code} body={resp.text[:300]}")
        return False
    except Exception as e:
        print(f"[line_notifier] 送信エラー: {e}")
        return False


if __name__ == "__main__":
    # 動作確認用: python -m engine.line_notifier "テストメッセージ"
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "boat_ai LINE通知テスト"
    ok = send_line_message(msg)
    print("OK" if ok else "NG")
