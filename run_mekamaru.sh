#!/usr/bin/env bash
# run_mekamaru.sh — USBマイク(ALSА/arecord)専用ランナー
set -euo pipefail

set -a
[ -f /home/pi/.env ] && . /home/pi/.env
set +a

# ---- venv ----
if [[ -f "$HOME/mekamaru-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/mekamaru-venv/bin/activate"
else
  echo "[runner] WARNING: venv が見つかりません: $HOME/mekamaru-venv/bin/activate" >&2
fi

# ---- .env 読み込み（あれば）----
ENV_FILE="$(dirname "$0")/.env"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

# ---- ロケール（日本語TTSの文字化け保険）----
export LANG=ja_JP.UTF-8
export LC_ALL=ja_JP.UTF-8

# ---- 必須環境（無くても起動はするが警告）----
: "${GOOGLE_API_KEY:=}"
: "${GOOGLE_APPLICATION_CREDENTIALS:=}"
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "[runner] WARNING: GOOGLE_API_KEY が未設定です（STT/AIが失敗します）"
fi
if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  echo "[runner] WARNING: GOOGLE_APPLICATION_CREDENTIALS が未設定です（TTSが失敗します）"
elif [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  echo "[runner] WARNING: key.json が見つかりません: $GOOGLE_APPLICATION_CREDENTIALS"
fi

# ---- ツール確認（ALSA）----
if ! command -v arecord >/dev/null 2>&1; then
  echo "[runner] WARNING: arecord が見つかりません（alsa-utils 未導入）" >&2
fi
if ! command -v aplay >/dev/null 2>&1; then
  echo "[runner] WARNING: aplay が見つかりません（alsa-utils 未導入）" >&2
fi

# ---- USBマイク設定（Python側で自動検出もあり）----
# 例: plughw:2,0  ← あなたの出力(カード2/デバイス0)から推奨
export MEKAMARU_ALSA_DEVICE="${MEKAMARU_ALSA_DEVICE:-}"

# ---- 音声・VAD 調整（必要に応じて上書き）----
export MEKAMARU_CHUNK_SEC="${MEKAMARU_CHUNK_SEC:-8.0}"
export MEKAMARU_VAD_MODE="${MEKAMARU_VAD_MODE:-2}"
export MEKAMARU_VAD_START_MS="${MEKAMARU_VAD_START_MS:-300}"
export MEKAMARU_VAD_END_MS="${MEKAMARU_VAD_END_MS:-600}"
export MEKAMARU_VAD_SILENCE_MS="${MEKAMARU_VAD_SILENCE_MS:-1200}"
export MEKAMARU_SEG_BRIDGING_MS="${MEKAMARU_SEG_BRIDGING_MS:-300}"

# ---- STT 構成（必要に応じて v1/v2 切替）----
export MEKAMARU_STT_API="${MEKAMARU_STT_API:-v2}"
export MEKAMARU_GC_PROJECT_NUMBER="${MEKAMARU_GC_PROJECT_NUMBER:-}"
export MEKAMARU_GC_LOCATION="${MEKAMARU_GC_LOCATION:-global}"
export MEKAMARU_GC_RECOGNIZER_ID="${MEKAMARU_GC_RECOGNIZER_ID:-_}"

# ---- 監視用の小音量再生（不要なら0）----
export MEKAMARU_MONITOR_PLAY="${MEKAMARU_MONITOR_PLAY:-0}"

echo "[runner] ALSA device: ${MEKAMARU_ALSA_DEVICE:-auto}"
echo "[runner] CHUNK_SEC: ${MEKAMARU_CHUNK_SEC}  VAD: start=${MEKAMARU_VAD_START_MS} end=${MEKAMARU_VAD_END_MS} silence=${MEKAMARU_VAD_SILENCE_MS}"

# ---- 実行 ----
cd "$(dirname "$0")"
exec python3 ok_mekamaru.py