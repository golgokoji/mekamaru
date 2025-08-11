#!/usr/bin/env bash
set -euo pipefail

# USBマイク用 起動スクリプト
# 使い方:
#   chmod +x run_usb.sh
#   ./run_usb.sh

# 必要な環境変数例
# export GOOGLE_API_KEY=...
# export GOOGLE_APPLICATION_CREDENTIALS=/path/to/cred.json

# ALSAデバイスの指定（自動検出に任せる場合は空のまま）
# 例: plughw:1,0 / hw:1,0 / default
export MEKAMARU_ALSA_DEVICE="${MEKAMARU_ALSA_DEVICE:-}"

# 音声・VAD調整（必要に応じて）
export MEKAMARU_CHUNK_SEC="${MEKAMARU_CHUNK_SEC:-8.0}"
export MEKAMARU_VAD_MODE="${MEKAMARU_VAD_MODE:-2}"
export MEKAMARU_VAD_START_MS="${MEKAMARU_VAD_START_MS:-300}"
export MEKAMARU_VAD_END_MS="${MEKAMARU_VAD_END_MS:-600}"
export MEKAMARU_VAD_SILENCE_MS="${MEKAMARU_VAD_SILENCE_MS:-1200}"

# STT構成（必要に応じてv1/v2切替）
export MEKAMARU_STT_API="${MEKAMARU_STT_API:-v2}"
export MEKAMARU_GC_PROJECT_NUMBER="${MEKAMARU_GC_PROJECT_NUMBER:-}"
export MEKAMARU_GC_LOCATION="${MEKAMARU_GC_LOCATION:-global}"
export MEKAMARU_GC_RECOGNIZER_ID="${MEKAMARU_GC_RECOGNIZER_ID:-_}"

# 監視用の小音量再生（不要なら0に）
export MEKAMARU_MONITOR_PLAY="${MEKAMARU_MONITOR_PLAY:-1}"

cd "$(dirname "$0")"
exec python3 ok_mekamaru.py
