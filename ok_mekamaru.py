#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OKメカ丸 (clean unified build)

import os, sys, time, wave, shutil, unicodedata, subprocess, re, math
from typing import Tuple, List
import logging
import webrtcvad
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech_v1
from google.cloud import speech_v2 as speech_v2

logging.basicConfig(level=logging.INFO)

# ====== 環境・定数 ======
HOME = os.path.expanduser("~")
VOICES_DIR = os.path.join(HOME, "voices")
START_WAV = os.path.join(VOICES_DIR, "start.wav")
EXIT_WAV  = os.path.join(VOICES_DIR, "exit.wav")

RATE = 16000
SAMPLE_WIDTH = 2
FRAME_MS = 20
CHUNK_SEC = float(os.getenv("MEKAMARU_CHUNK_SEC", "8.0"))

# VADチューニング（現実的な既定値）
VAD_MODE = int(os.getenv("MEKAMARU_VAD_MODE", "2"))
VAD_START_MS = int(os.getenv("MEKAMARU_VAD_START_MS", "300"))
VAD_END_MS   = int(os.getenv("MEKAMARU_VAD_END_MS", "600"))
VAD_SILENCE_MS = int(os.getenv("MEKAMARU_VAD_SILENCE_MS", "1200"))
SEG_BRIDGING_MS = int(os.getenv("MEKAMARU_SEG_BRIDGING_MS", "300"))
MIN_SEG_MS = int(os.getenv("MEKAMARU_MIN_SEG_MS", "300"))

# AGC/モニタ
AGC_ENABLE = os.getenv("MEKAMARU_AGC", "1") not in ("0","false","no","off")
AGC_TARGET_RMS = int(os.getenv("MEKAMARU_AGC_TARGET", "1200"))
AGC_MAX_GAIN = float(os.getenv("MEKAMARU_AGC_MAX_GAIN", "25"))
MONITOR_PLAY = os.getenv("MEKAMARU_MONITOR_PLAY", "0") not in ("0","false","no","off")
MONITOR_SEC = float(os.getenv("MEKAMARU_MONITOR_SEC", "1.0"))
MONITOR_GAIN = float(os.getenv("MEKAMARU_MONITOR_GAIN", "0.25"))
MONITOR_AUTO_GAIN = os.getenv("MEKAMARU_MONITOR_AUTO_GAIN", "1") not in ("0","false","no","off")
SEG_PICK_LOUDEST = os.getenv("MEKAMARU_SEG_PICK_LOUDEST", "1") not in ("0","false","no","off")
COOLDOWN_SEC = float(os.getenv("MEKAMARU_COOLDOWN_SEC", "1.5"))
DEBUG_PLAY = os.getenv("MEKAMARU_DEBUG_PLAY", "0") == "1"
TRIGGER_MODE = os.getenv("MEKAMARU_TRIGGER_MODE", "soft").lower()

# USBマイク（ALSA）用設定
ALSA_DEVICE = os.getenv("MEKAMARU_ALSA_DEVICE", "").strip()  # 例: plughw:1,0 / hw:1,0 / default
ALSA_FORMAT = os.getenv("MEKAMARU_ALSA_FORMAT", "S16_LE")    # arecord -f
ALSA_RATE = int(os.getenv("MEKAMARU_ALSA_RATE", str(RATE)))   # arecord -r
ALSA_CHANNELS = int(os.getenv("MEKAMARU_ALSA_CHANNELS", "1")) # arecord -c

# 入出力デバイス
SET_SRC_VOL = os.getenv("MEKAMARU_SET_SRC_VOL", "").strip()
AUTO_UNMUTE = os.getenv("MEKAMARU_AUTO_UNMUTE", "1") not in ("0","false","no","off")

# Google 設定
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY 未設定", file=sys.stderr); sys.exit(1)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

GCP_CRED = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GC_VOICE_NAME    = os.getenv("MEKAMARU_GC_VOICE", "ja-JP-Standard-A")
GC_SPEAKING_RATE = float(os.getenv("MEKAMARU_GC_RATE", "1.0"))
GC_PITCH         = float(os.getenv("MEKAMARU_GC_PITCH", "0.0"))
USE_SOX_FX       = (os.getenv("MEKAMARU_TTS_FX", "0") == "1")
TTS_PREROLL_MS   = int(os.getenv("MEKAMARU_TTS_PREROLL_MS", "220"))  # 再生頭切れ対策の無音付与（既定220ms）
APLAY_BUFFER_US  = int(os.getenv("MEKAMARU_APLAY_BUFFER_US", "0") or 0)  # aplay -B（未設定なら使わない）
APLAY_AVAIL_US   = int(os.getenv("MEKAMARU_APLAY_AVAIL_US", "0") or 0)   # aplay -A（未設定なら使わない）
TTS_PRIME_MS     = int(os.getenv("MEKAMARU_TTS_PRIME_MS", "60"))         # 再生直前に無音でデバイスを起こす
TTS_PRIME_GAP    = float(os.getenv("MEKAMARU_TTS_PRIME_GAP", "2.0"))     # 直近再生からこの秒数以上あけばprime

GC_PROJECT_NUMBER = os.getenv("MEKAMARU_GC_PROJECT_NUMBER", "").strip()
GC_LOCATION = os.getenv("MEKAMARU_GC_LOCATION", "global").strip()
GC_RECOGNIZER_ID = os.getenv("MEKAMARU_GC_RECOGNIZER_ID", "_").strip()

# STT API
API_VER = os.getenv("MEKAMARU_STT_API", "v1").lower()
STT_DUAL_FALLBACK = os.getenv("MEKAMARU_STT_DUAL", "1") not in ("0","false","no","off")

# v2要求でも必須のプロジェクト番号が未設定なら静かにv1へ切替
if API_VER == "v2" and not GC_PROJECT_NUMBER:
    print("[STT] v2設定不足: MEKAMARU_GC_PROJECT_NUMBER 未設定のため v1 に自動切替", flush=True)
    API_VER = "v1"

# 応答文字数上限
MAX_CHARS = int(os.getenv("MEKAMARU_MAX_CHARS", "180"))
PROFILE_PATH = os.getenv("MEKAMARU_PROFILE", "")

# YouTube/Browser
YOUTUBE_CHANNEL_HANDLE = "https://www.youtube.com/@abemutsuki"
YOUTUBE_CHANNEL_VIDEOS = "https://www.youtube.com/@abemutsuki/videos"
CHROME_PROFILE = os.path.join(HOME, ".mekamaru-chrome")

# 渋谷ライブ
SHIBUYA_LIVE_URLS = [
    "https://www.youtube.com/watch?v=tujkoXI8rWM&autoplay=1&mute=1",
    "https://www.youtube.com/channel/UCWs8rt4ofGmdV4N6KQpP10Q/live?autoplay=1&mute=1",
    "https://www.skylinewebcams.com/en/webcam/japan/kanto/tokyo/tokyo-shibuya-scramble-crossing.html",
    "https://www.youtube.com/results?search_query=渋谷+スクランブル+交差点+ライブ",
]

def open_shibuya_live(close_first: bool = True) -> bool:
    if close_first:
        try:
            close_browser()
        except Exception:
            pass
    for url in SHIBUYA_LIVE_URLS:
        try:
            ok = launch_chromium_single(url)
            if ok:
                _log(f"[INFO] 渋谷ライブカメラ起動: {url}")
                shibuya_wav = os.path.join(VOICES_DIR, "shibuya.wav")
                if os.path.exists(shibuya_wav):
                    play_wav(shibuya_wav, label="渋谷ライブ")
                else:
                    respond("渋谷スクランブル交差点のライブ映像を表示します。")
                return True
        except Exception as e:
            _log(f"[INFO] 渋谷ライブカメラ起動失敗: {e}")
    respond("申し訳ありません。渋谷ライブ映像を表示できませんでした。")
    return False

def read_wav_pcm(path: str) -> bytes:
    # wavファイルからPCMデータを抽出して返す。
    # 不正な場合は空bytes。
    try:
        with wave.open(path,"rb") as wf:
            if wf.getnchannels()!=1 or wf.getframerate()!=RATE or wf.getsampwidth()!=2:
                return b""
            return wf.readframes(wf.getnframes())
    except Exception:
        return b""

def pcm_to_frames(pcm: bytes, frame_ms=20) -> List[bytes]:
    # PCMデータを指定msごとのフレームに分割してリストで返す。
    frame_len = int(RATE * (frame_ms/1000.0)) * SAMPLE_WIDTH
    return [pcm[i:i+frame_len] for i in range(0, len(pcm)-frame_len+1, frame_len)]

# ====== VAD ======
class Segmenter:
    # VAD（無音区間検出）で有声区間のみ抽出するクラス。
    # consume_framesで音声フレームを渡すと有声区間ごとに分割して返す。
    def __init__(self, vad_mode=VAD_MODE, start_ms=VAD_START_MS, end_ms=VAD_END_MS, silence_ms=VAD_SILENCE_MS, bridging_ms=SEG_BRIDGING_MS):
        self.vad = webrtcvad.Vad(vad_mode)
        self.start_need = start_ms
        self.end_need   = end_ms
        self.silence_ms = silence_ms
        self.bridging_ms = bridging_ms
        self.hangover_frames = int(self.silence_ms / FRAME_MS)
        self.state = "idle"
        self.buf = bytearray()
        self.run_ms = 0
        self.sil_ms = 0
    def consume_frames(self, frames: List[bytes]) -> List[bytes]:
        # VADで有声区間のみ抽出し、区間ごとにbytesリストで返す。
        segments = []
        for idx, fr in enumerate(frames):
            is_voiced = self.vad.is_speech(fr, RATE)
            _log(f"[DIAG][VAD] frame={idx} state={self.state} is_voiced={is_voiced} run_ms={self.run_ms} sil_ms={self.sil_ms}")
            if self.state == "idle":
                if is_voiced:
                    self.run_ms += FRAME_MS
                    if self.run_ms >= self.start_need:
                        self.state = "run"
                        self.buf.extend(fr)
                        self.sil_ms = 0
                        _log(f"[DIAG][VAD] 状態遷移: idle→run")
                else:
                    self.run_ms = 0
            else:
                self.buf.extend(fr)
                if is_voiced:
                    self.sil_ms = 0
                else:
                    self.sil_ms += FRAME_MS
                    if self.sil_ms >= self.end_need:
                        segments.append(bytes(self.buf))
                        self.state = "idle"
                        self.buf = bytearray()
                        self.run_ms = 0
                        self.sil_ms = 0
                        _log(f"[DIAG][VAD] 状態遷移: run→idle 区間抽出")
        return segments  # 必ずリストを返す

# ====== WAV書き出し ======
def write_wav_from_pcm(path: str, pcm: bytes):
    # PCMデータをwavファイルとして保存する。
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
        wf.writeframes(pcm)

def _pcm_rms(pcm: bytes) -> float:
    try:
        import array
        arr = array.array('h'); arr.frombytes(pcm)
        if not arr: return 0.0
        s2 = 0
        for v in arr[::max(1, len(arr)//1600)]:  # 約1秒1600サンプル参照
            s2 += v*v
        n = max(1, len(arr)//max(1, len(arr)//1600))
        return (s2/n) ** 0.5
    except Exception:
        return 0.0

def _pcm_attenuate(pcm: bytes, gain: float) -> bytes:
    if gain >= 0.999: return pcm
    try:
        import array
        arr = array.array('h'); arr.frombytes(pcm)
        g = max(0.0, min(1.0, gain))
        for i in range(len(arr)):
            arr[i] = int(arr[i] * g)
        return arr.tobytes()
    except Exception:
        return pcm

def _pcm_slice_sec(pcm: bytes, sec: float) -> bytes:
    max_bytes = int(RATE * SAMPLE_WIDTH * max(0.0, sec))
    return pcm[:max_bytes]

def _pcm_loudest_slice(pcm: bytes, sec: float) -> tuple[bytes, float]:
    """PCMから指定秒の区間でRMSが最大のスライスを返す。(slice_bytes, offset_sec)"""
    try:
        import array
        win = max(1, int(RATE * SAMPLE_WIDTH * max(0.05, sec)))
        if len(pcm) <= win:
            return (pcm, 0.0)
        step = win // 2
        best_off = 0
        best_rms = -1.0
        # 16bitサンプル配列
        arr = array.array('h'); arr.frombytes(pcm)
        samples_per_win = win // SAMPLE_WIDTH
        samples_step = max(1, step // SAMPLE_WIDTH)
        for off_bytes in range(0, len(pcm) - win + 1, step):
            off_smp = off_bytes // SAMPLE_WIDTH
            window = arr[off_smp: off_smp + samples_per_win]
            if not window:
                continue
            s2 = 0
            for v in window[::max(1, len(window)//400)]:
                s2 += v*v
            n = max(1, len(window)//max(1, len(window)//400))
            rms = (s2/n) ** 0.5
            if rms > best_rms:
                best_rms = rms
                best_off = off_bytes
        return (pcm[best_off: best_off + win], best_off / (RATE * SAMPLE_WIDTH))
    except Exception:
        return (pcm, 0.0)

# ====== STT（v1/v2ディスパッチ） ======
def _recognizer_path() -> str:
    if not GC_PROJECT_NUMBER:
        raise RuntimeError("MEKAMARU_GC_PROJECT_NUMBER が未設定です。")
    rid = GC_RECOGNIZER_ID or "_"
    return f"projects/{GC_PROJECT_NUMBER}/locations/{GC_LOCATION}/recognizers/{rid}"

def stt_from_pcm_google_v2(pcm: bytes) -> str:
    tmp = "/tmp/seg.wav"
    write_wav_from_pcm(tmp, pcm)
    try:
        client = speech_v2.SpeechClient()
        with open(tmp, "rb") as f:
            content = f.read()
        config = speech_v2.RecognitionConfig(
            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
            language_codes=["ja-JP"],
            model="latest_short",
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        req = speech_v2.RecognizeRequest(
            recognizer=_recognizer_path(),
            config=config,
            content=content,
        )
        resp = client.recognize(request=req)
        cand = []
        for r in resp.results:
            if r.alternatives: cand.append(r.alternatives[0].transcript)
        return " ".join(cand).strip()
    except Exception as e:
        _log(f"[STT v2 ERROR] {e}")
        return ""

def stt_from_pcm_google_v1(pcm: bytes) -> str:
    try:
        client = speech_v1.SpeechClient()
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ja-JP",
            enable_automatic_punctuation=True,
        )
        # まずRAW PCMで試す
        audio = speech_v1.RecognitionAudio(content=pcm)
        resp = client.recognize(config=config, audio=audio)
        cand = []
        for r in resp.results:
            if r.alternatives: cand.append(r.alternatives[0].transcript)
        txt = " ".join(cand).strip()
        if txt:
            return txt
        # 次にWAVバイトで再試行（環境差異に強くなることがある）
        tmp = "/tmp/_v1_retry.wav"
        write_wav_from_pcm(tmp, pcm)
        with open(tmp, "rb") as f:
            wav_bytes = f.read()
        audio2 = speech_v1.RecognitionAudio(content=wav_bytes)
        resp2 = client.recognize(config=config, audio=audio2)
        cand2 = []
        for r in resp2.results:
            if r.alternatives: cand2.append(r.alternatives[0].transcript)
        return " ".join(cand2).strip()
    except Exception as e:
        _log(f"[STT v1 ERROR] {e}")
        return ""

def stt_from_pcm(pcm: bytes) -> str:
    # まず選択APIで試し、必要ならsoxで増幅→再試行、さらにもう一方APIへフォールバック
    primary = stt_from_pcm_google_v2 if API_VER == "v2" else stt_from_pcm_google_v1
    secondary = stt_from_pcm_google_v1 if API_VER == "v2" else stt_from_pcm_google_v2
    txt = primary(pcm)
    if txt:
        return txt
    # soxで正規化/コンパンド後に再試行（ある場合）
    enh = enhance_pcm_with_sox(pcm)
    if enh is not None and enh != pcm:
        _log("[STT] sox強化後で再試行(Primary)")
        txt = primary(enh)
        if txt:
            return txt
    if STT_DUAL_FALLBACK:
        _log("[STT] Primary空結果→Secondaryへフォールバック")
        txt = secondary(pcm)
        if txt:
            return txt
        if enh is not None and enh != pcm:
            _log("[STT] sox強化後で再試行(Secondary)")
            txt = secondary(enh)
            if txt:
                return txt
    return ""

def enhance_pcm_with_sox(pcm: bytes) -> bytes | None:
    """soxがあれば正規化/圧縮で音量・可聴性を上げたPCMを返す。sox無ければNone。"""
    if not shutil.which("sox"):
        return None
    try:
        src = "/tmp/_stt_in.wav"
        dst = "/tmp/_stt_enh.wav"
        write_wav_from_pcm(src, pcm)
        # やりすぎない程度に正規化 + コンパンド + 16k/mono
        cmd = [
            "sox", src, "-r", str(RATE), "-c", "1", dst,
            "gain", "-n", "-3",
            "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            return None
        enh_pcm = read_wav_pcm(dst)
        return enh_pcm if enh_pcm else None
    except Exception:
        return None

# ====== ソース自動検出（保険） ======
def _detect_usb_alsa_device() -> str:
    """arecord -l から USB マイクらしきデバイスを見つけて plughw:<card>,<dev> を返す（日本語ロケール対応）。見つからなければ空。"""
    if not shutil.which("arecord"):
        return ""
    try:
        out = subprocess.run(["arecord","-l"], capture_output=True, text=True, timeout=3)
        text = out.stdout or ""
        # 英/日どちらでもマッチする包括的パターンを全体から検索
        pat = r"(?:card|カード)\s+(\d+).*?(?:device|デバイス)\s+(\d+)"
        first = None
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            card, dev = m.group(1), m.group(2)
            span = m.span()
            around = text[max(0, span[0]-80): min(len(text), span[1]+80)].lower()
            if ("usb" in around) or ("mic" in around):
                return f"plughw:{card},{dev}"
            if first is None:
                first = (card, dev)
        if first:
            return f"plughw:{first[0]},{first[1]}"
    except Exception:
        pass
    return ""

ALSA_DEV = ALSA_DEVICE or _detect_usb_alsa_device()

# ---- Google 設定 ----
GCP_CRED = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GC_VOICE_NAME    = os.getenv("MEKAMARU_GC_VOICE", "ja-JP-Standard-A")
GC_SPEAKING_RATE = float(os.getenv("MEKAMARU_GC_RATE", "1.0"))
GC_PITCH         = float(os.getenv("MEKAMARU_GC_PITCH", "0.0"))
USE_SOX_FX       = (os.getenv("MEKAMARU_TTS_FX", "0") == "1")

GC_PROJECT_NUMBER = os.getenv("MEKAMARU_GC_PROJECT_NUMBER", "1077743802102").strip()
GC_LOCATION = os.getenv("MEKAMARU_GC_LOCATION", "global").strip()

# 応答最大文字数（最終ハード制限）
MAX_CHARS = int(os.getenv("MEKAMARU_MAX_CHARS", "180"))

PROFILE_PATH = os.getenv("MEKAMARU_PROFILE", "")

# YouTube（ラッキーマイン=安倍むつき様のチャンネル）
YOUTUBE_CHANNEL_HANDLE = "https://www.youtube.com/@abemutsuki"
YOUTUBE_CHANNEL_VIDEOS = "https://www.youtube.com/@abemutsuki/videos"

# 専用Chromium（単一ウインドウ運用）
CHROME_PROFILE = os.path.join(HOME, ".mekamaru-chrome")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY 未設定", file=sys.stderr); sys.exit(1)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# ====== 共通 ======
def _log(msg: str):
    # 標準出力にログを即時表示する関数。
    print(msg, flush=True)

_last_play_ts = 0.0
def play_wav(path: str, label: str = "") -> bool:
    # 指定したwavファイルをaplayで再生する。
    # labelはログ用のラベル。
    # 再生成功でTrue、失敗でFalse。
    if not path or not os.path.exists(path): return False
    if not shutil.which("aplay"):
        _log(f"[再生スキップ] aplay 不在: {path}"); return False
    # aplay 引数（必要に応じてバッファ調整）
    aplay_args = ["aplay","-q"]
    if APLAY_BUFFER_US > 0:
        aplay_args += ["-B", str(APLAY_BUFFER_US)]
    if APLAY_AVAIL_US > 0:
        aplay_args += ["-A", str(APLAY_AVAIL_US)]
    # しばらく再生していないなら、無音でウォームアップ
    global _last_play_ts
    now = time.time()
    if TTS_PRIME_MS > 0 and (now - _last_play_ts) > TTS_PRIME_GAP:
        try:
            prime_pcm = b"\x00" * int(RATE * SAMPLE_WIDTH * (TTS_PRIME_MS/1000.0))
            prime_wav = "/tmp/_play_prime.wav"
            write_wav_from_pcm(prime_wav, prime_pcm)
            subprocess.run(aplay_args + [prime_wav], check=False)
        except Exception:
            pass
    if label: _log(f'再生「{label}」')
    subprocess.run(aplay_args + [path], check=False)
    _last_play_ts = time.time()
    return True

def load_profile_text() -> str:
    # プロフィールテキストファイルを読み込んで返す。
    # 存在しない場合は空文字。
    if PROFILE_PATH and os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f: return f.read().strip()
        except Exception: pass
    return ""
PROFILE_TEXT = load_profile_text()

# ====== TTS ======
def say_jp(text: str) -> bool:
    # Google Cloud TTSで日本語テキストを音声合成し再生する。
    # 成功でTrue、失敗でFalse。
    try:
        # 資格情報チェック
        if not GCP_CRED or not os.path.exists(GCP_CRED):
            _log("[ERROR] GOOGLE_APPLICATION_CREDENTIALS 未設定/不正"); return False

        # 既定（ロボット風・低め・ゆっくり）＋ 環境変数で上書き
        def _safe_float(val: str, default: float) -> float:
            try:
                return float(val)
            except Exception:
                return default
        def _safe_int(val: str, default: int) -> int:
            try:
                iv = int(float(val))
                return iv if iv > 0 else default
            except Exception:
                return default

        voice_name = (os.getenv("MEKAMARU_TTS_VOICE", "ja-JP-Wavenet-D") or "").strip() or "ja-JP-Wavenet-D"
        speaking_rate = _safe_float(os.getenv("MEKAMARU_TTS_RATE", "0.78"), 0.78)
        pitch = _safe_float(os.getenv("MEKAMARU_TTS_PITCH", "-16.0"), -16.0)
        sample_rate = _safe_int(os.getenv("MEKAMARU_TTS_SAMPLE_RATE", "16000"), 16000)

        client = texttospeech.TextToSpeechClient()
        inp = texttospeech.SynthesisInput(text=(text or ""))
        voice = texttospeech.VoiceSelectionParams(language_code="ja-JP", name=voice_name)
        cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
            sample_rate_hertz=sample_rate,
        )
        resp = client.synthesize_speech(input=inp, voice=voice, audio_config=cfg)

        # 出力先は固定
        raw = "/tmp/mekamaru_tts_raw.wav"
        with open(raw, "wb") as f:
            f.write(resp.audio_content)

        # SoX系のエフェクトは無効化（要件により削除）
        # if USE_SOX_FX and shutil.which("sox"):
        #     pass

        # そのまま再生（aplay -q）
        subprocess.run(["aplay", "-q", raw], check=False)
        return True
    except Exception as e:
        _log(f"[ERROR] Google TTS失敗: {e}"); return False

# ====== 口癖 ======
CATCH_PHRASE_WAVS = [
    ("了解", os.path.join(VOICES_DIR,"ryoukai.wav"), lambda s: ("了解" in s) or ("りょうかい" in s)),
    ("それは無理", os.path.join(VOICES_DIR,"muri.wav"),     lambda s: ("無理" in s) or ("それは無理" in s)),
]
def respond(text: str):
    # 口癖にマッチすればwav再生、そうでなければTTSで応答。
    s = (text or "").strip()
    for key, wavp, cond in CATCH_PHRASE_WAVS:
        if cond(s) and os.path.exists(wavp):
            _log(f'口癖マッチ「{key}」 → wav再生: {wavp}')
            if play_wav(wavp, label=key): return
    # 理解不能な命令時は「それは無理だ」返答＆muri.wav再生
    if s in ["それは無理だ", "それは無理"] and os.path.exists(os.path.join(VOICES_DIR,"muri.wav")):
        play_wav(os.path.join(VOICES_DIR,"muri.wav"), label="それは無理")
        return
    if not say_jp(s): _log(f"[ERROR] 音声出力失敗: {s}")

# ====== 録音（小刻み） ======
def record_chunk_wav(path: str, seconds: float, retry: int = 3) -> bool:
    """USBマイク（ALSA/arecord）で指定秒数のWAVを収録する。失敗時はリトライ。"""
    if not shutil.which("arecord"):
        _log("[ERROR] arecord 不在（ALSA）"); return False
    sec_float = max(0.1, float(seconds))
    sec_int = max(1, int(math.ceil(sec_float)))  # arecord -d は整数秒のみ
    for attempt in range(retry):
        # 既存ファイルは削除
        try:
            if os.path.exists(path): os.remove(path)
        except Exception:
            pass
        cmd = [
            "arecord", "-q",
            "-d", str(sec_int),
            "-t", "wav",
            "-f", ALSA_FORMAT,
            "-r", str(ALSA_RATE),
            "-c", str(ALSA_CHANNELS),
        ]
        if ALSA_DEV:
            cmd += ["-D", ALSA_DEV]
        cmd += [path]
        _log(f"[REC] arecord 起動 device={ALSA_DEV or 'default'} sec={sec_int}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=sec_int + 2.0)
            if r.returncode != 0:
                _log(f"[WARN] arecord 終了コード={r.returncode} stderr={r.stderr.strip()[:200]}")
        except subprocess.TimeoutExpired:
            _log("[WARN] arecord タイムアウト")
        except Exception as e:
            _log(f"[ERROR] arecord 例外: {e}")

        # サイズ確認
        if os.path.exists(path):
            sz = os.path.getsize(path)
            expected_max = int(ALSA_RATE * SAMPLE_WIDTH * (sec_int + 0.5)) + 44
            if sz > 44 and sz <= expected_max * 2:
                return True
            _log(f"[WARN] 録音サイズ異常/失敗（{attempt+1}回目） size={sz} expected<=~{expected_max}")
        else:
            _log(f"[WARN] 録音ファイル未生成（{attempt+1}回目）")
        time.sleep(0.3)
    return False

def read_wav_pcm(path: str) -> bytes:
    # wavファイルからPCMデータを抽出して返す。
    # 不正な場合は空bytes。
    try:
        with wave.open(path,"rb") as wf:
            if wf.getnchannels()!=1 or wf.getframerate()!=RATE or wf.getsampwidth()!=2:
                return b""
            return wf.readframes(wf.getnframes())
    except Exception:
        return b""

def pcm_to_frames(pcm: bytes, frame_ms=20) -> List[bytes]:
    # PCMデータを指定msごとのフレームに分割してリストで返す。
    frame_len = int(RATE * (frame_ms/1000.0)) * SAMPLE_WIDTH
    return [pcm[i:i+frame_len] for i in range(0, len(pcm)-frame_len+1, frame_len)]

class Segmenter:
    def __init__(self, vad_mode=VAD_MODE, start_ms=VAD_START_MS, end_ms=VAD_END_MS, silence_ms=VAD_SILENCE_MS, bridging_ms=SEG_BRIDGING_MS):
        self.vad = webrtcvad.Vad(vad_mode)
        self.start_need = start_ms
        self.end_need   = end_ms
        self.silence_ms = silence_ms
        self.state = "idle"
        self.buf = bytearray()
        self.run_ms = 0
        self.sil_ms = 0
    def consume_frames(self, frames: List[bytes]) -> List[bytes]:
        segments = []
        for fr in frames:
            is_voiced = self.vad.is_speech(fr, RATE)
            if self.state == "idle":
                if is_voiced:
                    self.run_ms += FRAME_MS
                    if self.run_ms >= self.start_need:
                        self.state = "run"; self.buf.extend(fr); self.sil_ms = 0
                else:
                    self.run_ms = 0
            else:
                self.buf.extend(fr)
                if is_voiced:
                    self.sil_ms = 0
                else:
                    self.sil_ms += FRAME_MS
                    if self.sil_ms >= self.end_need:
                        segments.append(bytes(self.buf))
                        self.state = "idle"; self.buf = bytearray(); self.run_ms = 0; self.sil_ms = 0
        if self.state != "idle" and self.buf:
            segments.append(bytes(self.buf))
            self.state = "idle"; self.buf = bytearray(); self.run_ms = 0; self.sil_ms = 0
        return segments

# ====== WAV書き出し ======
def write_wav_from_pcm(path: str, pcm: bytes):
    # PCMデータをwavファイルとして保存する。
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
        wf.writeframes(pcm)

# ====== AGC（自動ゲイン調整） ======
def pcm_apply_agc(pcm: bytes, target_rms: int = AGC_TARGET_RMS) -> bytes:
    """16bit PCMに簡易AGCを適用して増幅。無音/極小なら原音を返す。
    クリップを避けるため倍率は環境変数で制限（既定: 最大25倍）。"""
    try:
        import array, math
        if not pcm:
            return pcm
        arr = array.array('h')
        arr.frombytes(pcm)
        # RMS計算（オーバーヘッド抑制のため一部サンプル）
        step = max(1, len(arr)//(RATE//10) )  # おおよそ0.1秒に1サンプル参照
        s2 = 0
        n = 0
        for i in range(0, len(arr), step):
            v = arr[i]
            s2 += v*v
            n += 1
        if n == 0:
            return pcm
        rms = int((s2/n) ** 0.5)
        if rms <= 0:
            return pcm
        if rms >= target_rms:
            return pcm  # 既に十分
        gain = min(AGC_MAX_GAIN, float(target_rms) / float(rms))
        _log(f"[AGC] rms={rms} → target={target_rms}, gain={gain:.2f}x")
        # クリップしないようスケール
        for i in range(len(arr)):
            nv = int(arr[i] * gain)
            if nv > 32767: nv = 32767
            elif nv < -32768: nv = -32768
            arr[i] = nv
        return arr.tobytes()
    except Exception as e:
        _log(f"[AGC] 失敗: {e}")
        return pcm

# ====== PulseAudio ソース音量設定（任意） ======
def set_source_volume_if_needed(src_name: str, vol: str) -> None:
    """環境変数が指定されていれば、pactlでソース音量を設定（例: 150%）。"""
    if not vol:
        return
    if not shutil.which("pactl"):
        _log("[VOL] pactl 不在のためスキップ")
        return
    try:
        # そのまま名前指定で試行
        r = subprocess.run(["pactl","set-source-volume", src_name, vol], capture_output=True, text=True)
        if r.returncode == 0:
            _log(f"[VOL] set-source-volume {src_name} {vol} 成功")
            return
        # ID取得して再試行
        out = subprocess.run(["pactl","list","short","sources"], capture_output=True, text=True)
        sid = None
        for ln in (out.stdout or "").splitlines():
            parts = ln.split('\t') if '\t' in ln else ln.split()
            if len(parts) >= 2 and parts[1] == src_name:
                sid = parts[0]
                break
        if sid:
            r2 = subprocess.run(["pactl","set-source-volume", sid, vol], capture_output=True, text=True)
            if r2.returncode == 0:
                _log(f"[VOL] set-source-volume id={sid} {vol} 成功")
            else:
                _log(f"[VOL] 失敗: {r2.stderr.strip()}")
        else:
            _log("[VOL] ソースID解決失敗。スキップ")
    except Exception as e:
        _log(f"[VOL] 例外: {e}")

def ensure_source_unmuted(src_name: str) -> None:
    """ソースのミュート解除を試みる。失敗しても続行。"""
    if not shutil.which("pactl"):
        return
    try:
        # 直接名前指定でミュート解除
        subprocess.run(["pactl","set-source-mute", src_name, "0"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # IDに対しても試行
        out = subprocess.run(["pactl","list","short","sources"], capture_output=True, text=True)
        for ln in (out.stdout or "").splitlines():
            parts = ln.split('\t') if '\t' in ln else ln.split()
            if len(parts) >= 2 and parts[1] == src_name:
                sid = parts[0]
                subprocess.run(["pactl","set-source-mute", sid, "0"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
    except Exception:
        pass

# 旧v2関数は削除（speech 参照の不整合回避）

# ====== 応答短文化（AI用） ======
def _shorten(text: str, max_chars: int) -> str:
    # テキストを最大文字数で切り詰めて返す。
    s = (text or "").strip()
    if len(s) <= max_chars: return s
    return s[:max_chars-1].rstrip() + "…"

STYLE_GUIDE = (
    "出力規則: 一人称は『俺』。自分を第三人称で呼ばない。"
    "必ず文末は『だ』で終わる。余計な語尾や装飾を付けない。"
    "質問に対して核心だけを答える。"
    f"主語は可能なら省略。文は簡潔に、最大{MAX_CHARS}文字。結論を先に。"
)

def normalize_ai_style(s: str) -> str:
    # GeminiのAI応答文を日本語スタイルに整形。
    if not s: return s
    s = re.sub(r'^\s*メカ\s*丸[は、]\s*', '私は', s)
    s = re.sub(r'^\s*メカ丸[は、]\s*', '私は', s)
    s = re.sub(r'(?<![^\n])メカ\s*丸は', '私は', s)
    return s.strip()

def mekamaru_ai_reply(user_text: str) -> str:
    # ユーザー入力をGeminiでAI応答生成し、短文化・スタイル整形して返す。
    head = f"{STYLE_GUIDE}\n\nユーザー入力: "
    prompt = head + (user_text or "")
    if PROFILE_PATH and PROFILE_TEXT:
        prompt = f"【参考プロフィール（必要時のみ参照）】\n{PROFILE_TEXT}\n\n" + prompt
    try:
        resp = model.generate_content(prompt)
        raw = (getattr(resp,"text","") or "").strip()
    except Exception as e:
        _log(f"[AI ERROR] {e}")
        raw = ""
    return normalize_ai_style(_shorten(raw, MAX_CHARS)) if raw else raw

# ====== OKトリガー（ファジー） ======
def _nfkc_lower(s: str) -> str:
    # 文字列をNFKC正規化＋小文字化。
    return unicodedata.normalize("NFKC", s or "").lower()
def _to_hiragana(s: str) -> str:
    # カタカナをひらがなに変換。
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(chr(ord(c)-0x60) if 0x30A1<=ord(c)<=0x30FA else c for c in s)
TRIG_PUNCTS = "、。,. 　:：-—–!！?？「」『』()（）\t\n\r"
def extract_after_trigger(text: str) -> Tuple[bool, str]:
    """先頭のトリガー語（OK/メカ丸 系）をすべて取り除き、(ヒット, 残り)を返す。
    例: "OK メカ丸 天気教えて" → (True, "天気教えて")
    """
    if not text:
        return (False, "")
    s = unicodedata.normalize("NFKC", text)
    # 先頭フィラー語を除去（例: えー/あの/ねえ/すげえ/ちょっと など）
    fillers = (
        "えー", "ええと", "えっと", "あー", "うーん", "あの", "その", "ねえ", "ねぇ", "ねー",
        "おい", "すげえ", "ちょっと", "まあ", "ね", "あのさ",
    )
    changed = True
    while changed:
        changed = False
        s2 = s.lstrip(TRIG_PUNCTS)
        if s2 != s:
            s = s2; changed = True
        for f in fillers:
            if s.startswith(f):
                s = s[len(f):]
                changed = True
                break
    # STTの空白混入を補正（メカ 丸 → メカ丸）
    s = re.sub(r"メカ\s*丸", "メカ丸", s)
    # OK/オーケー/オッケー/おけ/ホッケー(誤認補正) 等 + メカ丸/メカマル/めかまる
    trig = r"(?:ok|ｏｋ|オー?ケー?|オッケー?|おー?けー?|おっけー?|おけ|ホッ?ケー?|ほっ?けー?|メカ丸|メカマル|めかまる|ﾒｶﾏﾙ)"
    pattern = re.compile(rf"^\s*(?:{trig})\s*[{re.escape(TRIG_PUNCTS)}]*", re.IGNORECASE)
    hit = False
    while True:
        m = pattern.match(s)
        if not m:
            break
        hit = True
        s = s[m.end():]
    if hit:
        return (True, s.strip())
    # softモードでは、先頭トリガー無しでも質問/依頼っぽければ通す
    if TRIGGER_MODE != "strict":
        if re.search(r"(教えて|何|なに|どこ|いつ|だれ|誰|どうやって|方法|とは|って何|ください|お願い|して|できますか|？|\?)", s):
            return (True, s.strip())
    return (False, s.strip())

# 後方互換のため: 旧関数は新関数を呼ぶ
def extract_after_ok(text: str) -> Tuple[bool, str]:
    return extract_after_trigger(text)

# ====== 計算：数字だけ＋「だ」 ======
def try_calc_result(text: str) -> str:
    # テキストから四則演算式を抽出・評価。
    # 成功時は「<数値>だ」を返す。失敗時は空文字。
    """
    STTテキストから四則演算を抽出・評価。
    成功時は「<数値>だ」を返す。失敗時は空文字。
    対応：+, -, *, /, 小数, 括弧, 日本語表現（足す/引く/かける/割る、プラス/マイナス、×/÷、x）
    """
    if not text: return ""
    s = unicodedata.normalize("NFKC", text).lower()

    # 日本語→演算子へ正規化
    rep = [
        ("プラス", "+"), ("たす", "+"), ("足す", "+"),
        ("マイナス", "-"), ("ひく", "-"), ("引く", "-"),
        ("かける", "*"), ("掛ける", "*"), ("×", "*"), ("x", "*"),
        ("わる", "/"), ("割る", "/"), ("÷", "/"),
    ]
    for a, b in rep:
        s = s.replace(a.lower(), b)

    # 数式候補だけ残す
    s = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", " ", s)
    s = re.sub(r"\s+", "", s)

    # 少なくとも数字と演算子を含むか
    if not (re.search(r"[0-9]", s) and re.search(r"[\+\-\*\/]", s)):
        return ""

    # 安全評価：許可トークンのみ
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)]+", s):
        return ""

    try:
        # Pythonのevalを最小環境で
        val = eval(s, {"__builtins__": None}, {})
    except Exception:
        return ""

    # 数値整形：整数なら整数、小数はそのまま（末尾ゼロは簡易除去）
    if isinstance(val, (int, float)):
        if abs(val - int(round(val))) < 1e-12:
            out = str(int(round(val)))
        else:
            out = str(round(val, 10)).rstrip("0").rstrip(".")
        return f"{out}だ"
    return ""

# ====== YouTube 最新動画URL取得（yt-dlp 利用） ======
def get_latest_video_url() -> str:
    # yt-dlpでYouTubeチャンネルの最新動画URLを取得。
    # 失敗時は動画タブURL。
    if not shutil.which("yt-dlp"):
        _log("[YT] yt-dlp 不在。動画タブを開きます。")
        return YOUTUBE_CHANNEL_VIDEOS
    try:
        r = subprocess.run(
            ["yt-dlp", "--ignore-errors", "--flat-playlist",
             "--playlist-end", "1", "--get-id", YOUTUBE_CHANNEL_HANDLE],
            capture_output=True, text=True, timeout=15
        )
        vid = (r.stdout or "").strip().splitlines()[0] if r.returncode == 0 else ""
        if vid:
            url = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
            _log(f"[YT] 最新動画URL: {url}")
            return url
    except Exception as e:
        _log(f"[YT ERROR] {e}")
    _log("[YT] 取得失敗。動画タブを開きます。")
    return YOUTUBE_CHANNEL_VIDEOS

# ====== 単一ウインドウでChromium起動 ======
_last_launch_ts = 0.0
def launch_chromium_single(url: str):
    # Chromiumを専用プロフィール・単一ウインドウ・全画面で起動。
    # 既存ウインドウは終了してから再起動。
    # 連続起動は2秒デバウンス。
    """
    専用プロフィールで Chromium を 1ウインドウだけ起動。
    既存があれば終了してから再起動（増殖防止）。
    連続起動のデバウンス（2秒）付き。
    """
    global _last_launch_ts
    now = time.time()
    if now - _last_launch_ts < 2.0:
        _log("[BROWSER] 直近で起動済みのため抑止（デバウンス）")
        return True
    _last_launch_ts = now

    if not shutil.which("chromium-browser"):
        _log("[CMD ERROR] chromium-browser 不在"); return False

    # 既存の専用ウインドウを終了（-- でオプション終端、-f でコマンドライン全体マッチ）
    try:
        subprocess.run(
            ["pkill", "-f", "--", f"--user-data-dir={CHROME_PROFILE}"],
            check=False
        )
        time.sleep(0.4)
    except Exception:
        pass

    args = [
        "chromium-browser",
        "--kiosk",
        "--start-fullscreen",
        "--autoplay-policy=no-user-gesture-required",
        "--noerrdialogs",
        "--disable-session-crashed-bubble",
        "--disable-infobars",
        f"--user-data-dir={CHROME_PROFILE}",
        url
    ]
    _log(f"[BROWSER] 起動: {' '.join(args)}")
    try:
        subprocess.Popen(args)
        return True
    except Exception as e:
        _log(f"[CMD ERROR] ブラウザ起動失敗: {e}")
        return False


# ====== ブラウザ終了関数 ======
def close_browser() -> bool:
    # Chromium/Chromium-browser を静かに終了させる。
    # エラーログを抑制し、残存プロセスがあれば kill -9 で強制終了。
    # 成功時はTrue、失敗時はFalse。
    """
    Chromium/Chromium-browser を静かに終了させる。
    エラーログを抑制し、残存プロセスがあれば kill -9 で強制終了。
    """
    try:
        # pkill -f chromium (stderr/stdout抑制)
        r1 = subprocess.run(
            ["pkill", "-f", "chromium"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        # pkill -f chromium-browser (stderr/stdout抑制)
        r2 = subprocess.run(
            ["pkill", "-f", "chromium-browser"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        # pgrepで残存確認
        remain = []
        for proc in ["chromium", "chromium-browser"]:
            try:
                out = subprocess.run(
                    ["pgrep", "-f", proc],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
                )
                pids = [int(x) for x in out.stdout.strip().split() if x.isdigit()]
                remain.extend(pids)
            except Exception:
                pass
        # 残っていれば kill -9
        killed = False
        for pid in set(remain):
            try:
                subprocess.run(
                    ["kill", "-9", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
                killed = True
            except Exception:
                pass
        # 成功判定
        if r1.returncode == 0 or r2.returncode == 0 or killed:
            _log("[INFO] ブラウザ終了要求→成功")
            respond("ブラウザを終了しました。")
            return True
        else:
            _log("[INFO] ブラウザ終了要求→失敗")
            respond("申し訳ありません。ブラウザを終了できませんでした。")
            return False
    except Exception as e:
        _log(f"[INFO] ブラウザ終了要求→失敗: {e}")
        respond("申し訳ありません。ブラウザを終了できませんでした。")
        return False

# ====== カスタム命令（OKなしでも最優先） ======
def handle_custom_commands(text: str) -> bool:
    # カスタム命令（ブラウザ終了・ラッキーマイン動画再生等）を判定・実行。
    # 該当すればTrue、そうでなければFalse。
    # 空白・記号を除去して判定
    t = re.sub(r"[\s\u3000、。,.　:：!！?？「」『』()（）\-—–]", "", (text or ""))
    # 渋谷ライブカメラ表示コマンド語彙
    SHIBUYA_LIVE_WORDS = [
        "渋谷の状況", "渋谷の様子", "渋谷ライブ", "渋谷の定点", "渋谷スクランブル見せて",
        "渋谷スクランブル", "渋谷交差点", "渋谷ライブカメラ", "渋谷定点", "渋谷カメラ"
    ]
    for w in SHIBUYA_LIVE_WORDS:
        if w.replace(" ","") in t:
            _log("[CMD] 渋谷ライブカメラ命令検知")
            open_shibuya_live(close_first=True)
            return True
    # 「例の」「レイの」「レイノ」などにも反応
    if re.search(r"(okメカ丸.*例のものを|例のもの|れいのもの|例の|レイの|レイノ)", t, re.IGNORECASE):
        _log("[CMD] 例のものを再生命令検知")
        mp4_path = "/home/pi/Videos/mekamaru.mp4"
        if os.path.exists(mp4_path):
            # mpvで全画面再生
            if shutil.which("mpv"):
                subprocess.Popen(["mpv", "--fs", mp4_path])
                respond("例のものを再生します。")
            # VLC（GUI版）で全画面再生（デスクトップ表示）
            elif shutil.which("vlc"):
                subprocess.Popen(["vlc", "--fullscreen", mp4_path])
                respond("了解")
            # cvlc（コンソール版）
            elif shutil.which("cvlc"):
                subprocess.Popen(["cvlc", "--fullscreen", mp4_path])
                respond("例のものを再生します。（cvlc使用）")
            else:
                respond("申し訳ありません。動画再生ソフトが見つかりません。mpvまたはVLCが必要です。")
        else:
            respond("申し訳ありません。動画ファイルが見つかりません。")
        return True
    # ブラウザ・動画・閉じ系ワードが含まれていれば反応
    if re.search(r"(ブラウザ|動画).*(閉じ|終了|消|閉め|とじ|しめ)", t):
        _log("[CMD] ブラウザ/動画終了命令検知")
        close_browser()
        return True
    # ラッキーマイン最新動画の再生（誤認識吸収）
    if re.search(r"(ラッキー?マ[イい]ン|ラッキー?マリン|Lucky\s*Mine|ﾗｯｷｰﾏｲﾝ)", t) and \
       re.search(r"(最新|新しい)?\s*(動画|どうが|ビデオ|movie|video|再生|みたい|見たい|見せて)", t):
        _log("[CMD] ラッキーマイン最新動画 再生")
        url = get_latest_video_url()
        ok = launch_chromium_single(url)
        if ok:
            respond("了解。最新の動画を全画面で再生します。")
        else:
            respond("申し訳ありません。ブラウザを起動できませんでした。")
        return True
    return False

# ====== メイン ======
def main():
    if not play_wav(START_WAV, label="起動音"):
        respond("起動しました。マイクを待機します。")
    # 実効パラメータ（USBマイク前提でそのまま）
    eff_start_ms = VAD_START_MS
    eff_end_ms = VAD_END_MS
    eff_sil_ms = VAD_SILENCE_MS
    seg = Segmenter(vad_mode=VAD_MODE, start_ms=eff_start_ms, end_ms=eff_end_ms,
                    silence_ms=eff_sil_ms, bridging_ms=SEG_BRIDGING_MS)
    # 実行時パラメータを明示
    _log(f"[OKメカ丸] 使用USBマイク(alsa device): {ALSA_DEV or 'default/自動'}")
    eff_chunk_sec = CHUNK_SEC
    _log(
        f"[RUN] FRAME_MS={FRAME_MS} START(ms)={eff_start_ms} END(ms)={eff_end_ms} "
        f"SILENCE(ms)={eff_sil_ms} BRIDGE(ms)={SEG_BRIDGING_MS} MODE={VAD_MODE} "
        f"CHUNK_SEC={eff_chunk_sec} API={API_VER}"
    )
    _log("[OKメカ丸] 常時リスニング開始（VADで有声区間のみ送信）")
    while True:
        try:
            # mainループの最初で状態リセット
            last_text = None
            processed = False
            tmp = "/tmp/chunk.wav"
            if not record_chunk_wav(tmp, eff_chunk_sec):
                _log(f"[DIAG] 録音失敗: {tmp}")
                continue
            if not os.path.exists(tmp):
                _log(f"[DIAG] 録音ファイル未生成: {tmp}")
                continue
            sz = os.path.getsize(tmp)
            _log(f"[DIAG] chunk.wav size={sz}")
            if sz <= 44:
                _log(f"[DIAG] 録音ファイルが空です（44バイト）")
                continue
            # 録音内容を再生して確認（デバッグ用・任意）
            if DEBUG_PLAY and shutil.which("aplay"):
                _log(f"[DIAG] chunk.wav 再生（デバッグ）")
                subprocess.run(["aplay", "-q", tmp], check=False)
            # chunk.wavのフォーマット情報をログ出力
            try:
                with wave.open(tmp, "rb") as wf:
                    ch = wf.getnchannels()
                    rate = wf.getframerate()
                    sw = wf.getsampwidth()
                    _log(f"[DIAG] chunk.wav format: ch={ch}, rate={rate}, sw={sw}")
            except Exception as e:
                _log(f"[DIAG] chunk.wav format取得失敗: {e}")
            pcm = read_wav_pcm(tmp)
            if not pcm:
                _log(f"[DIAG] PCM抽出失敗: {tmp}")
                continue
            # モニタ再生（小音量・短時間）で録音の有無を可視化
            if MONITOR_PLAY and shutil.which("aplay"):
                try:
                    # 最もラウドな区間を抽出して可聴性を上げる
                    mon, off_sec = _pcm_loudest_slice(pcm, MONITOR_SEC)
                    # 極小入力なら減衰せずに再生（自動ゲイン選択）
                    mg = MONITOR_GAIN
                    if MONITOR_AUTO_GAIN:
                        rms = _pcm_rms(mon)
                        if rms < 50:
                            mg = 1.0
                    mon_play = mon if mg >= 0.999 else _pcm_attenuate(mon, mg)
                    write_wav_from_pcm("/tmp/chunk_monitor.wav", mon_play)
                    _log(f"[MONITOR] /tmp/chunk_monitor.wav 再生 gain={mg:.2f} sec={MONITOR_SEC} offset={off_sec:.2f}s")
                    subprocess.run(["aplay","-q","/tmp/chunk_monitor.wav"], check=False)
                except Exception:
                    pass
            # 平均音量を簡易診断（16bit）
            try:
                import array
                arr = array.array('h')
                arr.frombytes(pcm[: min(len(pcm), RATE * SAMPLE_WIDTH * 5)])  # 最大5秒分で評価
                mean_amp = sum(abs(x) for x in arr) / max(1, len(arr))
                _log(f"[DIAG] 平均音量(概算): {mean_amp:.1f}")
                if mean_amp < 50:
                    _log("[HINT] 入力が極小です。マイク位置やゲイン（amixer/alsamixer等）を調整してください。")
            except Exception:
                pass
            # AGC適用（任意）
            if AGC_ENABLE:
                pcm_agc = pcm_apply_agc(pcm)
                if pcm_agc != pcm:
                    pcm = pcm_agc
                    if DEBUG_PLAY and shutil.which("aplay"):
                        # デバッグ用にAGC後を再生可能
                        try:
                            write_wav_from_pcm("/tmp/chunk_agc.wav", pcm)
                        except Exception:
                            pass
            processed = False
            frames = pcm_to_frames(pcm, FRAME_MS)
            _log(f"[DIAG] frames数: {len(frames)}")
            try:
                _log(f"[DIAG] 推定録音秒数: {len(frames)*FRAME_MS/1000.0:.2f}s")
            except Exception:
                pass
            # 毎回新しいSegmenterを使う（状態持ち越し防止）
            seg = Segmenter(vad_mode=VAD_MODE, start_ms=eff_start_ms, end_ms=eff_end_ms,
                            silence_ms=eff_sil_ms, bridging_ms=SEG_BRIDGING_MS)
            segments = seg.consume_frames(frames)
            _log(f"[DIAG] VAD抽出区間数: {len(segments)}")
            # ラウドなセグメントを先に処理（一定音量が来たらそれを優先）
            order = list(range(len(segments)))
            if SEG_PICK_LOUDEST and len(segments) > 1:
                loud = [(_pcm_rms(s), i) for i, s in enumerate(segments)]
                order = [i for _, i in sorted(loud, key=lambda x: x[0], reverse=True)]
            # VAD区間優先で処理（chunk.wav全体で1回しか応答しない）
            for ord_i, idx in enumerate(order):
                seg_pcm = segments[idx]
                if processed:
                    break
                seg_path = f"/tmp/seg_{idx}.wav"
                write_wav_from_pcm(seg_path, seg_pcm)
                sz2 = os.path.getsize(seg_path) if os.path.exists(seg_path) else 0
                _log(f"[DIAG] seg_{idx}.wav size={sz2}")
                # 短すぎるセグメントはSTTをスキップ
                seg_ms = int((len(seg_pcm) / (RATE * SAMPLE_WIDTH)) * 1000)
                if seg_ms < MIN_SEG_MS:
                    _log(f"[INFO] セグメント{idx}が短すぎるためスキップ: {seg_ms}ms < {MIN_SEG_MS}ms")
                    continue
                if sz2 > 44 and DEBUG_PLAY and shutil.which("aplay"):
                    _log(f"[DIAG] seg_{idx}.wav 再生（デバッグ）")
                    subprocess.run(["aplay", "-q", seg_path], check=False)
                text = stt_from_pcm(seg_pcm)
                _log(f"[SEG STT] {text}")
                # 空/重複はスキップして次のセグメントへ（このchunkでは未処理のまま）
                if not text or (last_text and text == last_text):
                    _log(f"[DIAG] STT認識失敗または重複: seg_{idx}.wav")
                    continue
                last_text = text
                if handle_custom_commands(text):
                    processed = True
                    break
                calc = try_calc_result(text)
                if calc:
                    respond(calc)
                    processed = True
                    break
                trig, rest = extract_after_trigger(text)
                # トリガー外はこのセグメントを捨てて次へ（別セグメントに本命がある可能性）
                if not trig:
                    _log("[INFO] トリガー外（このセグメントは無視）")
                    continue
                _log(f'[抽出] OK以降: 「{rest}」')
                if handle_custom_commands(rest):
                    processed = True
                    break
                calc2 = try_calc_result(rest)
                if calc2:
                    respond(calc2)
                    processed = True
                    break
                # ここでAI応答（一般質問）
                ai_reply = mekamaru_ai_reply(rest)
                if not ai_reply:
                    _log("[INFO] AI応答が空。次のセグメントを確認")
                    continue
                if "ユーザー入力がない" in ai_reply or "入力してください" in ai_reply:
                    _log("[INFO] AI応答スキップ: ユーザー入力なし判定")
                    continue
                respond(ai_reply)
                processed = True
                break
            # VAD区間抽出が0件の場合の保険処理（未処理時のみ）
            if not processed:
                if len(segments) == 0:
                    _log("[WARN] VAD区間抽出が0件。chunk.wav全体をSTTにかけます。")
                else:
                    _log("[INFO] セグメントでは未処理。chunk.wav全体で保険STT実行。")
                text = stt_from_pcm(pcm)
                _log(f"[SEG STT] {text}")
                if not text or (last_text and text == last_text):
                    _log(f"[DIAG] STT認識失敗または重複: chunk.wav")
                    processed = True
                    continue
                last_text = text
                if handle_custom_commands(text):
                    processed = True
                    continue
                calc = try_calc_result(text)
                if calc:
                    respond(calc)
                    processed = True
                    continue
                trig, rest = extract_after_trigger(text)
                if not trig:
                    _log("[INFO] トリガー外（全体STT）")
                    processed = True
                    continue
                _log(f'[抽出] OK以降: 「{rest}」')
                if handle_custom_commands(rest):
                    processed = True
                    continue
                calc2 = try_calc_result(rest)
                if calc2:
                    respond(calc2)
                    processed = True
                    continue
                ai_reply = mekamaru_ai_reply(rest)
                if "ユーザー入力がない" in ai_reply or "入力してください" in ai_reply:
                    _log("[INFO] AI応答スキップ: ユーザー入力なし")
                    processed = True
                    continue
                respond(ai_reply)
                processed = True
            # クールダウンで多重反応を抑制
            if processed and COOLDOWN_SEC > 0:
                time.sleep(COOLDOWN_SEC)
        except KeyboardInterrupt:
            if not play_wav(EXIT_WAV, label="終了音"):
                respond("終了します。")
            _log("[OKメカ丸] 終了。")
            break
        except Exception as e:
            _log(f"[ERROR] mainループ例外: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
