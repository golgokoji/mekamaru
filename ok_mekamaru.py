# ====== 渋谷ライブカメラ表示 ======
SHIBUYA_LIVE_URLS = [
    "https://www.youtube.com/watch?v=tujkoXI8rWM&autoplay=1&mute=1",  # ANN公式
    "https://www.youtube.com/channel/UCWs8rt4ofGmdV4N6KQpP10Q/live?autoplay=1&mute=1",  # SHIBUYA SKY
    "https://www.skylinewebcams.com/en/webcam/japan/kanto/tokyo/tokyo-shibuya-scramble-crossing.html",  # SkylineWebcams
    "https://www.youtube.com/results?search_query=渋谷+スクランブル+交差点+ライブ"  # YouTube検索
]

def open_shibuya_live(close_first=True):
    # 渋谷スクランブル交差点ライブカメラを順に起動
    if close_first:
        close_browser()
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OKメカ丸 v7.5
#
# 【全体の処理の流れ】
# 1. 起動時に効果音を再生し、マイク入力を常時監視します。
# 2. 音声入力はVAD（無音区間検出）で有声部分のみ抽出し、Google STT v2でテキスト化します。
# 3. テキスト化された音声コマンドを以下の優先度で解析します：
#    a. カスタム命令（例：ラッキーマイン最新動画再生、ブラウザ終了）
#    b. 計算式（四則演算）→即答
#    c. 「OK」トリガー判定 → 以降の命令を再解析
#    d. 上記以外はAI（Gemini）で応答生成
# 4. 応答はTTS（音声合成）で返答、または効果音再生。
# 5. 終了時も効果音または音声で通知。
#
# 各関数には役割や引数・戻り値の説明コメントを付与しています。

import os, sys, time, wave, struct, shutil, unicodedata, subprocess, difflib, re, signal
from typing import Tuple, List
import webrtcvad
import google.generativeai as genai
from google.cloud import texttospeech, speech_v2 as speech

HOME = os.path.expanduser("~")
VOICES_DIR = os.path.join(HOME, "voices")
START_WAV = os.path.join(VOICES_DIR, "start.wav")
EXIT_WAV  = os.path.join(VOICES_DIR, "exit.wav")

RATE = 16000
SAMPLE_WIDTH = 2
FRAME_MS = 20
CHUNK_SEC = float(os.getenv("MEKAMARU_CHUNK_SEC", "0.4"))
# VAD/STT関連の環境変数（既定値で上書き可能）
VAD_MODE = int(os.getenv("MEKAMARU_VAD_MODE", "2"))
VAD_START_MS = int(os.getenv("MEKAMARU_VAD_START_MS", "200"))
VAD_END_MS   = int(os.getenv("MEKAMARU_VAD_END_MS", "500"))
VAD_SILENCE_MS = int(os.getenv("MEKAMARU_VAD_SILENCE_MS", "800"))
MAX_UTTER_SEC = int(os.getenv("MEKAMARU_MAX_UTTER_SEC", "15"))
SEG_BRIDGING_MS = int(os.getenv("MEKAMARU_SEG_BRIDGING_MS", "300"))
STT_SINGLE_UTTERANCE = int(os.getenv("MEKAMARU_STT_SINGLE_UTTERANCE", "1"))

SRC = os.getenv("MEKAMARU_SOURCE", None)

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

def play_wav(path: str, label: str = "") -> bool:
    # 指定したwavファイルをaplayで再生する。
    # labelはログ用のラベル。
    # 再生成功でTrue、失敗でFalse。
    if not path or not os.path.exists(path): return False
    if not shutil.which("aplay"):
        _log(f"[再生スキップ] aplay 不在: {path}"); return False
    if label: _log(f'再生「{label}」')
    subprocess.run(["aplay","-q",path], check=False); return True

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
        if not GCP_CRED or not os.path.exists(GCP_CRED):
            _log("[ERROR] GOOGLE_APPLICATION_CREDENTIALS 未設定/不正"); return False
        _log(f'もらった命令「{text}」')
        _log(f'Google Text to Speechで「{text}」を音声にしています。')
        _log('音声変換中。')
        client = texttospeech.TextToSpeechClient()
        inp = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="ja-JP", name=GC_VOICE_NAME)
        cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=GC_SPEAKING_RATE, pitch=GC_PITCH
        )
        resp = client.synthesize_speech(input=inp, voice=voice, audio_config=cfg)
        raw = "/tmp/mekamaru_tts_raw.wav"
        with open(raw,"wb") as f: f.write(resp.audio_content)
        play_path = raw
        if USE_SOX_FX and shutil.which("sox"):
            fx = "/tmp/mekamaru_tts_fx.wav"
            if subprocess.run(["sox", raw, fx, "treble","+3","reverb","10"],
                              capture_output=True, text=True).returncode == 0 and os.path.exists(fx):
                play_path = fx
        _log(f'再生「「{text}」」')
        subprocess.run(["aplay","-q",play_path], check=False)
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
def record_chunk_wav(path: str, seconds: float) -> bool:
    # 指定秒数だけマイクから録音しwavファイルに保存。
    # 成功でTrue、失敗でFalse。
    if not SRC:
        _log("[ERROR] MEKAMARU_SOURCE 未設定"); return False
    if not shutil.which("pw-record"):
        _log("[ERROR] pw-record 不在"); return False
    cmd = ["pw-record","--rate",str(RATE),"--channels","1","--target",SRC,path]
    r = subprocess.run(["bash","-lc", f"timeout {seconds}s " + " ".join(cmd)],
                       capture_output=True, text=True)
    return (r.returncode in (0,124)) and os.path.exists(path) and (os.path.getsize(path) > 44)

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
        for fr in frames:
            is_voiced = self.vad.is_speech(fr, RATE)
            if self.state == "idle":
                if is_voiced:
                    self.run_ms += FRAME_MS
                    if self.run_ms >= self.start_need:
                        self.state = "run"
                        self.buf.extend(fr)
                        self.sil_ms = 0
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
        return segments  # 必ずリストを返す

# ====== WAV書き出し ======
def write_wav_from_pcm(path: str, pcm: bytes):
    # PCMデータをwavファイルとして保存する。
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
        wf.writeframes(pcm)

# ====== STT v2 ======
def _recognizer_path() -> str:
    # Google STT v2用のrecognizerパスを生成。
    if not GC_PROJECT_NUMBER:
        raise RuntimeError("MEKAMARU_GC_PROJECT_NUMBER が未設定です。")
    return f"projects/{GC_PROJECT_NUMBER}/locations/{GC_LOCATION}/recognizers/_"

def stt_from_pcm_google(pcm: bytes) -> str:
    # PCM音声をGoogle STT v2でテキスト化して返す。
    # 失敗時は空文字。
    tmp = "/tmp/seg.wav"
    write_wav_from_pcm(tmp, pcm)
    try:
        client = speech.SpeechClient()
        with open(tmp,"rb") as f: content = f.read()
        cfg = {
            "auto_decoding_config": {},
            "language_codes": ["ja-JP"],
            "model": "latest_short",
            "features": {"enable_automatic_punctuation": True},
        }
        req = speech.RecognizeRequest(
            recognizer=_recognizer_path(),
            config=cfg,
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

# ====== 応答短文化（AI用） ======
def _shorten(text: str, max_chars: int) -> str:
    # テキストを最大文字数で切り詰めて返す。
    s = (text or "").strip()
    if len(s) <= max_chars: return s
    return s[:max_chars-1].rstrip() + "…"

STYLE_GUIDE = (
    "出力規則: 一人称は『私』。自分を第三人称で呼ばない。"
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
def _ok_fuzzy_hit(text: str) -> Tuple[bool,int]:
    # 「OK」トリガーのファジーマッチ判定。
    # ヒット時は(真,位置)を返す。
    if not text: return (False,-1)
    src_h = _to_hiragana(_nfkc_lower(text)).replace("ー","")
    window = src_h[:30]
    targets = ["おっけ","おけ","おーけ","ok"]
    for t in targets:
        pos = window.find(t)
        if pos != -1: return (True, pos+len(t))
    split_idx = min([window.find(d) for d in "、。,. 　:：-—–!！?？「」『』()（）" if window.find(d)!=-1] or [len(window)])
    head = window[:split_idx]
    for t in targets:
        if difflib.SequenceMatcher(None, head, t).ratio() >= 0.7:
            return (True, len(head))
    return (False,-1)
def extract_after_ok(text: str) -> Tuple[bool,str]:
    # 「OK」以降の命令部分を抽出。
    # (ヒット,残りテキスト)を返す。
    hit,end = _ok_fuzzy_hit(text)
    if not hit: return (False,"")
    rest = text[end:].lstrip("、。,. 　:：-—–!！?？「」『』()（）")
    return (True, rest.strip())

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
    # メインループ。
    # マイク入力→VAD→STT→コマンド解析→応答生成→音声出力の流れ。
    # Ctrl+Cで終了。
    if not play_wav(START_WAV, label="起動音"):
        respond("起動しました。マイクを待機します。")
    seg = Segmenter(vad_mode=VAD_MODE, start_ms=VAD_START_MS, end_ms=VAD_END_MS, silence_ms=VAD_SILENCE_MS, bridging_ms=SEG_BRIDGING_MS)
    _log("[OKメカ丸] 常時リスニング開始（VADで有声区間のみ送信）")
    try:
        while True:
            tmp = "/tmp/chunk.wav"
            if not record_chunk_wav(tmp, CHUNK_SEC):
                try:
                    client = speech.SpeechClient()
                    with open(tmp,"rb") as f: content = f.read()
                    cfg = {
                        "auto_decoding_config": {},
                        "language_codes": ["ja-JP"],
                        "model": "latest_short",
                        "features": {"enable_automatic_punctuation": True},
                        "single_utterance": bool(STT_SINGLE_UTTERANCE),
                    }
                    req = speech.RecognizeRequest(
                        recognizer=_recognizer_path(),
                        config=cfg,
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
                if not trig:
                    _log("[INFO] トリガー外")
                    continue
                _log(f'[抽出] OK以降: 「{rest}」')
                if not rest:
                    respond("はい。")
                    continue

                # 4) “OK以降” にも計算や直命令が含まれる場合を再チェック
                if handle_custom_commands(rest):
                    continue
                calc2 = try_calc_result(rest)
                if calc2:
                    respond(calc2)
                    continue

                # 5) 通常はAIへ → 短文化 → 発話
                _log(f'命令受領「{rest}」')
                ai = mekamaru_ai_reply(rest)
                _log(f"[AI] {ai}")
                if not ai:
                    respond("それは無理だ")
                else:
                    respond(ai)
    except KeyboardInterrupt:
        if not play_wav(EXIT_WAV, label="終了音"):
            respond("終了します。")
        _log("[OKメカ丸] 終了。")

if __name__ == "__main__":
    main()
