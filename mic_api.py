# mic_api.py
# VADで音声の切れ目を検出→OpenAI Whisperでテキスト変換→control.pyへ送信
import socket
import sounddevice as sd
import numpy as np
import webrtcvad
import time
from scipy.signal import butter, lfilter
import wave
import datetime
import unicodedata
import threading
import os
from dotenv import load_dotenv
import json
import socketserver
from threading import Lock
import sys
import tempfile
import traceback
from openai import OpenAI

# 実行時刻のファイル名
this_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputfile = f"./log/mic_text_{this_time}.txt"

# グローバル状態（TTSメータ）
playback_rms = 0.0    # 0..1
last_meter_ts = 0.0   # 最終受信時刻
_state_lock = Lock()

# VAD設定
vad = webrtcvad.Vad(0)  # 0~3（感度）高いほど無音を検出しやすい
samplerate = 16000
frame_duration = 30  # ms
frame_size = int(samplerate * frame_duration / 1000)
min_duration = 0.8  # 秒
min_bytes = int(samplerate * min_duration * 2)  # 2バイト = int16

# 誤作動防止のための設定
MIN_TEXT_LENGTH = 5  # 最小テキスト長（文字数）- 意味のある発言のため引き上げ
MAX_TEXT_LENGTH = 500  # 最大テキスト長（文字数）
SILENCE_THRESHOLD = 0.3  # 無音検出の閾値（秒）
SILENCE_GRACE_PERIOD = 1.5  # 無音検出の猶予期間（秒）- 発言途中の間を許容

def init_params(file_path):
    load_dotenv(file_path)
    return {
        "control_port": int(os.getenv("CONTROL_PORT", 50000)),
        "mic_port": int(os.getenv("MIC_PORT", 50001)),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "filter_model": os.getenv("FILTER_MODEL", "gpt-4o-mini"),
        "filter_confidence_threshold": float(os.getenv("FILTER_CONFIDENCE_THRESHOLD", "0.6")),
        "enable_llm_filter": os.getenv("ENABLE_LLM_FILTER", "true").lower() == "true",
    }

def hankaku_to_zenkaku(text):
    new_text = ''
    for c in text:
        if unicodedata.east_asian_width(c) in ('Na', 'H'):
            try:
                new_text += unicodedata.normalize('NFKC', c)
            except:
                new_text += c
        else:
            new_text += c
    return new_text

def sanitize_filename(text):
    text = text.replace(" ", "　")
    for ch, zch in zip(r'\/:*?"<>|', "￥／：＊？＜＞｜"):
        text = text.replace(ch, zch)
    text = hankaku_to_zenkaku(text)
    return text

def save_wav(filename, audio_bytes, samplerate=16000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_bytes)

wav_set = 0

def bandpass_filter(audio, sr=16000, lowcut=200, highcut=4000):
    b, a = butter(2, [lowcut/(sr/2), highcut/(sr/2)], btype='band')
    return lfilter(b, a, audio)

def set_filter(audio):
    # ① バイト列→int16→float32
    np_audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    # ② フィルタ
    np_audio = bandpass_filter(np_audio, sr=samplerate)
    # ③ int16に戻す（クリッピングも推奨）
    np_audio = np.clip(np_audio, -32768, 32767).astype(np.int16)
    # ④ bytes化
    return np_audio.tobytes()

def text_output(text):
    global outputfile
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    with open(outputfile, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, samplerate)

def output(duration, speech_time, text, data):
    global wav_set, this_time
    print(f"録音時間: {duration:.2f}")
    print(f"発話時間: {speech_time}")
    print(f"信頼性: {speech_time / duration:.2f}")
    out_text = f"{wav_set}_{text}"
    wav_set += 1
    text_output(out_text)
    save_wav(f"./log/{this_time}_{out_text}.wav", data)

# TTSからの通知を受ける軽量サーバ
class ControlHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global playback_rms, last_meter_ts
        try:
            data = self.request.recv(4096)
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            return
        if "meter" in msg:
            with _state_lock:
                playback_rms = float(msg["meter"])
                last_meter_ts = time.time()

def start_control_server(port: int):
    srv = socketserver.TCPServer(("0.0.0.0", port), ControlHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[mic] control server listening on {port}")

# OpenAI Whisperで文字起こし
def transcribe_with_openai(audio_bytes, params, sr=16000):
    """
    16kHz/mono/PCM16のWAVバイト列をOpenAIのgpt-4o-mini-transcribeで文字起こし。
    失敗時は空文字を返す。
    """
    api_key = params.get("openai_api_key", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY が設定されていません（.envに追記してください）")
        return ""

    client = OpenAI(api_key=api_key)

    # 一時ファイルにWAVとして保存
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16bit PCM
            wf.setframerate(sr)
            wf.writeframes(audio_bytes)

        with open(tmp.name, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language="ja",
            )
        text = getattr(resp, "text", "") or ""
        return text.strip()
    except Exception as e:
        print("OpenAI STT error:", e)
        traceback.print_exc()
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def is_complete_sentence(text):
    """
    機械的に文が完結しているかをチェックする
    句読点や終助詞の存在を確認
    """
    if not text:
        return False
    
    # 句読点チェック
    if text.endswith(("。", "！", "？", ".", "!", "?")):
        return True
    
    # 終助詞チェック（よくある終助詞パターン）
    ending_patterns = [
        "です", "ます", "ました", "でした",
        "ね", "よ", "な", "か", "わ",
        "だね", "だよ", "だな", "だわ",
        "ですね", "ですよ", "ますね", "ますよ",
        "ません", "ませんでした",
        "ください", "ましょう", "でしょう",
    ]
    
    for pattern in ending_patterns:
        if text.endswith(pattern):
            return True
    
    return False

def check_utterance_completeness_llm(text, params):
    """
    GPT-4o-miniを使って発言の完全性を判定する
    
    戻り値:
        dict: {
            "is_complete": bool,  # 完全な発言かどうか
            "confidence": float,  # 確信度 (0.0-1.0)
            "reason": str        # 判定理由
        }
    """
    api_key = params.get("openai_api_key", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY が設定されていません。LLMフィルタをスキップします。")
        return {"is_complete": True, "confidence": 0.0, "reason": "API key not set"}
    
    if not params.get("enable_llm_filter", True):
        print("LLMフィルタが無効化されています。")
        return {"is_complete": True, "confidence": 1.0, "reason": "LLM filter disabled"}
    
    client = OpenAI(api_key=api_key)
    model = params.get("filter_model", "gpt-4o-mini")
    
    prompt = f"""以下の発言が完全な文か、途中で切れている不完全な文かを判定してください。

判定基準:
- 完全: 文として意味が完結している、主語・述語が揃っている、文末が適切
- 不完全: 文が途中で終わっている、主語や述語が欠けている、フィラーのみ

発言: "{text}"

以下のJSON形式で回答してください:
{{
  "is_complete": true または false,
  "confidence": 0.0から1.0の数値,
  "reason": "判定理由を簡潔に"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは日本語の文の完全性を判定する専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        print(f"[LLMフィルタ] 発言: '{text[:30]}...' -> 完全性: {result.get('is_complete')}, 確信度: {result.get('confidence'):.2f}")
        return result
        
    except Exception as e:
        print(f"LLMフィルタエラー: {e}")
        traceback.print_exc()
        # エラー時はフィルタリングしない（発言を通す）
        return {"is_complete": True, "confidence": 0.0, "reason": f"Error: {str(e)}"}

def should_filter_text(text, params):
    """
    テキストをフィルタリングすべきかを判定する
    
    段階的フィルタリング:
    1. 機械的フィルタ（高速）
    2. LLMフィルタ（精密）
    
    戻り値:
        tuple: (should_filter: bool, reason: str)
        - should_filter: Trueならフィルタリング（破棄）
        - reason: フィルタリング理由
    """
    # ステージ1: 機械的フィルタ
    if len(text) < MIN_TEXT_LENGTH:
        return True, f"テキストが短すぎます（{len(text)}文字 < {MIN_TEXT_LENGTH}文字）"
    
    if len(text) > MAX_TEXT_LENGTH:
        # 長すぎる場合は切り詰めるが、フィルタリングはしない
        print(f"テキストが長すぎます（{len(text)}文字 > {MAX_TEXT_LENGTH}文字）。切り詰めます。")
        return False, "OK (truncated)"
    
    # まず機械的に完全性をチェック
    if is_complete_sentence(text):
        print(f"[機械的フィルタ] 完全な文と判定: '{text[:30]}...'")
        return False, "OK (mechanically complete)"
    
    # ステージ2: LLMフィルタ（機械的フィルタで不完全と判定された場合のみ）
    if params.get("enable_llm_filter", True):
        llm_result = check_utterance_completeness_llm(text, params)
        is_complete = llm_result.get("is_complete", True)
        confidence = llm_result.get("confidence", 0.0)
        reason = llm_result.get("reason", "")
        
        threshold = params.get("filter_confidence_threshold", 0.6)
        
        # 確信度が閾値以上の場合のみLLMの判定を信頼
        if confidence >= threshold:
            if not is_complete:
                return True, f"LLMフィルタ: 不完全な発言（確信度: {confidence:.2f}, 理由: {reason}）"
            else:
                return False, f"LLMフィルタ: 完全な発言（確信度: {confidence:.2f}）"
        else:
            # 確信度が低い場合は通す（偽陽性を避けるため）
            print(f"[LLMフィルタ] 確信度が低いため通します（{confidence:.2f} < {threshold}）")
            return False, f"OK (low confidence: {confidence:.2f})"
    
    # LLMフィルタが無効、または機械的フィルタで判定できなかった場合は通す
    print(f"[フィルタ] 判定不能のため通します: '{text[:30]}...'")
    return False, "OK (uncertain)"

def send_audio(full_data, params):
    """
    音声データをOpenAI Whisperで文字起こししてcontrol.pyへ送信
    段階的フィルタリング（機械的 → LLM）で不完全な発言を除外
    """
    global samplerate

    data = set_filter(full_data)
    duration = len(full_data) / samplerate / 2.0

    # 誤作動防止: 短すぎる音声はスキップ
    if duration < min_duration:
        print(f"音声が短すぎます（{duration:.2f}秒 < {min_duration}秒）。スキップします。")
        return 0.0, "", duration

    # OpenAIで文字起こし
    text = transcribe_with_openai(data, params, sr=samplerate)

    # 誤作動防止: テキストの検証
    if not text:
        print("文字起こし結果が空でした。スキップします。")
        return 0.0, "", duration

    # 段階的フィルタリング（機械的 + LLM）
    should_filter, filter_reason = should_filter_text(text, params)
    if should_filter:
        print(f"フィルタリング: {filter_reason}")
        return 0.0, "", duration
    
    # 長すぎる場合は切り詰め
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    s_text = sanitize_filename(text)
    conf = 1.0  # OpenAI Whisperは信頼性が高いため1.0
    s_time = duration

    output(duration, s_time, s_text, full_data)
    print(f"[フィルタ通過] '{text[:50]}...' (理由: {filter_reason})")
    return conf, s_text, duration

def start_client(ip, port, message):
    """control.pyへメッセージを送信"""
    try:
        print(f"start_client: {message}\tport:{port}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, port))
        message = json.dumps(message, ensure_ascii=False)
        client_socket.send(message.encode("utf-8"))
        client_socket.close()
    except Exception as e:
        print(f"クライアントエラー: {e}")
        traceback.print_exc()

def record(params):
    """メイン録音ループ: VADで音声検出→Whisperで文字起こし→control.pyへ送信"""
    was_speaking = False
    notified_speaking = False  # speaking:True通知を送ったかどうか
    print("🎙️ Listening ...")
    
    with sd.RawInputStream(samplerate=samplerate, channels=1, dtype="int16", blocksize=frame_size) as stream:
        all_buffer = bytearray()

        while True:
            frame, _ = stream.read(frame_size)
            is_speaking = is_speech(frame)
            
            # 自己一致フィルタ（エコー防止）
            mic_rms = float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32)**2)) / 32768.0)

            with _state_lock:
                prms = playback_rms
                age = time.time() - last_meter_ts
            meter_valid = (age < 0.3)  # 直近300ms以内のメータだけ有効

            leak_factor = 0.25  # スピーカー漏れの想定割合
            offset = 0.005  # -46dBFS相当の底上げ
            if meter_valid:
                likely_echo = mic_rms <= (prms * leak_factor + offset)
            else:
                likely_echo = False

            if not likely_echo:
                if is_speaking:
                    all_buffer.extend(frame)
                    if not was_speaking:
                        # 最初の音声検出時は即座にwas_speakingをTrueにする
                        was_speaking = True
                        notified_speaking = False
                        print(f"[DEBUG] 音声検出開始: buffer={len(all_buffer)}バイト, min={min_bytes}バイト")
                    # min_bytesに達したら通知を送る（1回だけ）
                    if len(all_buffer) >= min_bytes and was_speaking and not notified_speaking:
                        print(f"speaking:True")
                        start_client("localhost", params["control_port"], {"speaking": "Yes"})
                        notified_speaking = True
                else:
                    if was_speaking:
                        # 音声が終了した時、min_bytes以上なら処理する
                        if len(all_buffer) >= min_bytes:
                            was_speaking = False
                            notified_speaking = False
                            conf, text, duration = send_audio(all_buffer, params)
                            if conf > 0.0 and text:
                                print(f"user:{text}")
                                # control.pyへテキストを送信
                                start_client("localhost", params["control_port"], {
                                    "user": "",
                                    "text": text,
                                    "speaking": "No"
                                })
                            else:
                                print(f"speaking:False (文字起こし失敗)")
                                start_client("localhost", params["control_port"], {"speaking": "Error"})
                        else:
                            # 短すぎる音声はスキップ
                            print(f"音声が短すぎます（{len(all_buffer)}バイト < {min_bytes}バイト）。スキップします。")
                            was_speaking = False
                            notified_speaking = False
                    all_buffer.clear()

if __name__ == "__main__":
    if "--file" in sys.argv:
        params_file = sys.argv[sys.argv.index("--file") + 1]
    else:
        params_file = ".env"
    
    params = init_params(params_file)

    # ログ格納先の用意
    os.makedirs("./log", exist_ok=True)

    socket_thread = threading.Thread(target=record, args=(params,), daemon=True)
    socket_thread.start()
    start_control_server(int(params.get("mic_port", 5001)))

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Bye.")
            break

