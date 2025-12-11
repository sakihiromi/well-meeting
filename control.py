# control.py
# 中心となるコントロールサーバ
# LLMカットを使用してスコアリングと確信度を取得・保存
from dotenv import load_dotenv
import os
import json
import pandas as pd
import threading
import sys
import time
from my_node import Node
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
from llmcut import LLMCut
import socket
import traceback
import statistics

# =========================
# extra_info の初期化関数
# =========================
def load_extra_info(extra_json_file: str) -> Dict[str, Any]:
    """
    extra.json から指標定義を読み込む。
    なければデフォルト定義（4指標）を書き出して返す。
    """
    try:
        with open(extra_json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # デフォルト定義（ユーザ要件に合わせた4指標）
        default_extra = {
            "威圧度": {
                "定義": "発言の威圧性（言葉の強さ・強要・皮肉・相手に与える心理的プレッシャー）を評価する。",
                "スコア基準": {
                    "低(1-3)": "敬意・配慮があり威圧的でない。",
                    "中(4-6)": "やや強め・配慮不足が一部見られる。",
                    "高(7-9)": "強要・侮蔑・恫喝など過度な圧力。"
                },
                "質問": "この発言は不当な圧力や威圧感を与えていないか？"
            },
            "逸脱度": {
                "定義": "会議の目的からの逸脱度合いを評価する。",
                "スコア基準": {
                    "低(1-3)": "目的に沿う内容。",
                    "中(4-6)": "一部関連するが焦点がぼやける。",
                    "高(7-9)": "無関係・進行妨害。"
                },
                "質問": "この発言は会議の目的に直接関係しているか？"
            },
            "発言無効度": {
                "定義": "発言が目的達成・理解深化・合意形成にどれだけ寄与していないか（無効さ）を評価する。",
                "スコア基準": {
                    "低(1-3)": "無効度が低い（＝高い有効性）。目的達成に明確に寄与し、理解や共通理解の形成に貢献。",        
                    "中(4-6)": "一部に無効さが見られる。部分的には目的達成に寄与するが、目的との結びつきや反映が不十分。",
                    "高(7-9)": "無効度が高い。目的達成や共通理解の形成にほとんど寄与せず、会議を停滞させる。"
                },
                "質問": "この発言はどの程度「無効」か？会議目的の達成や共通理解の形成を妨げていないか？"
            },
            "偏り度": {
                "定義": "討議の手続的公正（発言機会の平等・代表性・倫理性など）に対する偏り。",
                "スコア基準": {
                    "低(1-3)": "偏り抑制が保たれている。",
                    "中(4-6)": "一部の偏りが見られる。",
                    "高(7-9)": "独占・抑制・代表性欠如。"
                },
                "質問": "このやり取りは私的バイアスに強く引っ張られていないか？"
            }
        }
        with open(extra_json_file, 'w', encoding='utf-8') as f:
            json.dump(default_extra, f, ensure_ascii=False, indent=4)
        return default_extra

# ===============
# 環境変数の読込
# ===============
def init_params(file_path: str) -> Dict[str, Any]:
    load_dotenv(file_path)
    return {
        "control_port": int(os.getenv("CONTROL_PORT", 50000)),
        "mic_port": int(os.getenv("MIC_PORT", 50001)),
        "mic_ip": os.getenv("MIC_IP", "localhost"),
        "api_port": int(os.getenv("API_PORT", 8008)),
        "api_ip": os.getenv("API_IP", "localhost"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "endpoint": os.getenv("OPENAI_ENDPOINT", ""),
        "api_version": os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-2024-08-06"),
        "user_name": os.getenv("USER_NAME", ""),
        "meeting_type": os.getenv("MEETING_TYPE", ""),
        "meeting_goal": os.getenv("MEETING_GOAL", ""),
        "eval_lines": int(os.getenv("EVAL_LINES", "10")),
        "extra_json_file": os.getenv("EXTRA_JSON_FILE", "extra.json"),
        "extra_info": load_extra_info(os.getenv("EXTRA_JSON_FILE", "extra.json")),
        "selected_metrics": [m.strip() for m in os.getenv("SELECTED_METRICS", "").split(",") if m.strip()] if os.getenv("SELECTED_METRICS", "") else [],
    }

# =========================
# プロンプト生成（複数指標）
# =========================
def make_prompt_multi(past_text: List[str],
                      meeting_type: str,
                      meeting_goal: str,
                      metrics_list: List[str],
                      eval_lines: int,
                      extra_info: Dict[str, Any]) -> Dict[str, str]:

    # 会議の前提
    if meeting_goal and meeting_goal.strip():
        if meeting_type and meeting_type.strip():
            ex_topic = f'この会議の形式は「{meeting_type}」で、その目的は「{meeting_goal}」です。\n'
        else:
            ex_topic = f'この会議の目的は「{meeting_goal}」です。\n'
    elif meeting_type and meeting_type.strip():
        ex_topic = f'この会議の形式は「{meeting_type}」です。\n'
    else:
        ex_topic = ''

    # 直近発話の抽出
    start_line = max(0, len(past_text) - eval_lines)
    extra_text = "\n".join(past_text[start_line:])

    # 指標定義の整形
    metrics_explanations = {}
    for metric in metrics_list:
        if metric in extra_info:
            info = extra_info[metric]
            if isinstance(info, dict) and "定義" in info:
                definition = info.get("定義", "")
                score_criteria = info.get("スコア基準", {})
                question = info.get("質問", "")
                explanation = f"「{metric}」: {definition}"
                if score_criteria:
                    explanation += "\nスコア基準:"
                    for k, v in score_criteria.items():
                        explanation += f"\n- {k}: {v}"
                if question:
                    explanation += f"\n評価時の質問: {question}"
                metrics_explanations[metric] = explanation
            else:
                metrics_explanations[metric] = f"「{metric}」: {info}"
    
    sample_text = (
        '以下に数例示します\n'
        '「馬、鹿野郎」は「馬鹿野郎」のASR誤りとみなす。\n'
        '「20起動」は「二重起動」のASR誤りとみなす。\n'
    )
    prompts = {}
    for metric in metrics_list:
        prompts[metric] = {'prompt': (
            f'{ex_topic}以下の発話に対して、発話内容から「{metric}」を1~9でスコアリングしてください。\n'
            f'{metrics_explanations[metric]}\n\n'
            f'これらの文章は日本語ASRの文字起こしです。意味不明箇所は誤りを考慮し、文脈補完して評価してください。\n'
            f'{sample_text}'
            f'＜＜過去発話（直近）＞＞\n{extra_text}\n＜＜過去発話終わり＞＞\n\n'
            f'【重要】スコアリングと確信度（confidence）は絶対に使用してください。\n'
            f'出力はスコアの整数即ち数値1文字のみで返してください：\n'
        )}
    return prompts

# =========================
# 総合スコアの作成
# =========================
def make_total_score(global_df: pd.DataFrame, row_index: int, metrics_list: List[str]):
    """各スコアが出そろうまで待機"""
    while True:
        
        if global_df.at[row_index, 'title'] == "総合(平均)":
            print(f"総合(平均)が出そろった: {global_df.at[row_index, 'text']}")
            break
        if global_df.at[row_index, 'metrics'] is not None:
            if len(global_df.at[row_index, 'metrics']) == len(metrics_list):
                per_scores = []
                per_confs = []
                for metric in metrics_list:
                    per_scores.append(global_df.at[row_index, f"{metric}スコア"])
                    per_confs.append(global_df.at[row_index, f"{metric}確信度"])

                global_df.at[row_index, 'title'] = "総合(平均)"
                global_df.at[row_index, 'score'] = int(round(statistics.mean(per_scores))) if per_scores else None
                global_df.at[row_index, 'conf'] = round(float(statistics.mean(per_confs)), 2) if per_confs else None
                print(f"総合(平均)が出そろった: {global_df.at[row_index, 'text']}")
                break
        time.sleep(1)
    

# ================
# Main Node
# ================
class MainNode(Node):
    def __init__(self, host, port, global_df, params, lock):
        super().__init__(host, port, None, None)
        self.global_df = global_df
        self.params_ref = params
        self.lock = lock
        self.filename = "meeting_keeper.csv"
        # LLMカットを初期化（確信度を使用）
        self.llmcut = LLMCut(
            self.params_ref["api_key"], 
            template_path='prompt_template1.json',
            endpoint=self.params_ref.get("endpoint"),
            model_name=self.params_ref["model_name"]
        )
        self.past_text: List[str] = []
        self.metrics_to_evaluate = self.params_ref["selected_metrics"] if self.params_ref["selected_metrics"] else list(self.params_ref["extra_info"].keys())

    def process_data(self, data):
        """
        マイクからテキストを受け取って分析し、結果を保存する
        【重要】LLMカットを使用してスコアリングと確信度を取得
        """
        try:
            dict_data = json.loads(data)
            if "user" not in dict_data or "text" not in dict_data:
                print(f"不正なデータ形式: {data}")
                return "Error: Invalid data format"
            
            if not dict_data.get('text', '').strip():
                print("空のテキストはスキップします")
                return "Skipped: Empty text"

            # ユーザー名の設定
            if dict_data['user'] == "" and self.params_ref["user_name"] != "":
                dict_data['user'] = self.params_ref["user_name"]

            timestamp = pd.Timestamp.now().isoformat()
            
            with self.lock:
                idx = len(self.global_df)
                self.global_df.loc[idx, 'user'] = dict_data['user']
                self.global_df.loc[idx, 'text'] = dict_data['text']
                self.global_df.loc[idx, 'timestamp'] = timestamp
                self.global_df.loc[idx, 'title'] = "評価中"
                self.global_df.loc[idx, 'score'] = None
                self.global_df.loc[idx, 'conf'] = None
                self.global_df.loc[idx, 'metrics'] = None

            # プロンプト生成
            tmp_prompts = make_prompt_multi(
                self.past_text,
                self.params_ref["meeting_type"],
                self.params_ref["meeting_goal"],
                self.metrics_to_evaluate,
                self.params_ref["eval_lines"],
                self.params_ref["extra_info"]
            )

            # 各指標を並列で評価（LLMカットを使用）
            threads = []
            for metric in self.metrics_to_evaluate:
                thread = threading.Thread(
                    target=self.start_llmcut, 
                    args=(dict_data['text'], tmp_prompts[metric], idx, metric)
                )
                thread.start()
                threads.append(thread)
            
            # 全てのスレッドの完了を待つ
            for t in threads:
                t.join()

            # 直近履歴に追加
            with self.lock:
                if idx < len(self.global_df):
                    self.past_text.append(self.global_df.iloc[idx]['text'])
            
            # 総合の作成
            make_total_score(self.global_df, idx, self.metrics_to_evaluate)

            print(f"処理完了: {dict_data['text'][:50]}...")
            return "Main Processed"

        except Exception as e:
            print(f"process_data エラー: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def start_llmcut(self, text, prompt, row_index: int, metric_to_evaluate: str):
        """
        LLMカットを使用してスコアリングと確信度を取得する。
        【重要】スコアリングと確信度（confidence）は絶対に使用する。
        v_meeting2の構造をそのまま使用。
        """
        try:
            print(f"複数指標評価開始: {metric_to_evaluate}")
            print(f"Model: {self.params_ref['model_name']}")
            
            df = pd.DataFrame({'text': [f"{text}"]})
            # LLMカットを使用してスコアリングと確信度を取得（絶対に使用）
            res = self.llmcut.add_label(prompt, df['text'])
            print(f"LLM応答: {res}")
            
            # スコアリングと確信度を取得（絶対に使用）
            score = int(res['score'].iloc[0])
            conf = float(res['conf'].iloc[0])
            
            metrics = {"指標": metric_to_evaluate, "スコア": score, "確信度": conf}
            m = metric_to_evaluate
            sc = f"{m}スコア"
            cf = f"{m}確信度"
            
            with self.lock:
                # metrics 配列
                if self.global_df.at[row_index, 'metrics'] is None:
                    self.global_df.at[row_index, 'metrics'] = []
                self.global_df.at[row_index, 'metrics'].append(metrics)
                
                # 列が無ければ作る
                if sc not in self.global_df.columns:
                    self.global_df[sc] = None
                if cf not in self.global_df.columns:
                    self.global_df[cf] = None
                
                self.global_df.at[row_index, sc] = score
                self.global_df.at[row_index, cf] = conf
                
                # CSV保存
                self.global_df.to_csv(self.filename, index=False)
            
            print(f"評価完了: row={row_index} を横持ちで更新（スコア={score}, 確信度={conf}）")
            
        except Exception as e:
            print(f"start_llmcut エラー ({metric_to_evaluate}): {e}")
            traceback.print_exc()

    def start_server(self):
        """TCPサーバを起動してmic_api.pyからの接続を待つ"""
        host = self.host
        port = self.port
        print(f"[MainNode] TCPサーバを起動します: {host}:{port}")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(5)
            print(f"[MainNode] LISTEN中: {host}:{port}")
        except Exception as e:
            print(f"[MainNode] サーバ起動失敗: {host}:{port} -> {e}")
            traceback.print_exc()
            return

        while True:
            try:
                conn, addr = srv.accept()
                # 1コネクションをスレッドで処理
                th = threading.Thread(target=self._handle_conn, args=(conn, addr), daemon=True)
                th.start()
            except KeyboardInterrupt:
                print("[MainNode] KeyboardInterrupt: サーバを終了します")
                break
            except Exception as e:
                print(f"[MainNode] accept エラー: {e}")
                traceback.print_exc()

    def _handle_conn(self, conn, addr):
        """クライアント接続を処理"""
        try:
            raw = conn.recv(65536)
            if not raw:
                return
            try:
                payload = raw.decode("utf-8", errors="ignore")
            except Exception:
                payload = ""
            if payload:
                self.process_data(payload)
        except Exception as e:
            print(f"[MainNode] クライアント処理エラー from {addr}: {e}")
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass

# ================
# FastAPI (表示用)
# ================
class MyAPI:
    def __init__(self, global_df, params, lock):
        self.app = FastAPI(title="meeting_keeper API", version="0.1.0")
        self.global_df = global_df
        self.params = params
        self.lock = lock
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._mount_routes()

    def _mount_routes(self) -> None:
        @self.app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @self.app.get("/params")
        def get_params() -> Dict[str, Any]:
            safe = {k: v for k, v in self.params.items() if k not in ("api_key",)}
            return {"params": safe}

        @self.app.get("/data")
        def get_data() -> Dict[str, List[Dict[str, Any]]]:
            with self.lock:
                df = self.global_df.copy()
            for col in ["user", "text", "metrics", "title", "score", "conf", "timestamp"]:
                if col not in df.columns:
                    df[col] = None
            records = df.fillna(value=pd.NA).astype(object).where(pd.notna(df), None).to_dict(orient="records")
            return {"rows": records}

        @self.app.post("/update_settings")
        def update_settings(settings_data: Dict[str, str]) -> Dict[str, str]:
            try:
                new_type = settings_data.get("meeting_type", "")
                new_goal = settings_data.get("meeting_goal", "")
                with self.lock:
                    self.params["meeting_type"] = new_type
                    self.params["meeting_goal"] = new_goal
                return {"status": "success", "message": f"会議の設定を更新しました（形式: {new_type}, 目的: {new_goal}）"}
            except Exception as e:
                return {"status": "error", "message": f"更新に失敗しました: {str(e)}"}

    def start_api(self, host: str = "0.0.0.0", port: int = 8008) -> threading.Thread:
        def _run():
            uvicorn.run(self.app, host=host, port=port, log_level="info")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

# ================
# エントリポイント
# ================
if __name__ == "__main__":
    if "--file" in sys.argv:
        params_file = sys.argv[sys.argv.index("--file") + 1]
    else:
        params_file = ".env"

    params = init_params(params_file)
    print("初期化パラメータ:")
    print(f"  control_port: {params['control_port']}")
    print(f"  api_port: {params['api_port']}")
    print(f"  selected_metrics: {params['selected_metrics'] or list(params['extra_info'].keys())}")
    print(f"  extra_info keys: {list(params['extra_info'].keys())}")

    lock = threading.Lock()
    # 1発言=1行の横持ち設計
    global_df = pd.DataFrame(columns=['user', 'text', 'metrics', 'title', 'score', 'conf', 'timestamp'])

    # 起動
    main_node = MainNode("0.0.0.0", params["control_port"], global_df, params, lock)
    socket_thread = threading.Thread(target=main_node.start_server, daemon=True)
    socket_thread.start()

    api_node = MyAPI(global_df, params, lock)
    api_thread = api_node.start_api(params["api_ip"], params["api_port"])

    # 監視出力
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Bye.")

