# dash_server.py
# control.pyが持っているスコアを表示する
import requests
import pandas as pd
import os
import sys
import json
from dotenv import load_dotenv
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
from typing import List, Dict, Any
import numpy as np

# =====================
# 設定読込
# =====================
def init_params(file_path):
    load_dotenv(file_path)
    return {
        "dash_port": int(os.getenv("DASH_PORT", 8050)),
        "dash_ip": os.getenv("DASH_IP", "localhost"),
        "api_port": int(os.getenv("API_PORT", 8008)),
        "api_ip": os.getenv("API_IP", "localhost"),
        "extra_json_file": os.getenv("EXTRA_JSON_FILE", "extra.json"),
    }

# =====================
# 指標リスト読込
# =====================
def load_extra_metrics(extra_json_file) -> List[str]:
    try:
        with open(extra_json_file, 'r', encoding='utf-8') as f:
            extra_data = json.load(f)
            if isinstance(extra_data, dict):
                return list(extra_data.keys())
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"指標読み込みエラー: {e}")
    return ["威圧度", "逸脱度", "発言無効度", "偏り度"]

# =====================
# DataFrame 正規化
# =====================
def parse_metrics_column(df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    if df.empty:
        for m in metric_names:
            sc, cf = f"{m}スコア", f"{m}確信度"
            if sc not in df.columns:
                df[sc] = None
            if cf not in df.columns:
                df[cf] = None
        return df

    def to_metric_map(val) -> Dict[str, Dict[str, Any]]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return {}
        if isinstance(val, str):
            try:
                arr = json.loads(val)
            except Exception:
                return {}
        elif isinstance(val, list):
            arr = val
        else:
            return {}

        out = {}
        for item in arr:
            try:
                name = item.get("指標")
                if name:
                    out[name] = {"スコア": item.get("スコア"), "確信度": item.get("確信度")}
            except Exception:
                continue
        return out

    maps = df["metrics"].apply(to_metric_map) if "metrics" in df.columns else pd.Series([{}] * len(df))
    for m in metric_names:
        sc, cf = f"{m}スコア", f"{m}確信度"
        if sc not in df.columns:
            df[sc] = None
        if cf not in df.columns:
            df[cf] = None

        for idx in df.index:
            if pd.isna(df.at[idx, sc]) or df.at[idx, sc] is None:
                val = maps.iloc[idx].get(m, {})
                if "スコア" in val:
                    df.at[idx, sc] = val["スコア"]
                if "確信度" in val:
                    df.at[idx, cf] = val["確信度"]

    for m in metric_names:
        sc, cf = f"{m}スコア", f"{m}確信度"
        df[sc] = pd.to_numeric(df[sc], errors="coerce")
        df[cf] = pd.to_numeric(df[cf], errors="coerce")

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "conf" in df.columns:
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce")

    if "timestamp" in df.columns:
        try:
            df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception as e:
            print(f"タイムスタンプ変換エラー: {e}")
            df["_ts"] = pd.NaT
    else:
        df["_ts"] = pd.NaT

    return df

# =====================
# グラフ関連
# =====================
def zone_color_for(metric_name: str, score: float) -> str:
    if pd.isna(score):
        return "gray"
    if metric_name in ["威圧度", "逸脱度", "発言無効度", "偏り度"]:
        if score < 3:
            return "green"
        elif score < 7:
            return "orange"
        else:
            return "red"
    return "blue"

def add_zone_background(fig: go.Figure, metric_name: str):
    if metric_name in ["威圧度", "逸脱度", "発言無効度", "偏り度"]:
        fig.add_hrect(y0=0, y1=3, fillcolor="lightgreen", opacity=0.2,
                      annotation_text="通常ゾーン", annotation_position="top left")
        fig.add_hrect(y0=3, y1=7, fillcolor="lightyellow", opacity=0.2,
                      annotation_text="注意ゾーン", annotation_position="top left")
        fig.add_hrect(y0=7, y1=9, fillcolor="lightcoral", opacity=0.2,
                      annotation_text="警戒ゾーン", annotation_position="top left")

def build_metric_figure(df: pd.DataFrame, metric_name: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{metric_name}（データなし）", height=300)
        return fig

    if "_ts" in df.columns and not df["_ts"].isna().all():
        df_plot = df.copy().sort_values("_ts")
    else:
        df_plot = df.copy()
        df_plot["_ts"] = df_plot.index

    if metric_name == "総合":
        y = pd.to_numeric(df_plot["score"], errors="coerce")
    else:
        col = f"{metric_name}スコア"
        y = pd.to_numeric(df_plot[col] if col in df_plot.columns else pd.Series([None]*len(df_plot)), errors="coerce")

    mask = y.notna() & df_plot["_ts"].notna()
    if mask.any():
        x_values = df_plot.loc[mask, "_ts"]
        y_values = y.loc[mask]
        colors = [zone_color_for(metric_name, v) for v in y_values]

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name=metric_name,
            marker=dict(color=colors, size=8),
            line=dict(width=2),
            connectgaps=False
        ))
    else:
        fig.update_layout(title=f"{metric_name}（データなし）", height=300)

    add_zone_background(fig, metric_name)
    fig.update_layout(
        title=f"{metric_name}の推移",
        xaxis_title="時間",
        yaxis_title="スコア",
        xaxis=dict(type="date"),
        yaxis=dict(range=[0, 9]),
        margin=dict(l=30, r=20, t=50, b=30),
        height=300
    )
    return fig

# =====================
# Dash アプリ
# =====================
app = Dash(__name__)
app.title = "Meeting Dashboard"

app.layout = html.Div([
    # 評価開始状態を保存するストア
    dcc.Store(id="evaluation-started", data=False),
    dcc.Interval(id="tick", interval=5000, n_intervals=0),
    
    # 設定画面（未設定時のみ表示）
    html.Div(id="settings-page", children=[
        html.Div(style={
            "maxWidth": "600px",
            "margin": "80px auto",
            "padding": "40px",
            "backgroundColor": "white",
            "borderRadius": "10px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.1)"
        }, children=[
            html.H2("🎯 Meeting Keeper", style={"textAlign": "center", "marginBottom": "10px", "color": "#333"}),
            html.P("会議を開始する前に、以下の設定を行ってください。", 
                   style={"textAlign": "center", "marginBottom": "30px", "color": "#666"}),
            
            # 会議の目的
            html.Div([
                html.Label("会議の目的:", 
                          style={"fontWeight": "bold", "marginBottom": "8px", "display": "block", "fontSize": "16px"}),
                dcc.Textarea(
                    id="initial-meeting-goal-input",
                    placeholder="例：新製品のアイデアを出し合い、実現可能性の高い案を3つに絞り込む",
                    style={
                        "width": "100%", 
                        "marginBottom": "20px", 
                        "padding": "12px",
                        "border": "2px solid #e0e0e0", 
                        "borderRadius": "6px", 
                        "height": "100px",
                        "resize": "vertical",
                        "fontSize": "14px",
                        "boxSizing": "border-box"
                    }
                )
            ]),
            
            # 会議の形式（チェックボックス）
            html.Div([
                html.Label("会議の形式（複数選択可）:", 
                          style={"fontWeight": "bold", "marginBottom": "10px", "display": "block", "fontSize": "16px"}),
                dcc.Checklist(
                    id="initial-meeting-type-input",
                    options=[
                        {"label": "発散（ブレインストーミング）", "value": "発散"},
                        {"label": "収束（アイデアの絞り込み）", "value": "収束"},
                        {"label": "アイスブレイク（関係構築）", "value": "アイスブレイク"},
                        {"label": "意思決定", "value": "意思決定"},
                        {"label": "振り返り（評価・フィードバック）", "value": "振り返り"},
                        {"label": "情報共有", "value": "情報共有"},
                        {"label": "問題解決", "value": "問題解決"},
                        {"label": "合意形成", "value": "合意形成"},
                        {"label": "定例会議", "value": "定例会議"},
                        {"label": "臨時会議", "value": "臨時会議"},
                    ],
                    value=[],
                    style={"marginBottom": "30px"},
                    labelStyle={
                        "display": "block", 
                        "marginBottom": "10px", 
                        "cursor": "pointer",
                        "padding": "8px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "4px",
                        "transition": "background-color 0.2s"
                    },
                    inputStyle={"marginRight": "10px"}
                )
            ]),
            
            # スタートボタン
            html.Div(style={"textAlign": "center"}, children=[
                html.Button("スタート", id="start-meeting-button", n_clicks=0,
                           style={
                               "padding": "15px 60px", 
                               "backgroundColor": "#28a745",
                               "color": "white", 
                               "border": "none", 
                               "borderRadius": "6px",
                               "fontSize": "18px", 
                               "fontWeight": "bold", 
                               "cursor": "pointer",
                               "boxShadow": "0 2px 8px rgba(40, 167, 69, 0.3)",
                               "transition": "all 0.3s"
                           })
            ])
        ])
    ], style={"display": "block", "backgroundColor": "#f5f5f5", "minHeight": "100vh"}),
    
    # ダッシュボード（設定完了後に表示）
    html.Div(id="dashboard-page", children=[
        html.H2("Meeting Keeper - Dashboard"),
        html.Div(id="params-box", style={"marginBottom": "12px", "fontSize": "14px"}),

        # 現在の設定と設定変更ボタン
        html.Div([
            html.H4("現在の設定"),
            html.Div(id="current-settings",
                     style={"marginBottom": "20px", "padding": "10px",
                            "backgroundColor": "#f8f9fa", "border": "1px solid #dee2e6",
                            "borderRadius": "5px"}),
            html.Div([
                html.Button("設定を変更", id="open-settings-button", n_clicks=0,
                            style={"padding": "8px 16px", "backgroundColor": "#28a745", 
                                   "color": "white", "border": "none", "borderRadius": "4px",
                                   "marginRight": "10px"}),
            ])
        ]),

        # 設定変更モーダル
        html.Div(id="settings-modal", children=[
            # モーダルの背景オーバーレイ
            html.Div(style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(0,0,0,0.5)",
                "zIndex": 1000
            }),
            # モーダルコンテンツ
            html.Div(style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "backgroundColor": "white",
                "padding": "30px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
                "zIndex": 1001,
                "minWidth": "500px",
                "maxWidth": "80vw"
            }, children=[
                html.H3("会議設定の変更", style={"marginTop": 0, "marginBottom": "20px", "color": "#333"}),
                
                # 会議の形式（チェックボックス）
                html.Div([
                    html.Label("会議の形式（複数選択可）:", 
                              style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                    dcc.Checklist(
                        id="modal-meeting-type-input",
                        options=[
                            {"label": "発散（ブレインストーミング）", "value": "発散"},
                            {"label": "収束（アイデアの絞り込み）", "value": "収束"},
                            {"label": "アイスブレイク（関係構築）", "value": "アイスブレイク"},
                            {"label": "意思決定", "value": "意思決定"},
                            {"label": "振り返り（評価・フィードバック）", "value": "振り返り"},
                            {"label": "情報共有", "value": "情報共有"},
                            {"label": "問題解決", "value": "問題解決"},
                            {"label": "合意形成", "value": "合意形成"},
                        ],
                        value=[],
                        style={"marginBottom": "20px"},
                        labelStyle={"display": "block", "marginBottom": "8px", "cursor": "pointer"},
                        inputStyle={"marginRight": "8px"}
                    )
                ]),
                
                # 会議の目的
                html.Div([
                    html.Label("会議の目的:", 
                              style={"fontWeight": "bold", "marginBottom": "5px", "display": "block"}),
                    dcc.Textarea(
                        id="modal-meeting-goal-input",
                        placeholder="例：新製品のアイデアを出し合い、実現可能性の高い案を3つに絞り込む",
                        style={
                            "width": "100%", 
                            "marginBottom": "20px", 
                            "padding": "10px",
                            "border": "1px solid #ccc", 
                            "borderRadius": "4px", 
                            "height": "100px",
                            "resize": "vertical",
                            "fontSize": "14px"
                        }
                    )
                ]),
                
                # ボタン
                html.Div(id="settings-buttons-container", style={"textAlign": "right"}, children=[
                    html.Button("キャンセル", id="cancel-settings-button", n_clicks=0,
                               style={"marginRight": "10px", "padding": "10px 20px",
                                      "backgroundColor": "#6c757d", "color": "white",
                                      "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                    html.Button("保存", id="save-settings-button", n_clicks=0,
                               style={"padding": "10px 30px", "backgroundColor": "#28a745",
                                      "color": "white", "border": "none", "borderRadius": "4px",
                                      "fontSize": "16px", "fontWeight": "bold", "cursor": "pointer"})
                ])
            ])
        ], style={"display": "none"}),

        # 指標選択
        html.Div([
            html.Label("表示する指標を選択してください:"),
            dcc.Dropdown(id="metric-dropdown", multi=True,
                         placeholder="指標を選択してください",
                         style={"marginBottom": "20px"})
        ]),

        dash_table.DataTable(
            id="table",
            columns=[
                {"name": "timestamp", "id": "timestamp"},
                {"name": "user", "id": "user"},
                {"name": "title", "id": "title"},
                {"name": "score(総合)", "id": "score"},
                {"name": "conf(総合)", "id": "conf"},
                {"name": "text", "id": "text"},
                {"name": "metrics(JSON)", "id": "metrics"},
            ],
            page_size=20,
            style_table={"height": "60vh", "overflowY": "auto"},
            style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
        ),

        html.Div(id="graphs-container"),
    ], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "12px", "display": "none"}),
], style={"fontFamily": "Arial, sans-serif"})

# ============ コールバック群 ============

# ページの表示切り替え（初期設定ページ vs ダッシュボード）
@app.callback(
    Output("settings-page", "style"),
    Output("dashboard-page", "style"),
    Input("tick", "n_intervals"),
    Input("start-meeting-button", "n_clicks"),
    prevent_initial_call=False
)
def toggle_pages(n_intervals, start_clicks):
    """初期設定ページとダッシュボードの切り替え"""
    ctx = callback_context
    
    # スタートボタンが押された後はダッシュボードを表示
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "start-meeting-button.n_clicks":
        if start_clicks and start_clicks > 0:
            return (
                {"display": "none"},  # 設定ページを非表示
                {"maxWidth": "1180px", "margin": "0 auto", "padding": "12px", "display": "block"}  # ダッシュボードを表示
            )
    
    # 初期ロード時：設定をチェック
    try:
        r = requests.get(f"{API_BASE}/params", timeout=2)
        if r.ok:
            p = r.json().get("params", {})
            meeting_type = p.get("meeting_type", "")
            meeting_goal = p.get("meeting_goal", "")
            
            # 設定が存在する場合はダッシュボードを表示
            if meeting_type and meeting_goal:
                return (
                    {"display": "none"},
                    {"maxWidth": "1180px", "margin": "0 auto", "padding": "12px", "display": "block"}
                )
    except Exception:
        pass
    
    # デフォルト：設定ページを表示
    return (
        {"display": "block", "backgroundColor": "#f5f5f5", "minHeight": "100vh"},
        {"display": "none"}
    )

# スタートボタンの処理
@app.callback(
    Output("start-meeting-button", "children"),
    Input("start-meeting-button", "n_clicks"),
    State("initial-meeting-type-input", "value"),
    State("initial-meeting-goal-input", "value"),
    prevent_initial_call=True
)
def start_meeting(n_clicks, type_value, goal_value):
    """初期設定でスタートボタンが押されたときの処理"""
    if n_clicks > 0:
        try:
            # チェックボックスの値（リスト）をカンマ区切り文字列に変換
            if isinstance(type_value, list):
                meeting_type_str = ", ".join(type_value)
            else:
                meeting_type_str = type_value or ""
            
            response = requests.post(
                f"{API_BASE}/update_settings",
                json={"meeting_type": meeting_type_str, "meeting_goal": goal_value or ""},
                timeout=2
            )
            if response.ok:
                return "✓ 設定完了"
            else:
                return "✗ エラー"
        except Exception as e:
            return f"✗ エラー"
    return "スタート"

@app.callback(
    Output("metric-dropdown", "options"),
    Input("tick", "n_intervals"),
    prevent_initial_call=False
)
def update_metric_options(_):
    try:
        metrics = load_extra_metrics(os.getenv("EXTRA_JSON_FILE", "extra.json"))
        metrics = ["総合"] + metrics
        return [{"label": m, "value": m} for m in metrics]
    except Exception:
        return [{"label": m, "value": m} for m in ["総合", "威圧度", "逸脱度", "発言無効度", "偏り度"]]

@app.callback(
    Output("modal-meeting-type-input", "value"),
    Output("modal-meeting-goal-input", "value"),
    Input("open-settings-button", "n_clicks"),
    prevent_initial_call=True
)
def load_meeting_settings_for_modal(n_clicks):
    """モーダルを開くときに現在の会議の設定を読み込み"""
    if n_clicks > 0:
        try:
            r = requests.get(f"{API_BASE}/params", timeout=2)
            if r.ok:
                p = r.json().get("params", {})
                meeting_type = p.get("meeting_type", "")
                meeting_goal = p.get("meeting_goal", "")
                
                # meeting_typeがカンマ区切りの場合、リストに分割
                if isinstance(meeting_type, str) and meeting_type:
                    type_list = [t.strip() for t in meeting_type.split(",")]
                elif isinstance(meeting_type, list):
                    type_list = meeting_type
                else:
                    type_list = []
                
                return type_list, meeting_goal
        except Exception:
            pass
    return [], ""


@app.callback(
    Output("current-settings", "children"),
    Input("tick", "n_intervals"),
    prevent_initial_call=False
)
def display_current_settings(_):
    """現在の設定を表示"""
    try:
        r = requests.get(f"{API_BASE}/params", timeout=2)
        if r.ok:
            p = r.json().get("params", {})
            meeting_type = p.get("meeting_type", "未設定")
            meeting_goal = p.get("meeting_goal", "未設定")
            return [
                html.P(f"会議の形式: {meeting_type}", style={"margin": "5px 0", "fontWeight": "bold"}),
                html.P(f"会議の目的: {meeting_goal}", style={"margin": "5px 0", "fontWeight": "bold"})
            ]
    except Exception as e:
        return html.P(f"設定の読み込みに失敗しました: {str(e)}", style={"color": "red"})
    return html.P("設定を読み込み中...", style={"color": "gray"})


# モーダルの表示/非表示を制御（初期表示を含む）
@app.callback(
    Output("settings-modal", "style"),
    Output("cancel-settings-button", "style"),
    Input("tick", "n_intervals"),
    Input("open-settings-button", "n_clicks"),
    Input("cancel-settings-button", "n_clicks"),
    Input("save-settings-button", "n_clicks"),
    prevent_initial_call=False
)
def toggle_modal(n_intervals, open_clicks, cancel_clicks, save_clicks):
    """モーダルの表示/非表示を制御"""
    ctx = callback_context
    
    # 初期ロード時：設定が未設定の場合はモーダルを表示
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == "tick.n_intervals":
        try:
            r = requests.get(f"{API_BASE}/params", timeout=2)
            if r.ok:
                p = r.json().get("params", {})
                meeting_type = p.get("meeting_type", "")
                meeting_goal = p.get("meeting_goal", "")
                
                # 設定が未設定の場合、モーダルを表示（キャンセルボタンは非表示）
                if not meeting_type or not meeting_goal:
                    cancel_style = {"marginRight": "10px", "padding": "10px 20px",
                                   "backgroundColor": "#6c757d", "color": "white",
                                   "border": "none", "borderRadius": "4px", "cursor": "pointer",
                                   "display": "none"}  # 初期設定時はキャンセル不可
                    return {"display": "block"}, cancel_style
        except Exception:
            pass
        return {"display": "none"}, {"marginRight": "10px", "padding": "10px 20px",
                                      "backgroundColor": "#6c757d", "color": "white",
                                      "border": "none", "borderRadius": "4px", "cursor": "pointer"}
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    cancel_style = {"marginRight": "10px", "padding": "10px 20px",
                   "backgroundColor": "#6c757d", "color": "white",
                   "border": "none", "borderRadius": "4px", "cursor": "pointer"}
    
    if trigger_id == "open-settings-button":
        return {"display": "block"}, cancel_style
    elif trigger_id in ["cancel-settings-button", "save-settings-button"]:
        return {"display": "none"}, cancel_style
    
    return {"display": "none"}, cancel_style


@app.callback(
    Output("save-settings-button", "children"),
    Output("current-settings", "children", allow_duplicate=True),
    Input("save-settings-button", "n_clicks"),
    State("modal-meeting-type-input", "value"),
    State("modal-meeting-goal-input", "value"),
    prevent_initial_call=True
)
def save_meeting_settings(n_clicks, type_value, goal_value):
    """会議の設定を保存"""
    if n_clicks > 0:
        try:
            # チェックボックスの値（リスト）をカンマ区切り文字列に変換
            if isinstance(type_value, list):
                meeting_type_str = ", ".join(type_value)
            else:
                meeting_type_str = type_value or ""
            
            response = requests.post(
                f"{API_BASE}/update_settings",
                json={"meeting_type": meeting_type_str, "meeting_goal": goal_value or ""},
                timeout=2
            )
            if response.ok:
                # 設定保存後、現在の設定を即座に更新
                try:
                    r = requests.get(f"{API_BASE}/params", timeout=2)
                    if r.ok:
                        p = r.json().get("params", {})
                        meeting_type = p.get("meeting_type", "未設定")
                        meeting_goal = p.get("meeting_goal", "未設定")
                        updated_settings = [
                            html.P(f"会議の形式: {meeting_type}", style={"margin": "5px 0", "fontWeight": "bold"}),
                            html.P(f"会議の目的: {meeting_goal}", style={"margin": "5px 0", "fontWeight": "bold"})
                        ]
                        return "✓ 設定完了", updated_settings
                except Exception:
                    pass
                return "✓ 設定完了", no_update
            else:
                return "✗ 保存失敗", no_update
        except Exception as e:
            return f"✗ エラー: {str(e)}", no_update
    return "スタート", no_update

@app.callback(
    Output("table", "data"),
    Output("table", "columns"),
    Output("graphs-container", "children"),
    Input("tick", "n_intervals"),
    Input("metric-dropdown", "value"),
    prevent_initial_call=False
)
def tick(_, selected_metrics):
    try:
        r = requests.get(f"{API_BASE}/data", timeout=2)
        r.raise_for_status()
        data = r.json()
        rows = data.get("rows", [])
    except Exception as e:
        print(f"データ取得エラー: {e}")
        rows = []

    df = pd.DataFrame(rows)

    for base_col in ["user", "text", "title", "score", "conf", "timestamp", "metrics"]:
        if base_col not in df.columns:
            df[base_col] = None

    extra_metrics = load_extra_metrics(os.getenv("EXTRA_JSON_FILE", "extra.json"))
    df = parse_metrics_column(df, extra_metrics)
    
    if "metrics" in df.columns:
        df["metrics"] = df["metrics"].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        )

    if "_ts" in df.columns:
        df = df.sort_values("_ts", ascending=False)

    table_columns = [
        {"name": "timestamp", "id": "timestamp"},
        {"name": "user", "id": "user"},
        {"name": "title", "id": "title"},
        {"name": "score(総合)", "id": "score"},
        {"name": "conf(総合)", "id": "conf"},
        {"name": "text", "id": "text"},
        {"name": "metrics(JSON)", "id": "metrics"},
    ]
    for m in extra_metrics:
        sc, cf = f"{m}スコア", f"{m}確信度"
        if sc in df.columns:
            table_columns.append({"name": sc, "id": sc})
        if cf in df.columns:
            table_columns.append({"name": cf, "id": cf})

    table_df = df[[c["id"] for c in table_columns if c["id"] in df.columns]].replace({np.nan: None})
    table_data = table_df.to_dict(orient="records")

    graphs = []
    display_metrics = selected_metrics if (selected_metrics and len(selected_metrics) > 0) else ["総合"] + extra_metrics

    for metric_name in display_metrics:
        try:
            if df.empty:
                fig = go.Figure()
                fig.update_layout(title=f"{metric_name}（データなし）", height=300)
                graphs.append(dcc.Graph(figure=fig, id=f"graph-{metric_name}"))
                continue

            if metric_name == "総合":
                fig = build_metric_figure(df, "総合")
            else:
                fig = build_metric_figure(df, metric_name)
            
            graphs.append(dcc.Graph(figure=fig, id=f"graph-{metric_name}"))
        except Exception as e:
            print(f"グラフ生成エラー ({metric_name}): {e}")
            fig = go.Figure()
            fig.update_layout(title=f"{metric_name}（エラー）", height=300)
            graphs.append(dcc.Graph(figure=fig, id=f"graph-{metric_name}"))

    return table_data, table_columns, graphs

# =====================
# エントリポイント
# =====================
if __name__ == "__main__":
    if "--file" in sys.argv:
        params_file = sys.argv[sys.argv.index("--file") + 1]
    else:
        params_file = ".env"
    params = init_params(params_file)

    global API_BASE
    API_BASE = f"http://{params['api_ip']}:{params['api_port']}"
    print(f"API_BASE: {API_BASE}")
    
    app.run(host=params["dash_ip"], port=params["dash_port"], debug=False)

