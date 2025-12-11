# my_node.py
import json
import socket
import threading


class Node:
    def __init__(self, host, port, peer_host, peer_port):
        self.host = host  # 自身のホスト
        self.port = port  # 自身のポート
        self.peer_host = peer_host  # 相手のホスト
        self.peer_port = peer_port  # 相手のポート

    def start_server(self):
        """サーバーを起動して接続を待機する"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        self.server_socket = server_socket
        print(f"サーバー起動中: {self.host}:{self.port} で待機中...")

        while True:
            client_socket, addr = server_socket.accept()
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        """クライアントからのデータを処理する"""
        #大きなデータ受信を考慮してrecvをループで回す
        data = b""
        while True:
            chunk = client_socket.recv(1024)
            if not chunk:
                break
            data += chunk
        client_socket.close()      

        # 反応
        self.process_data(data.decode("utf-8").strip())
        

    def process_data(self, data):
        """データを処理して応答を生成する（オーバーライド可能）"""
        pass

    def start_client(self, message):
        """クライアントとしてサーバーに接続してデータを送信する"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.peer_host, self.peer_port))
            # メッセージをjsonに変換
            message = json.dumps(message)
            client_socket.send(message.encode("utf-8"))
            client_socket.close()
        except Exception as e:
            print(f"クライアントエラー: {e}")

    def start(self):
        """サーバーとクライアントを並列に起動する"""
        # サーバーを別スレッドで起動
        threading.Thread(target=self.start_server, daemon=True).start()

        # ユーザー入力を使ってクライアント動作を開始
        while True:
            message = input("送信するメッセージを入力してください ('exit'で終了): ")
            if message.lower() == "exit":
                break
            self.start_client(message)

