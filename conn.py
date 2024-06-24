import numpy as np
import msgpack
import socket
import threading

class ConnManager():
    IP = "127.0.0.1"
    PORT = 5000

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.IP, self.PORT))
        self.socket.listen()
        conn_thread = threading.Thread(target=self.receive_messages)
        conn_thread.daemon = True
        conn_thread.start()
    
    def connect_managers(self, data_manager, ui_manager):
        self.data_manager = data_manager
        self.ui_manager = ui_manager
    
    def create_workspace(self, workspace_id: int) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "create_workspace",
                "workspace_id": workspace_id,
            }))
        except Exception as e:
            print(e)

    def delete_workspace(self, workspace_id: int) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "delete_workspace",
                "workspace_id": workspace_id,
            }))
        except Exception as e:
            print(e)

    def update_points(self, workspace_id: int, points: np.array) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "update_points",
                "workspace_id": workspace_id,
                "points": points.tolist(),
            }))
        except Exception as e:
            print(e)

    def update_labels(self, workspace_id: int, labels: list[int]) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "update_labels",
                "workspace_id": workspace_id,
                "labels": labels,
            }))
        except Exception as e:
            print(e)
    
    def remove_point_selection(self, workspace_id: int, index: int) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "remove_point_selection",
                "workspace_id": workspace_id,
                "index": index,
            }))
        except Exception as e:
            print(e)
    
    def receive_messages(self) -> None:
        self.conn, _ = self.socket.accept()
        print("Navegador conectado")
        unpacker = msgpack.Unpacker()
        while True:
            data = self.conn.recv(512)
            if not data:
                break
            unpacker.feed(data)
            for msg in unpacker:
                if msg["type"] == "request_image":
                    self.conn.sendall(msgpack.packb({
                        "type": "response_image",
                        "title": msg["workspace_id"]+"_"+msg["index"],
                        "image": self.data_manager.request_image(msg["index"]),
                        "coords": self.ui_manager.request_coords(msg["workspace_id"], msg["index"])
                    }))
                    
                elif msg["type"] == "set_selection":
                    self.ui_manager.set_selection(msg["workspace_id"], msg["indexes"])

                elif msg["type"] == "clear_selection":
                    self.ui_manager.clear_selection(msg["workspace_id"])