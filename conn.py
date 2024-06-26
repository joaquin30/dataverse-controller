import numpy as np
import msgpack
import socket
import threading
import utils

class ConnManager():
    IP = "192.168.103.100"
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
            print("create_workspace:", e)

    def delete_workspace(self, workspace_id: int) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "delete_workspace",
                "workspace_id": workspace_id,
            }))
        except Exception as e:
            print("delete_workspace:", e)

    def update_points(self, workspace_id: int, points: np.ndarray) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "update_points",
                "workspace_id": workspace_id,
                "points": points.tolist(),
            }))
        except Exception as e:
            print("update_points:", e)

    def update_labels(self, workspace_id: int, labels: list[int]) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "update_labels",
                "workspace_id": workspace_id,
                "labels": labels.tolist(),
                "colors": utils.get_normalized_colors(max(labels) + 1),
            }))
        except Exception as e:
            print("update_labels:", e)
    
    def remove_point_selection(self, workspace_id: int, index: int) -> None:
        try:
            self.conn.sendall(msgpack.packb({
                "type": "remove_point_selection",
                "workspace_id": workspace_id,
                "index": index,
            }))
        except Exception as e:
            print("remove_point_selection:", e)
    
    def disconnect(self) -> None:
        try:
            self.conn.close()
        except Exception as e:
            print("disconnect:", e)

    def receive_messages(self) -> None:
        while True:
            self.conn, _ = self.socket.accept()
            self.ui_manager.set_navigator_status(True)
            print("navegador conectado")
            unpacker = msgpack.Unpacker()
            while True:
                data = self.conn.recv(512)
                if not data:
                    break
                unpacker.feed(data)
                for msg in unpacker:
                    if msg["type"] == "request_image":
                        try:
                            self.conn.sendall(msgpack.packb({
                                "type": "response_image",
                                "title": self.data_manager.get_filename(msg["index"]),
                                "image": self.data_manager.get_image(msg["index"]),
                                "coords": self.data_manager.get_coords(msg["workspace_id"], msg["index"]).tolist(),
                            }))
                        except Exception as e:
                            print("response_image:", e)
                        
                    elif msg["type"] == "set_selection":
                        self.ui_manager.set_selection(msg["workspace_id"], msg["indexes"])

                    elif msg["type"] == "clear_selection":
                        self.ui_manager.clear_selection(msg["workspace_id"])
            self.ui_manager.set_navigator_status(False)