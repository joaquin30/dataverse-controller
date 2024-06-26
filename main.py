import ui, data, conn
import msgpack
import msgpack_numpy as msg_np

def main():
    msg_np.patch()
    data_manager = data.DataManager()
    conn_manager = conn.ConnManager()
    ui_manager = ui.UIManager()
    ui_manager.connect_managers(conn_manager, data_manager)
    data_manager.connect_managers(conn_manager, ui_manager)
    conn_manager.connect_managers(data_manager, ui_manager)
    ui_manager.run()

if __name__ == "__main__":
    main()