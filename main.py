import ui, data, conn

def main():
    data_manager = data.DataManager()
    conn_manager = conn.ConnManager()
    ui_manager = ui.UIManager()
    data_manager.connect_managers(conn_manager, ui_manager)
    conn_manager.connect_managers(data_manager, ui_manager)
    ui_manager.connect_managers(conn_manager, data_manager)
    ui_manager.run()

if __name__ == "__main__":
    main()