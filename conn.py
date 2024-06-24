import numpy as np

class ConnManager():
    def __init__(self, port):
        pass
    
    def connect_managers(self, data_manager, ui_manager):
        self.data_manager = data_manager
        self.ui_manager = ui_manager
    
    def create_workspace(self, new_workspace_id: int) -> None:
        pass

    def delete_workspace(self, workspace_id: int) -> None:
        pass

    def update_points(self, workspace_id: int, points: np.array) -> None:
        pass

    def update_labels(self, workspace_id: int, labels: list[int]) -> None:
        pass
    
    def remove_point_selection(self, workspace_id: int, index: int) -> None:
        pass