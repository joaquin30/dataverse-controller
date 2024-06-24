import numpy as np

class DataManager():
    def __init__(self):
        pass
    
    def connect_managers(self, conn_manager, ui_manager) -> None:
        self.conn_manager = conn_manager
        self.ui_manager = ui_manager
    
    def load_folder(self, folder_path: str) -> None:
        pass

    def create_dataset(self, folder_path: str) -> None:
        pass

    ### REDUCCION DE DIM ###
    # Aplica <METODO> al dataset, envia los puntos 3D al navegador
    # y devuelve los puntos 2D en dos listas con el componente X y Y respectivamente

    def apply_umap(self, workspace_id: int, n_neighbors: int, min_dist: float, metric: str) -> tuple[list[float], list[float]]:
        return np.random.rand(100).tolist(), np.random.rand(100).tolist()
    
    # TODO otros metodos
    
    ### CLUSTERING ###
    # Aplica <METODO> a los componentes X y Y, envia los labels al navegador
    # y devuelve los mismos labels

    def apply_hdbscan(self, workspace_id: int, xdata: list[float], ydata: list[float], min_cluster_size: int, min_samples: int, cluster_selection_epsilon: float, cluster_selection_method: str) -> list[int]:
        return []

    # TODO otros metodos