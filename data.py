import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import HDBSCAN, KMeans


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
    
    def apply_tsne(self, workspace_id: int, learning_rate: float, perplexity: float, early_exaggeration: float, metric: str) -> tuple[list[float], list[float]]:
        return np.random.rand(100).tolist(), np.random.rand(100).tolist()
    
    def apply_pca(self, workspace_id: int, whiten: bool, tolerance: float, svd_solver: str) -> tuple[list[float], list[float]]:
        return np.random.rand(100).tolist(), np.random.rand(100).tolist()
    
    ### CLUSTERING ###
    # Aplica <METODO> a los componentes X y Y, envia los labels al navegador
    # y devuelve los mismos labels

    def apply_hdbscan(self, workspace_id: int, xdata: list[float], ydata: list[float], min_cluster_size: int, min_samples: int, cluster_selection_epsilon: float, cluster_selection_method: str) -> list[int]:
        return []

    def apply_kmeans(self, workspace_id: int, xdata: list[float], ydata: list[float], n_clusters: int, max_iter: int, init: str, algorithm: str) -> list[int]:
        return []