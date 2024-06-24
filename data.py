import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import HDBSCAN, KMeans


class DataManager():
    def __init__(self):
        self.workspace_data = {}
        self.workspace_labels = {}
    
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
        umap = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        self.workspace_data[workspace_id] = umap.fit_transform(np.random.rand(100, 1000))
        return self.workspace_data[workspace_id][:, 0].tolist(), self.workspace_data[workspace_id][:, 1].tolist()
    
    def apply_tsne(self, workspace_id: int, learning_rate: float, perplexity: float, early_exaggeration: float, metric: str) -> tuple[list[float], list[float]]:
        tsne = TSNE(n_components=3, learning_rate=learning_rate, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric)
        self.workspace_data[workspace_id] = tsne.fit_transform(np.random.rand(100, 1000))
        return self.workspace_data[workspace_id][:, 0].tolist(), self.workspace_data[workspace_id][:, 1].tolist()
    
    def apply_pca(self, workspace_id: int, whiten: bool, tolerance: float, svd_solver: str) -> tuple[list[float], list[float]]:
        pca = PCA(n_components=3, whiten=whiten, tolerance=tolerance, svd_solver=svd_solver)
        self.workspace_data[workspace_id] = pca.fit_transform(np.random.rand(100, 1000)) 
        return self.workspace_data[workspace_id][:, 0].tolist(), self.workspace_data[workspace_id][:, 1].tolist()
    # TODO: Conectar datos de imagenes preprocesadas con los apply_xxxx()
    ### CLUSTERING ###
    # Aplica <METODO> a los componentes X y Y, envia los labels al navegador
    # y devuelve los mismos labels

    def apply_hdbscan(self, workspace_id: int, xdata: list[float], ydata: list[float], min_cluster_size: int, min_samples: int, cluster_selection_epsilon: float, cluster_selection_method: str) -> list[int]:
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=cluster_selection_method)
        self.workspace_data[workspace_id] = hdbscan.fit_predict(self.workspace_data[workspace_id])
        return self.workspace_data[workspace_id].tolist()

    def apply_kmeans(self, workspace_id: int, xdata: list[float], ydata: list[float], n_clusters: int, max_iter: int, init: str, algorithm: str) -> list[int]:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, algorithm=algorithm)
        self.workspace_labels[workspace_id] = kmeans.fit_predict(self.workspace_data[workspace_id])
        return self.workspace_labels[workspace_id].tolist()