import numpy as np

import os
import io
import msgpack
from datetime import datetime
from PIL import Image
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import HDBSCAN, KMeans, OPTICS, SpectralClustering

class DataManager():
    MODEL_PATH = "efficientnet-lite4-11.onnx"

    def __init__(self):
        self.workspace_data = {}
        self.workspace_labels = {}
        self.folder_path = ""
        self.dataset = {}
    
    def init_dataset(self, files):
        self.dataset = {
            "length": len(files),
            "filenames": files,
            "vectors": None,
        }

    def connect_managers(self, conn_manager, ui_manager) -> None:
        self.conn_manager = conn_manager
        self.ui_manager = ui_manager
        # Por ahora hardcodeado
        self.load_folder("Top20")
    
    def load_folder(self, folder_path: str) -> None:
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise Exception("data_manager: folder not found")

        # Check if dataset file exists
        dataset_path = os.path.join(folder_path, "dataset.mpack")
        if not os.path.exists(dataset_path):
            print("Generando dataset")
            self.create_dataset(folder_path)
        else:
            with open(dataset_path, "rb") as file:
                self.dataset = msgpack.load(file)
        
        self.folder_path = folder_path
        self.ui_manager.create_texture_registry(self.dataset["length"])

    def get_files_from(self, folder_path, extensions):
        files = []
        for file in os.listdir(folder_path):
            for ext in extensions:
                if file.lower().endswith("."+ext):
                    files.append(file)
        return files

    def create_dataset(self, folder_path: str) -> None:
        session = ort.InferenceSession(self.MODEL_PATH)
        input_name = session.get_inputs()[0].name

        extensions = ["bmp", "png", "jpeg", "jpg"]
        files = self.get_files_from(folder_path, extensions)
        self.init_dataset(files)

        vectors = []
        for i, filename in enumerate(files):
            print("processing:", os.path.join(folder_path, filename))
            image = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            width, height = image.size
            smallest_side = min(width, height)
            left = (width - smallest_side) / 2
            top = (height - smallest_side) / 2
            right = (width + smallest_side) / 2
            bottom = (height + smallest_side) / 2

            image = image.crop((left, top, right, bottom))
            image = image.resize((224, 224), Image.NEAREST)

            input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        
            for x in range(224):
                for y in range(224):
                    pixel = image.getpixel((x, y))
                    input_tensor[0, x, y, 0] = (pixel[0] - 127.0) / 128.0  # Red channel
                    input_tensor[0, x, y, 1] = (pixel[1] - 127.0) / 128.0  # Green channel
                    input_tensor[0, x, y, 2] = (pixel[2] - 127.0) / 128.0  # Blue channel

            output = session.run(None, {input_name: input_tensor})
            vectors.append(output[0].flatten())
        
        self.dataset["vectors"] = np.array(vectors)
        with open(os.path.join(folder_path, "dataset.mpack"), "wb") as file:
            msgpack.dump(self.dataset, file)

    def get_path(self, index: int) -> str:
        return os.path.join(self.folder_path, self.dataset["filenames"][index])

    def get_filename(self, index: int) -> str:
        return self.dataset["filenames"][index]
    
    def get_image(self, index: int) -> bytearray:
        with Image.open(self.get_path(index)) as im:
            im.thumbnail((256, 256))
            data = io.BytesIO()
            im.save(data, format="JPEG")
            return data.getvalue()
    
    def get_coords(self, workspace_id: int, index: int) -> np.ndarray:
        return self.workspace_data[workspace_id][index]
        
    ### REDUCCION DE DIM ###
    # Aplica <METODO> al dataset, envia los puntos 3D al navegador
    # y devuelve los puntos 2D en dos listas con el componente X y Y respectivamente

    def apply_umap(self, workspace_id: int, n_neighbors: int, min_dist: float, metric: str) -> tuple[list[float], list[float]]:
        umap = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        self.workspace_data[workspace_id] = umap.fit_transform(self.dataset["vectors"])
        self.conn_manager.update_points(workspace_id, self.workspace_data[workspace_id])
        # Generando puntos 2D para la UI
        data2d = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(self.dataset["vectors"])
        return data2d[:, 0].tolist(), data2d[:, 1].tolist()
    
    def apply_tsne(self, workspace_id: int, learning_rate: float, perplexity: float, early_exaggeration: float, metric: str) -> tuple[list[float], list[float]]:
        tsne = TSNE(n_components=3, learning_rate=learning_rate, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric)
        self.workspace_data[workspace_id] = tsne.fit_transform(self.dataset["vectors"])
        self.conn_manager.update_points(workspace_id, self.workspace_data[workspace_id])
        # Generando puntos 2D para la UI
        data2d = TSNE(n_components=2, learning_rate=learning_rate, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric).fit_transform(self.dataset["vectors"])
        return data2d[:, 0].tolist(), data2d[:, 1].tolist()
    
    def apply_pca(self, workspace_id: int, whiten: bool, tolerance: float, svd_solver: str) -> tuple[list[float], list[float]]:
        pca = PCA(n_components=3, whiten=not whiten, tol=tolerance, svd_solver=svd_solver)
        self.workspace_data[workspace_id] = pca.fit_transform(self.dataset["vectors"])
        self.conn_manager.update_points(workspace_id, self.workspace_data[workspace_id]) 
        # Generando puntos 2D para la UI
        data2d = PCA(n_components=2, whiten=not whiten, tol=tolerance, svd_solver=svd_solver).fit_transform(self.dataset["vectors"])
        return data2d[:, 0].tolist(), data2d[:, 1].tolist()
    
    ### CLUSTERING ###
    # Aplica <METODO> a los componentes X y Y, envia los labels al navegador
    # y devuelve los mismos labels

    def apply_hdbscan(self, workspace_id: int, min_cluster_size: int, min_samples: int, cluster_selection_epsilon: float, cluster_selection_method: str) -> list[int]:
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=cluster_selection_method)
        hdbscan.fit_predict(self.workspace_data[workspace_id])
        self.conn_manager.update_labels(workspace_id, hdbscan.labels_)
        return hdbscan.labels_

    def apply_kmeans(self, workspace_id: int, n_clusters: int, max_iter: int, init: str, algorithm: str) -> list[int]:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, algorithm=algorithm)
        kmeans.fit_predict(self.workspace_data[workspace_id])
        self.conn_manager.update_labels(workspace_id, kmeans.labels_)
        return kmeans.labels_
    
    def apply_optics(self, workspace_id: int, min_samples: int, max_eps: float, metric: str, cluster_method: str) -> list[int]:
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, cluster_method=cluster_method)
        optics.fit_predict(self.workspace_data[workspace_id])
        self.conn_manager.update_labels(workspace_id, optics.labels_)
        return optics.labels_
    
    def apply_spectral(self, workspace_id: int, n_clusters: int, eigen_solver: str, affinity: str, assign_labels: str) -> list[int]:
        spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, affinity=affinity, assign_labels=assign_labels)
        spectral.fit_predict(self.workspace_data[workspace_id])
        self.conn_manager.update_labels(workspace_id, spectral.labels_)
        return spectral.labels_