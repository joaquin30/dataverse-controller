import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, folder_path, length):
        self.FolderPath = folder_path
        self.Length = length
        self.Filenames = [None] * length
        self.Vectors = [None] * length

class DataManager():
    def __init__(self):
        pass
    
    def connect_managers(self, conn_manager, ui_manager) -> None:
        self.conn_manager = conn_manager
        self.ui_manager = ui_manager
    
    def load_folder(self, folder_path: str) -> None:
        # Check if folder exists
        if not os.path.exists(folder_path):
            print("Error: Folder not found")
            exit()

        # Check if dataset file exists
        dataset_path = os.path.join(folder_path, "dataset.json")
        if not os.path.exists(dataset_path):
            print("Generating dataset")
            self.create_dataset(folder_path)
        
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        print(f"Dataset loaded at {datetime.now()}")
        return dataset

    def get_files_from(self, folder_path, extensions, recursive=False):
        files = []
        for ext in extensions:
            files.extend([os.path.join(root, file) for root, _, filenames in os.walk(folder_path) for file in filenames if file.lower().endswith(ext)])
        return files

    def create_dataset(self, folder_path: str) -> None:
        model_path = 'efficientnet-lite4-11.onnx'  # Path to the ONNX model file

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        extensions = ["bmp", "png", "jpeg", "jpg"]
        files = self.get_files_from(folder_path, extensions, False)

        Data = Dataset(folder_path, len(files))

        for i, file_path in enumerate(files):
            Data.Filenames[i] = os.path.relpath(file_path, folder_path)

            image = Image.open(file_path).convert('RGB')
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
            Data.Vectors[i] = output[0].flatten()

        Data.Vectors = [vector.tolist() for vector in Data.Vectors]

        json_data = json.dumps(Data, default=lambda o: o.__dict__)
        print("JSON Length:", len(json_data))
        with open(os.path.join(folder_path, "dataset.json"), "w") as json_file:
            json_file.write(json_data)
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