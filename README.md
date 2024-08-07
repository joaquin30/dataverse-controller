# Dataverse Controller

Aplicación para visualizar conjuntos de imágenes que maneja las diferentes técnicas de reducción de dimensionalidad y clusterización para ser comparadas lado a lado
Para usarse primero instale las librerias en `requirements.txt` y ejecute `python main.py`. Puede usarse de manera individual o conectarse a un visualizador
VR que se encuentra en <https://github.com/Nyanzey/Dataverse-Navigator>.

## Librerias usadas

- umap-learn
- scikit-learn
- numpy
- dearpygui
- onnx-runtime (con el modelo efficientnet-lite4-11.onnx)

## Características

### Selección de carpeta de imágenes

![](images/1.png)

### Múltiples algoritmos para seleccionar

![](images/2.png)

### Parámetros interactivos para cada técnica

![](images/3.png)

### Generación de puntos 2D

![](images/4.png)

### Coloreo de clusters

![](images/5.png)

### Visualización de selección de puntos

![](images/6.png)

### Todo esto con guias para evitar perderse en el programa

## Técnicas disponibles

- UMAP
- T-SNE
- PCA
- HDBSCAN
- K-Means
- OPTICS
- Spectral

## Créditos

- Bruno Fernandez Gutierrez (bruno.fernandez@ucsp.edu.pe)
- Joaquin Pino Zavala (joaquin.pino@ucsp.edu.pe)
- Fredy Quispe Neira (fredy.quispe@ucsp.edu.pe)
