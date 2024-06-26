import dearpygui.dearpygui as dpg
import utils
from PIL import Image
import numpy as np

class UIManager:
    WIDTH = 1280
    HEIGHT = 720
    IMAGE_SIZE = (90, 90)

    def __init__(self):
        dpg.create_context()
        self.tabs_by_index = {}
        self.tabs_by_id = {}
        self.image_ids = []

    def connect_managers(self, conn_manager, data_manager):
        self.conn_manager = conn_manager
        self.data_manager = data_manager

    def run(self):  
        with dpg.window(label="Dataverse controler") as win:
            dpg.set_primary_window(win, True)
            with dpg.window(modal=True, no_title_bar=True, show=False) as popup:
                self.popup_id = popup
                dpg.add_text("Nuevo espacio de trabajo")
                dpg.add_separator()
                dpg.add_text("Algoritmo de reducción de dim.")
                algorithm = dpg.add_combo(items=["PCA", "T-SNE", "UMAP"], default_value="UMAP")
                dpg.add_text("Algoritmo de clusterización")
                clustering = dpg.add_combo(items=["K-means", "HDBSCAN", "OPTICS", "Spectral"], default_value="HDBSCAN")
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Crear", callback=lambda: self.create_tab(dpg.get_value(algorithm), dpg.get_value(clustering)))
                    dpg.add_button(label="Cancelar", callback=lambda: dpg.configure_item(self.popup_id, show=False))

            with dpg.window(modal=True, no_title_bar=True, show=False) as about:
                self.about_id = about
                dpg.add_text("Dataverse Controller")
                dpg.add_separator()
                dpg.add_text("Creado por:")
                dpg.add_text("  - Bruno Fernandez\n  - Fredy Quispe\n  - Joaquin Pino")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Salir", callback=lambda: dpg.configure_item(self.about_id, show=False))

            with dpg.menu_bar():
                with dpg.menu(label="Archivo"):
                    dpg.add_menu_item(label="Cargar imágenes")
                    dpg.add_menu_item(label="Crear espacio de trabajo", callback=lambda: dpg.configure_item(self.popup_id, show=True))
                    dpg.add_menu_item(label="Eliminar espacio de trabajo actual", callback=self.delete_tab)
                    dpg.add_menu_item(label="Desconectar navegador", callback=self.conn_manager.disconnect)

                with dpg.menu(label="Ayuda"):
                    dpg.add_menu_item(label="Activar tutorial")
                    dpg.add_menu_item(label="Abrir manual de uso")
                    dpg.add_menu_item(label="Acerca de", callback=lambda: dpg.configure_item(self.about_id, show=True))
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Cargar imágenes")
                dpg.add_button(label="Crear espacio de trabajo", callback=lambda: dpg.configure_item(self.popup_id, show=True))
                dpg.add_button(label="Eliminar espacio de trabajo actual", callback=self.delete_tab)
                self.status_id = dpg.add_text("Navegador desconectado :(")

            self.tab_bar_id = dpg.add_tab_bar()
            self.next_tab_index = 1

        dpg.create_viewport(title="Dataverse Controller", width=self.WIDTH, height=self.HEIGHT)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def create_texture_registry(self, size):
        print("Cargando imágenes a la GPU")
        self.images_ids = []
        texture_data = []
        for i in range(120 * 120):
            texture_data.append(1)
            texture_data.append(0)
            texture_data.append(0)
            texture_data.append(1)

        with dpg.texture_registry():
            # print("hola")
            for i in range(size):
                # print(i, self.data_manager.get_path(i))
                with Image.open(self.data_manager.get_path(i)) as im:
                    im.thumbnail(self.IMAGE_SIZE)
                    im = im.convert("RGBA")
                    a = np.array(im) / 255
                    # print(a.shape)
                    image_id = dpg.add_static_texture(width=a.shape[1], height=a.shape[0], default_value=a)
                    self.image_ids.append(image_id)
    
    def get_image_id(self, index: int) -> int:
        return self.image_ids[index]

    def create_tab(self, algorithm, clustering):
        index = self.next_tab_index 
        self.next_tab_index += 1
        tab = TabManager(self, self.data_manager, self.conn_manager, index)
        id = tab.create(self.tab_bar_id, algorithm, clustering)
        self.tabs_by_id[id] = tab
        self.tabs_by_index[index] = tab
        dpg.configure_item(self.popup_id, show=False)
    
    def delete_tab(self):
        try:
            id = dpg.get_value(self.tab_bar_id)
            index = self.tabs_by_id[id].get_index()
            self.tabs_by_id[id].delete()
            del self.tabs_by_id[id]
            del self.tabs_by_index[index]
        except:
            print("no hay pestañas abiertas")
    
    def set_selection(self, workspace_id: int, indexes: list[int]):
        if workspace_id in self.tabs_by_index:
            self.tabs_by_index[workspace_id].set_selection(indexes)
    
    def clear_selection(self, workspace_id: int):
        if workspace_id in self.tabs_by_index:
            self.tabs_by_index[workspace_id].clear_selection()
    
    def set_navigator_status(self, status: bool):
        if status:
            dpg.set_value(self.status_id, "Navegador conectado :)")
        else:
            dpg.set_value(self.status_id, "Navegador desconectado :(")

class TabManager:
    PARAMETER_WIDTH = 150
    IMAGE_COLUMNS = 4

    def __init__(self, ui_manager, data_manager, conn_manager, index):
        self.ui_manager = ui_manager
        self.data_manager = data_manager
        self.conn_manager = conn_manager
        self.xdata = []
        self.ydata = []
        self.labels = []
        self.selected = []
        self.index = index
        self.scatters = []
    
    def create(self, parent, algorithm, clustering):
        with dpg.tab(label=f"Espacio de trabajo {self.index} ({algorithm} - {clustering})", parent=parent) as tab:
            with dpg.table(header_row=False):
                dpg.add_table_column(init_width_or_weight=0.3)
                dpg.add_table_column(init_width_or_weight=0.7)
                with dpg.table_row():
                    with dpg.group() as cell:
                        if algorithm == "UMAP":
                            self.create_umap(cell)
                        elif algorithm == "T-SNE":
                            self.create_tsne(cell)
                        elif algorithm == "PCA":
                            self.create_pca(cell)
                        else:
                            raise Exception("algoritmo de reduccion de dim no existe")
                        
                        dpg.add_separator()
                        if clustering == "HDBSCAN":
                            self.create_hdbscan(cell)
                        elif clustering == "K-means":
                            self.create_kmeans(cell)
                        elif clustering == "OPTICS":
                            self.create_optics(cell)
                        elif clustering == "Spectral":
                            self.create_spectral(cell)
                        else:
                            raise Exception("algoritmo de clusterizacion no existe")
                        
                        dpg.add_separator()
                        dpg.add_text("Imágenes seleccionadas (Navegador)")
                        with dpg.group() as table_parent_id:
                            self.table_parent_id = table_parent_id
                            self.table_id = dpg.add_table(header_row=False)
                        
                    with dpg.plot(label="Imágenes en 2D", width=870, height=590):
                        self.xaxis = dpg.add_plot_axis(dpg.mvXAxis)
                        self.yaxis = dpg.add_plot_axis(dpg.mvYAxis)
            
            self.id = tab
            return tab
    
    def delete(self):
        dpg.delete_item(self.id)
        self.conn_manager.delete_workspace(self.index)
    
    def get_id(self): return self.id
    def get_index(self): return self.index

    def set_selection(self, selected_points):
        for i in selected_points:
            self.selected[i] = True
        
        dpg.delete_item(self.table_id)
        count = 0
        for point in self.selected:
            if point:
                count += 1
        
        index = 0
        with dpg.table(parent=self.table_parent_id, header_row=False, height=290, scrollY=True) as table_id:
            self.table_id = table_id
            for i in range(self.IMAGE_COLUMNS):
                dpg.add_table_column()
            
            while index < count:
                with dpg.table_row():
                    for i in range(self.IMAGE_COLUMNS):
                        if index >= count:
                            break
                            
                        dpg.add_image(self.ui_manager.get_image_id(index))
                        index += 1

        self.update_plot()

    def clear_selection(self):
        self.selected = [False for _ in range(len(self.labels))]
        dpg.delete_item(self.table_id)
        self.table_id = dpg.add_table(parent=self.table_parent_id, header_row=False)
        self.update_plot()

    def update_plot(self):
        for id in self.scatters:
            dpg.delete_item(id)

        self.scatters = []
        colors = utils.get_colors(max(self.labels) + 1)

        # No seleccionados
        xs, ys = self.filter_by_label_and_selection(-1, False)
        if len(xs) > 0:
            # Coloreo de outliers (-1)
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 255), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 255), category=dpg.mvThemeCat_Plots)

            scatter = dpg.add_scatter_series(xs, ys, parent=self.yaxis)
            dpg.bind_item_theme(scatter, theme)
            self.scatters.append(scatter)
        
        for label in range(len(colors)):
            xs, ys = self.filter_by_label_and_selection(label, False)
            if len(xs) > 0:
                with dpg.theme() as theme:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, colors[label], category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, colors[label], category=dpg.mvThemeCat_Plots)

                scatter = dpg.add_scatter_series(xs, ys, parent=self.yaxis)
                dpg.bind_item_theme(scatter, theme)
                self.scatters.append(scatter)

        # Seleccionados
        xs, ys = self.filter_by_label_and_selection(-1, True)
        if len(xs) > 0:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 255), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (0, 0, 0), category=dpg.mvThemeCat_Plots)

            scatter = dpg.add_scatter_series(xs, ys, parent=self.yaxis)
            dpg.bind_item_theme(scatter, theme)
            self.scatters.append(scatter)
        
        for label in range(len(colors)):
            xs, ys = self.filter_by_label_and_selection(label, True)
            if len(xs) > 0:
                with dpg.theme() as theme:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, colors[label], category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 255, 255), category=dpg.mvThemeCat_Plots)

                scatter = dpg.add_scatter_series(xs, ys, parent=self.yaxis)
                dpg.bind_item_theme(scatter, theme)
                self.scatters.append(scatter)

        dpg.fit_axis_data(self.xaxis)
        dpg.fit_axis_data(self.yaxis)

    def filter_by_label_and_selection(self, label, selected):
        xs, ys = [], []
        for i in range(len(self.xdata)):
            if self.labels[i] == label and self.selected[i] == selected:
                xs.append(self.xdata[i])
                ys.append(self.ydata[i])
        
        return xs, ys
    
    ### REDUCCION DIM ###

    def create_umap(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros UMAP")
            n_neighbors = dpg.add_input_int(label="n_neighbors", default_value=15, width=self.PARAMETER_WIDTH)
            min_dist = dpg.add_input_float(label="min_dist", default_value=0.01, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                metric = dpg.add_combo(items=["euclidean", "manhattan", "cosine", "correlation"], default_value="euclidean", width=self.PARAMETER_WIDTH)
                dpg.add_text("metric")

            dpg.add_spacer(height=5)
            dpg.add_button(label="Aplicar", callback=lambda: self.apply_umap(dpg.get_value(n_neighbors),
                                                                             dpg.get_value(min_dist),
                                                                             dpg.get_value(metric)))
    
    def create_tsne(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros T-SNE")
            learning_rate = dpg.add_input_float(label="learning_rate", default_value=200.0, width=self.PARAMETER_WIDTH)
            perplexity = dpg.add_input_float(label="perplexity", default_value=30.0, width=self.PARAMETER_WIDTH)
            early_exaggeration = dpg.add_input_float(label="early_exaggeration", default_value=12.0, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                metric = dpg.add_combo(items=["euclidean", "manhattan", "cosine", "correlation"], default_value="euclidean", width=self.PARAMETER_WIDTH)
                dpg.add_text("metric")

            dpg.add_spacer(height=5)
            dpg.add_button(label="Aplicar", callback=lambda: self.apply_tsne(dpg.get_value(learning_rate),
                                                                             dpg.get_value(perplexity),
                                                                             dpg.get_value(early_exaggeration),
                                                                             dpg.get_value(metric)))

    def create_pca(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros PCA")
            whiten = dpg.add_checkbox(label="whiten", default_value=True) # width=self.PARAMETER_WIDTH produce an error
            tolerance = dpg.add_input_float(label="tolerance", default_value=0.0, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                svd_solver = dpg.add_combo(items=["auto", "full", "arpack", "randomized"], default_value="auto", width=self.PARAMETER_WIDTH)
                dpg.add_text("svd_solver")

            dpg.add_spacer(height=5)
            dpg.add_button(label="Aplicar", callback=lambda: self.apply_pca(dpg.get_value(whiten),
                                                                            dpg.get_value(tolerance),
                                                                            dpg.get_value(svd_solver)))

    def clear_plot(self):
        if len(self.labels) == 0:
            self.labels = [-1 for _ in range(len(self.xdata))]
            self.selected = [False for _ in range(len(self.xdata))]
            self.update_plot()
        else:
            self.clear_clustering()
            self.clear_selection()

    def apply_umap(self, n_neighbors, min_dist, metric):
        self.xdata, self.ydata = self.data_manager.apply_umap(self.index, n_neighbors, min_dist, metric)
        self.clear_plot()

    def apply_tsne(self, learning_rate, perplexity, early_exaggeration, metric):
        self.xdata, self.ydata = self.data_manager.apply_tsne(self.index, learning_rate, perplexity, early_exaggeration, metric)
        self.clear_plot()
    
    def apply_pca(self, whiten, tolerance, svd_solver):
        self.xdata, self.ydata = self.data_manager.apply_pca(self.index, whiten, tolerance, svd_solver)
        self.clear_plot()

    ### CLUSTERIZACION ###

    def create_hdbscan(self, parent):    
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros HDBSCAN")
            min_cluster_size = dpg.add_input_int(label="min_cluster_size", default_value=5, width=self.PARAMETER_WIDTH)
            min_samples = dpg.add_input_int(label="min_samples", default_value=5, width=self.PARAMETER_WIDTH)
            cluster_selection_epsilon = dpg.add_input_float(label="cluster_selection_epsilon", default_value=0, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                cluster_selection_method = dpg.add_combo(items=["eom", "leaf"], default_value="eom", width=self.PARAMETER_WIDTH)
                dpg.add_text("cluster_selection_method")
            
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Aplicar", callback=lambda: self.apply_hdbscan(dpg.get_value(min_cluster_size),
                                                                                    dpg.get_value(min_samples),
                                                                                    dpg.get_value(cluster_selection_epsilon),
                                                                                    dpg.get_value(cluster_selection_method)))
                dpg.add_button(label="Limpiar clustering", callback=self.clear_clustering)
    
    def create_kmeans(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros K-Means")
            n_clusters = dpg.add_input_int(label="n_clusters", default_value=8, width=self.PARAMETER_WIDTH)
            max_iter = dpg.add_input_int(label="max_iter", default_value=300, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                init = dpg.add_combo(items=["k-means++", "random"], default_value="k-means++", width=self.PARAMETER_WIDTH)
                dpg.add_text("init")
            with dpg.group(horizontal=True):
                algorithm = dpg.add_combo(items=["lloyd", "elkan"], default_value="lloyd", width=self.PARAMETER_WIDTH)
                dpg.add_text("algorithm")

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Aplicar", callback=lambda: self.apply_kmeans(dpg.get_value(n_clusters),
                                                                                   dpg.get_value(max_iter),
                                                                                   dpg.get_value(init),
                                                                                   dpg.get_value(algorithm)))
                dpg.add_button(label="Limpiar clustering", callback=self.clear_clustering)

    def create_optics(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros OPTICS")
            min_samples = dpg.add_input_int(label="min_samples", default_value=5, width=self.PARAMETER_WIDTH, min_value=2, min_clamped=True)
            max_eps = dpg.add_input_float(label="max_eps", default_value=float("inf"), width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                metric = dpg.add_combo(items=["minkowski", "manhattan", "cosine", "correlation"], default_value="minkowski", width=self.PARAMETER_WIDTH)
                dpg.add_text("metric")
            with dpg.group(horizontal=True):
                cluster_method = dpg.add_combo(items=["xi", "dbscan"], default_value="xi", width=self.PARAMETER_WIDTH)
                dpg.add_text("cluster_method")

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Aplicar", callback=lambda: self.apply_optics(dpg.get_value(min_samples),
                                                                                    dpg.get_value(max_eps),
                                                                                    dpg.get_value(metric),
                                                                                    dpg.get_value(cluster_method)))
                dpg.add_button(label="Limpiar clustering", callback=self.clear_clustering)

    def create_spectral(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros Spectral Clustering")
            n_clusters = dpg.add_input_int(label="n_clusters", default_value=8, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                eigen_solver = dpg.add_combo(items=["arpack", "lobpcg", "amg"], default_value="arpack", width=self.PARAMETER_WIDTH)
                dpg.add_text("eigen_solver")
            with dpg.group(horizontal=True):
                affinity = dpg.add_combo(items=["nearest_neighbors", "rbf", "precomputed"], default_value="rbf", width=self.PARAMETER_WIDTH)
                dpg.add_text("affinity")
            with dpg.group(horizontal=True):
                assign_labels = dpg.add_combo(items=["kmeans", "discretize"], default_value="kmeans", width=self.PARAMETER_WIDTH)
                dpg.add_text("assign_labels")

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Aplicar", callback=lambda: self.apply_spectral(dpg.get_value(n_clusters),
                                                                                    dpg.get_value(eigen_solver),
                                                                                    dpg.get_value(affinity),
                                                                                    dpg.get_value(assign_labels)))
                dpg.add_button(label="Limpiar clustering", callback=self.clear_clustering)

    def apply_hdbscan(self, min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method):
        self.labels = self.data_manager.apply_hdbscan(self.index, min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method)
        self.update_plot()

    def apply_kmeans(self, n_clusters, max_iter, init, algorithm):
        self.labels = self.data_manager.apply_kmeans(self.index, n_clusters, max_iter, init, algorithm)
        self.update_plot()

    def apply_optics(self, min_samples, cluster_selection_epsilon, metric, cluster_method):
        self.labels = self.data_manager.apply_optics(self.index, min_samples, cluster_selection_epsilon, metric, cluster_method)
        self.update_plot()

    def apply_spectral(self, n_clusters, eigen_solver, affinity, assign_labels):
        self.labels = self.data_manager.apply_spectral(self.index, n_clusters, eigen_solver, affinity, assign_labels)
        self.update_plot()
    
    def clear_clustering(self):
        self.labels = [-1 for _ in range(len(self.xdata))]
        self.update_plot()