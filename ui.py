import dearpygui.dearpygui as dpg

class UIManager:
    WIDTH = 1280
    HEIGHT = 720

    def __init__(self):
        pass

    def connect_managers(self, conn_manager, data_manager):
        self.conn_manager = conn_manager
        self.data_manager = data_manager
        self.tabs_by_index = {}
        self.tabs_by_id = {}

    def run(self):
        dpg.create_context()
        with dpg.window(label="Dataverse controler") as win:
            dpg.set_primary_window(win, True)
            with dpg.window(modal=True, no_title_bar=True, show=False) as popup:
                self.popup_id = popup
                dpg.add_text("Nuevo espacio de trabajo")
                dpg.add_separator()
                dpg.add_text("Algoritmo de reducción de dim.")
                algorithm = dpg.add_combo(items=["PCA", "T-SNE", "UMAP"], default_value="UMAP")
                dpg.add_text("Algoritmo de clusterización")
                clustering = dpg.add_combo(items=["K-means", "HDBSCAN"], default_value="HDBSCAN")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Crear", callback=lambda: self.create_tab(dpg.get_value(algorithm), dpg.get_value(clustering)))
                    dpg.add_button(label="Cancelar", callback=lambda: dpg.configure_item(self.popup_id, show=False))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Cargar imágenes")
                dpg.add_button(label="Crear espacio de trabajo", callback=lambda: dpg.configure_item(self.popup_id, show=True))
                dpg.add_button(label="Eliminar espacio de trabajo actual", callback=self.delete_tab)

            with dpg.menu_bar():
                with dpg.menu(label="Archivo"):
                    dpg.add_menu_item(label="Cargar imágenes")
                    dpg.add_menu_item(label="Salir")

                with dpg.menu(label="Opciones"):
                    dpg.add_menu_item(label="Crear espacio de trabajo", callback=lambda: dpg.configure_item(self.popup_id, show=True))
                    dpg.add_menu_item(label="Eliminar espacio de trabajo actual", callback=self.delete_tab)
                    dpg.add_menu_item(label="Reiniciar conexion con el navegador")

                with dpg.menu(label="Ayuda"):
                    dpg.add_menu_item(label="Activar tutorial")
                    dpg.add_menu_item(label="Abrir manual de uso")
                    dpg.add_menu_item(label="Acerca de")

            self.tab_bar_id = dpg.add_tab_bar()
            self.next_tab_index = 1

        dpg.create_viewport(title="Dataverse Controller", width=self.WIDTH, height=self.HEIGHT)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def create_tab(self, algorithm, clustering):
        index = self.next_tab_index 
        self.next_tab_index += 1
        tab = TabManager(self.data_manager, self.conn_manager, index)
        id = tab.create(self.tab_bar_id, algorithm, clustering)
        self.tabs_by_id[id] = tab
        self.tabs_by_index[index] = tab
        dpg.configure_item(self.popup_id, show=False)
    
    def delete_tab(self):
        id = dpg.get_value(self.tab_bar_id)

        if id != None:
            index = self.tabs_by_id[id].get_index()
            self.tabs_by_id[id].delete()
            del self.tabs_by_id[id]
            del self.tabs_by_index[index]
    
    def set_selection(self, workspace_id, indexes):
        if workspace_id in self.tabs_by_index:
            self.tabs_by_index[workspace_id].set_selection(indexes)
    
    def clear_selection(self, workspace_id):
        if workspace_id in self.tabs_by_index:
            self.tabs_by_index[workspace_id].clear_selection()

class TabManager:
    PARAMETER_WIDTH = 150

    def __init__(self, data_manager, conn_manager, index):
        self.data_manager = data_manager
        self.conn_manager = conn_manager
        self.xdata = []
        self.ydata = []
        self.labels = []
        self.selected = []
        self.index = index
    
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
                        else:
                            raise Exception("algoritmo de clusterizacion no existe")
                        
                        dpg.add_separator()
                        dpg.add_text("Imágenes seleccionadas (Navegador)")
                        
                    with dpg.plot(label="Imágenes en 2D", width=870, height=590):
                        self.xaxis = dpg.add_plot_axis(dpg.mvXAxis)
                        with dpg.plot_axis(dpg.mvYAxis) as ax:
                            self.yaxis = ax
                            self.plot_id = dpg.add_scatter_series([], [])
            
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

        self.update_plot()

    def clear_selection(self):
        self.selected = [False for _ in range(len(self.xdata))]
        self.update_plot()

    def update_plot(self):
        # TODO colores y seleccion
        dpg.set_value(self.plot_id, [self.xdata, self.ydata])
        dpg.fit_axis_data(self.xaxis)
        dpg.fit_axis_data(self.yaxis)

    ### REDUCCION DIM ###

    def create_umap(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros UMAP")
            n_neighbors = dpg.add_input_int(label="n_neighbors", default_value=15, width=self.PARAMETER_WIDTH)
            min_dist = dpg.add_input_float(label="min_dist", default_value=0.01, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                metric = dpg.add_combo(items=["euclidean", "manhattan", "cosine", "correlation"], default_value="euclidean", width=self.PARAMETER_WIDTH)
                dpg.add_text("metric")

            dpg.add_button(label="Aplicar", callback=lambda: self.apply_umap(dpg.get_value(n_neighbors), dpg.get_value(min_dist), dpg.get_value(metric)))
    
    def create_tsne(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros TSNE")
            learning_rate = dpg.add_input_float(label="learning_rate", default_value=200.0, width=self.PARAMETER_WIDTH)
            perplexity = dpg.add_input_float(label="perplexity", default_value=30.0, width=self.PARAMETER_WIDTH)
            early_exaggeration = dpg.add_input_float(label="early_exaggeration", default_value=12.0, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                metric = dpg.add_combo(items=["euclidean", "manhattan", "cosine", "correlation"], default_value="euclidean", width=self.PARAMETER_WIDTH)
                dpg.add_text("metric")

            dpg.add_button(label="Aplicar", callback=lambda: self.apply_tsne(dpg.get_value(learning_rate), dpg.get_value(perplexity), dpg.get_value(early_exaggeration), dpg.get_value(metric)))

    def create_pca(self, parent):
        with dpg.group(parent=parent):
            dpg.add_text("Parámetros PCA")
            whiten = dpg.add_checkbox(label="whiten", default_value=True) # width=self.PARAMETER_WIDTH produce an error
            tolerance = dpg.add_input_float(label="tolerance", default_value=0.0, width=self.PARAMETER_WIDTH)
            with dpg.group(horizontal=True):
                svd_solver = dpg.add_combo(items=["auto", "full", "arpack", "randomized"], default_value="auto", width=self.PARAMETER_WIDTH)
                dpg.add_text("svd_solver")

            dpg.add_button(label="Aplicar", callback=lambda: self.apply_pca(dpg.get_value(whiten), dpg.get_value(tolerance), dpg.get_value(svd_solver)))

    def apply_umap(self, n_neighbors, min_dist, metric):
        self.xdata, self.ydata = self.data_manager.apply_umap(self.index, n_neighbors, min_dist, metric)
        self.labels = [-1 for _ in range(len(self.xdata))]
        self.selected = [False for _ in range(len(self.xdata))]
        self.update_plot()

    def apply_tsne(self, learning_rate, perplexity, early_exaggeration, metric):
        self.xdata, self.ydata = self.data_manager.apply_tsne(self.index, learning_rate, perplexity, early_exaggeration, metric)
        self.labels = [-1 for _ in range(len(self.xdata))]
        self.selected = [False for _ in range(len(self.xdata))]
        self.update_plot()
    
    def apply_pca(self, whiten, tolerance, svd_solver):
        self.xdata, self.ydata = self.data_manager.apply_pca(self.index, whiten, tolerance, svd_solver)
        self.labels = [-1 for _ in range(len(self.xdata))]
        self.selected = [False for _ in range(len(self.xdata))]
        self.update_plot()

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
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Aplicar", callback=lambda: self.apply_hdbscan(min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method))
                dpg.add_button(label="Limpiar clustering", callback=self.clear_clustering)
    
    def apply_hdbscan(self, min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method):
        self.labels = self.data_manager.apply_hdbscan(self.index, self.xdata, self.ydata, min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method)
        self.update_plot()
    
    # TODO otros metodos de clustering
    
    def clear_clustering(self):
        self.labels = [-1 for _ in range(len(self.xdata))]
        self.update_plot()