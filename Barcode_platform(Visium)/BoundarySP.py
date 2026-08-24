import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import infercnvpy as cnv
import BoundarySP_utility  
import json
from GraphST import GraphST
import rpy2
import os
import torch
import multiprocessing as mp

# Set device for GPU acceleration if available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = "/usr/lib/R"


class pp:
    """
    Preprocessing and spatial analysis utility class for spatial transcriptomics data.
    Integrates GraphST clustering, inferCNV analysis, and boundary detection.
    """

    def __init__(self):
        pass

    @staticmethod
    def umap_plot(adata, color):
        """
        Generate UMAP visualization with a custom color palette.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix containing UMAP embeddings in `obsm['X_umap']`.
        color : str or list of str
            Column(s) in `adata.obs` to use for coloring the UMAP plot.
        """
        coords = adata.obsm['spatial']
        # Ensure spatial coordinates are numeric
        if not np.issubdtype(coords[:, 0].dtype, np.number):
            coords[:, 0] = coords[:, 0].astype(float)
        if not np.issubdtype(coords[:, 1].dtype, np.number):
            coords[:, 1] = coords[:, 1].astype(float)

        # Custom color map for consistent visualization across plots
        color_map = [
            "#DC050C", "#FB8072", "#1965B0", "#7BAFDE", "#882E72",
            "#B17BA6", "#FF7F00", "#FDB462", "#E7298A", "#E78AC3",
            "#33A02C", "#B2DF8A", "#55B1B1", "#8DD3C7", "#A6761D",
            "#E6AB02", "#7570B3", "#BEAED4", "#666666", "#999999",
            "#aa8282", "#d4b7b7", "#8600bf", "#ba5ce3", "#808000",
            "#aeae5c", "#1e90ff", "#00bfff", "#56ff0d", "#ffff00"
        ]
        sc.pl.umap(adata=adata, color=color, palette=color_map, wspace=0.4)

    @staticmethod
    def spatial_plot(adata, color):
        """
        Generate spatial tissue visualization with a custom color palette.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix containing spatial coordinates in `obsm['spatial']`.
        color : str or list of str
            Column(s) in `adata.obs` to use for coloring the spatial plot.
        """
        color_map = [
            "#DC050C", "#FB8072", "#1965B0", "#7BAFDE", "#882E72",
            "#B17BA6", "#FF7F00", "#FDB462", "#E7298A", "#E78AC3",
            "#33A02C", "#B2DF8A", "#55B1B1", "#8DD3C7", "#A6761D",
            "#E6AB02", "#7570B3", "#BEAED4", "#666666", "#999999",
            "#aa8282", "#d4b7b7", "#8600bf", "#ba5ce3", "#808000",
            "#aeae5c", "#1e90ff", "#00bfff", "#56ff0d", "#ffff00"
        ]
        sc.pl.spatial(adata=adata, color=color, palette=color_map)

    @staticmethod
    def preproboundary(type, path, library_id, n_clusters, method, radius):
        """
        Preprocess spatial transcriptomics data, perform GraphST clustering,
        and identify the putative normal cell cluster based on immune/stromal markers.

        Parameters
        ----------
        type : str
            Data format type: 'Visium', 'h5ad', or 'VisiumHD'.
        path : str
            Path to the data directory or h5ad file.
        library_id : str
            Library ID for Visium data.
        n_clusters : int
            Target number of clusters for spatial domain identification.
        method : str
            Clustering algorithm: 'mclust', 'leiden', or 'louvain'.
        radius : int
            Radius parameter for neighborhood refinement during clustering.

        Returns
        -------
        adata : anndata.AnnData
            Processed AnnData object with clustering results, UMAP embeddings,
            and an 'immune_score' column identifying the normal cluster.
        """
        if type == "Visium":
            adata = sc.read_visium(
                path=path, library_id=library_id,
                count_file="filtered_feature_bc_matrix.h5",
                source_image_path="spatial"
            )
            adata.var_names_make_unique()
            # Truncate decimal part, keep only integer coordinates
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
            adata.obsm['spatial'] = (adata.obsm['spatial'] // 1).astype(int)
            # Define and train GraphST model
            model = GraphST.GraphST(adata, device=device)
            adata = model.train()

        elif type == "h5ad":
            adata = sc.read_h5ad(filename=path)
            adata.var_names_make_unique()
            # Truncate decimal part, keep only integer coordinates
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
            adata.obsm['spatial'] = (adata.obsm['spatial'] // 1).astype(int)
            # Define and train GraphST model
            model = GraphST.GraphST(adata, device=device)
            adata = model.train()

        elif type == "VisiumHD":
            leadout_position = pd.read_parquet(path + "spatial/tissue_positions.parquet")
            leadout_position.to_csv(path + "spatial/tissue_positions_list.csv", index=False, header=None)
            # Truncate decimal part, keep only integer coordinates
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
            adata.obsm['spatial'] = (adata.obsm['spatial'] // 1).astype(int)
            # Convert coordinates to float32 instead of int64 for compatibility
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
            # Define and train GraphST model with Stereo datatype
            model = GraphST.GraphST(adata, device=device, datatype='Stereo')
            adata = model.train()
        else:
            raise ValueError("Unsupported type. Please use 'Visium', 'h5ad', or 'VisiumHD'.")

        # Set radius for neighbor consideration during refinement
        tool = method  # mclust, leiden, or louvain

        # Perform clustering
        from GraphST.utils import clustering
        if tool == 'mclust':
            clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
        elif tool in ['leiden', 'louvain']:
            clustering(adata, n_clusters, radius=radius, method=tool,
                       start=0.1, end=2.0, increment=0.1, refinement=False)

        # Compute neighbors and UMAP on PCA embeddings
        sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
        sc.tl.umap(adata)
        adata.obs = adata.obs.rename(columns={'domain': 'clusters'})

        # Identify recommended normal cluster based on immune/stromal marker expression
        immune_features = ["CD3D", "CD3E", "CD3G", "CD34", "CD79A", "COL1A1", "COL6A1", "TTR", "ALB"]
        adata.var["gene"] = adata.var.index

        coords = adata.obsm['spatial']
        if not np.issubdtype(coords[:, 0].dtype, np.number):
            coords[:, 0] = coords[:, 0].astype(float)
        if not np.issubdtype(coords[:, 1].dtype, np.number):
            coords[:, 1] = coords[:, 1].astype(float)

        adata.var["gene"] = adata.var.index
        immune_data = adata[:, adata.var.index.isin(immune_features)]
        score = np.array(immune_data.X.mean(axis=1), dtype="float64")
        adata.obs['immune_score'] = score

        # The cluster with highest mean immune score is designated as normal
        NormalCluster = (
            adata.obs[["clusters", "immune_score"]]
            .groupby('clusters')['immune_score']
            .mean()
            .sort_values(ascending=False)
            .idxmax()
        )
        print(f'Putative normal cell cluster: {NormalCluster}')

        # Ensure spatial coordinates are stored as float64
        spatial_data = np.array(adata.obsm['spatial'], dtype=np.float64)
        adata.obsm['spatial'] = spatial_data

        return adata

    @staticmethod
    def bdy_cnv(adata, gtf_file, reference_key, reference_cat, resolution):
        """
        Run inferCNV analysis to detect copy number variations from spatial data.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix with genomic position annotations.
        gtf_file : str
            Path to GTF annotation file for mapping genes to genomic positions.
        reference_key : str
            Column in `adata.obs` defining the reference category key.
        reference_cat : str
            Value in `reference_key` column identifying normal/reference cells.
        resolution : float
            Resolution parameter for Leiden clustering on CNV profiles.

        Returns
        -------
        adata : anndata.AnnData
            AnnData object augmented with CNV inference results, PCA, UMAP, and CNV scores.
        """
        cnv.io.genomic_position_from_gtf(
            adata=adata, gtf_file=gtf_file,
            gtf_gene_id='gene_name', inplace=True
        )
        adata.var.loc[:, ["chromosome", "start", "end"]].isna().sum()

        cnv.tl.infercnv(adata=adata, reference_key=reference_key, reference_cat=reference_cat)
        cnv.tl.pca(adata)
        cnv.pp.neighbors(adata)
        cnv.tl.leiden(adata, resolution=resolution)
        cnv.tl.umap(adata)
        cnv.tl.cnv_score(adata)

        return adata

    @staticmethod
    def cnv_heatmap(adata, groupby):
        """
        Plot chromosome-level CNV heatmap grouped by cell clusters.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object with inferred CNV values.
        groupby : str
            Column in `adata.obs` to group spots/cells in the heatmap.
        """
        cnv.pl.chromosome_heatmap(adata, groupby=groupby, dendrogram=True)

    @staticmethod
    def cnv_cnv_score_boxplot(adata, clusters):
        """
        Generate boxplot of CNV scores across clusters with mean and max-mean reference lines.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing 'cnv_score' in `adata.obs`.
        clusters : str
            Column name in `adata.obs` representing cluster labels.
        """
        sns.boxplot(
            x=adata.obs[clusters], y=adata.obs.cnv_score,
            data=adata.to_df(), showmeans=True, meanline=True,
            meanprops={'ls': '--', 'ms': 10}
        )
        median_normalscore = adata.obs.groupby(clusters)['cnv_score'].mean().reset_index()
        max_median = median_normalscore["cnv_score"].max()
        plt.axhline(y=max_median, color='r', linestyle='--', linewidth=0.5)

    @staticmethod
    def boundary(data, n_tumor, selectcluster=None):
        """
        Identify tumor-normal boundary spots using spatial neighborhood analysis
        and embedding-based deviation scoring.

        Parameters
        ----------
        data : anndata.AnnData
            Preprocessed AnnData with clustering and CNV scores.
        n_tumor : int
            Number of top CNV-high clusters to define as malignant.
        selectcluster : str or None, optional
            Manually specify the normal cluster label. If None, auto-detected
            via highest immune_score.

        Returns
        -------
        spatial : anndata.AnnData
            AnnData object with refined boundary labels.
        adata : anndata.AnnData
            Original AnnData object with updated annotations.
        """
        adata = data
        UMAPembeddings = pd.DataFrame(adata.obsm['X_umap'], columns=["x", "y"])
        UMAPembeddings.index = adata.obs.index

        position1 = pd.DataFrame(adata.obs[["in_tissue", "array_row", "array_col"]])
        position2 = pd.DataFrame(adata.obsm['spatial'], columns=["imagecol", "imagerow"], index=adata.obs.index)
        position = pd.concat([position1, position2], axis=1)
        position['spot.ids'] = np.arange(1, len(position) + 1, dtype=int)
        position = position.astype(float)

        # Compute inter-spot distances and build spatial neighbor graph
        dists = BoundarySP_utility.compute_interspot_distances(position, scale_factor=1.05)
        df_j = BoundarySP_utility.find_neighbors(
            position=position, radius=dists['radius'], n=1, method="manhattan"
        )

        # Assign CNV labels from clustering results
        adata.obs["CNVLabel"] = adata.obs['clusters']
        MalLabel = None
        MalCellID = None

        if MalLabel is None:
            grouped = adata.obs.groupby('clusters')
            median_scores = grouped['cnv_score'].mean()
            print(median_scores)
            # Select top n_tumor clusters with highest mean CNV score as malignant
            MalLabel = median_scores.sort_values(ascending=False).head(n_tumor).index.tolist()
            MalCellID = np.array(adata.obs.index[adata.obs['clusters'].isin(MalLabel)])

        # Identify normal cell IDs based on immune score
        cluster_normal_score = adata.obs[['clusters', 'immune_score']]
        grouped = cluster_normal_score.groupby('clusters')
        mean_scores = grouped['immune_score'].mean()
        sorted_index = mean_scores.sort_values(ascending=False).index

        if selectcluster is None:
            NormalCluster = sorted_index.tolist()[0]
            NormalCellID = np.array(adata.obs.index[adata.obs['clusters'].isin([NormalCluster])])
        else:
            NormalCluster = selectcluster
            NormalCellID = np.array(adata.obs.index[adata.obs['clusters'].isin([NormalCluster])])

        # Confirm malignant sub-clusters: retain clusters where >50% of cells are in MalLabel
        cnv_seurat_df = pd.crosstab(adata.obs['clusters'], adata.obs['clusters'])
        cnv_seurat_df = cnv_seurat_df.loc[MalLabel]
        ClusterID = []
        for cluster in adata.obs['clusters'].unique():
            sub_cluster_colsum = cnv_seurat_df[cluster].sum()
            total_in_cluster = adata.obs['clusters'].value_counts()[cluster]
            if sub_cluster_colsum > total_in_cluster * 0.5:
                ClusterID.append(cluster)

        # Extract malignant spot IDs within confirmed malignant UMAP sub-clusters
        CiMal = pd.DataFrame({'cluster': ClusterID})

        def get_sub_MalCellID(cluster):
            cluster_cells = adata.obs.index[adata.obs['clusters'] == cluster].tolist()
            sub_MalCellID = list(set(MalCellID).intersection(cluster_cells))
            return sub_MalCellID

        CiMal['sub_MalCellID'] = CiMal['cluster'].apply(get_sub_MalCellID)

        # Compute centroid of normal cluster core in UMAP space
        NormalCellID = list(NormalCellID)
        valid_ids = [cell_id for cell_id in NormalCellID if cell_id in UMAPembeddings.index]
        if valid_ids:
            CiNormal = UMAPembeddings.loc[valid_ids].mean(axis=0)
        else:
            CiNormal = np.array([np.nan, np.nan])

        # Flatten nested lists of malignant cell IDs
        MalCellIDsi = [id for cell_id in CiMal["sub_MalCellID"] for id in cell_id]

        # Handle isolated malignant spots (no malignant neighbors)
        MalCellIDL = []
        for name in MalCellIDsi:
            if all(item not in df_j[name] for item in MalCellIDsi):
                MalCellIDL.append(name)
            else:
                MalCellIDL.append(pd.NA)
        MalCellIDL = pd.Series(MalCellIDL).dropna()

        BdyCellID = []
        CellIDRaw = MalCellIDsi + NormalCellID + BdyCellID
        nbrs_of_Mall = BoundarySP_utility.nbrs(
            df_j=df_j, MalCellIDAdd=MalCellIDL, CellIDRaw=CellIDRaw
        )

        # Initialize tumor anchor dictionary using PCA embeddings
        anchor_dict = BoundarySP_utility.init_tumor_anchor(
            adata=adata, mal_ids=MalCellIDsi, emb_key='emb_pca'
        )

        if not nbrs_of_Mall:
            print("nbrs_of_Mall is empty. Check the data or the logic that generates it.")
            ClusterL = pd.DataFrame({'CellID': [], 'Location': []})
        else:
            ClusterL = pd.DataFrame({'CellID': [], 'Location': []})
            for cell in nbrs_of_Mall.keys():
                nbrs_of_cell = nbrs_of_Mall[cell]
                cluster_data = []
                for nbr in nbrs_of_cell:
                    location = BoundarySP_utility.calculate_location_by_emb_deviation(
                        anchor_dict=anchor_dict,
                        target_id=nbr,
                        df_j=df_j,
                        adata=adata,
                        emb_key='emb_pca'
                    )
                    cluster_data.append({'CellID': nbr, 'Location': location})
                cluster_df = pd.DataFrame(cluster_data)
                ClusterL = pd.concat([ClusterL, cluster_df], ignore_index=True)

        frequency_table = pd.crosstab(ClusterL['CellID'], ClusterL['Location'])

        def set_location(cell_id):
            return "Mal" if frequency_table.loc[cell_id, 'Mal'] > 0 else "Bdy"

        try:
            ClusterL['Location'] = ClusterL['CellID'].apply(set_location)
        except Exception:
            ClusterL['Location'] = "Bdy"

        # Pre-iteration inference: consolidate Mal/Bdy/Normal labels
        MalCellID = MalCellIDsi + ClusterL[ClusterL['Location'] == "Mal"]['CellID'].tolist()
        BdyCellID = ClusterL[ClusterL['Location'] == "Bdy"]['CellID'].astype(str).tolist()
        MalCellIDN = MalCellID.copy()

        df_normal = pd.DataFrame({'CellID': NormalCellID, 'Location': ['Normal'] * len(NormalCellID)})
        df_mal = pd.DataFrame({'CellID': MalCellID, 'Location': ['Mal'] * len(MalCellID)})
        df_bdy = pd.DataFrame({'CellID': BdyCellID, 'Location': ['Bdy'] * len(BdyCellID)})
        Clustern = pd.concat([df_normal, df_mal, df_bdy], ignore_index=True)

        TumorST = adata
        spatial = BoundarySP_utility.process_clusters(
            TumorST, Clustern, MalCellIDN, NormalCellID, BdyCellID, df_j, MalCellID, anchor_dict
        )
        return spatial, adata

    @staticmethod
    def boundary_both(spatial, adata, levels):
        """
        Expand boundary labels outward by hexagonal neighborhood levels to create
        graded tumor-normal transition zones.

        Parameters
        ----------
        spatial : anndata.AnnData
            Spatial AnnData with initial boundary labels ('LabelNew').
        adata : anndata.AnnData
            Original AnnData object to be annotated with expanded boundary levels.
        levels : int
            Number of hexagonal expansion levels around boundary spots.

        Returns
        -------
        spatial_df_j : anndata.AnnData
            Updated AnnData with hierarchical boundary labels (e.g., Mal_100, Normal_200).
        """
        spatial_dataframe = spatial.obs[['barcode', 'LabelNew']]
        index = adata.obs.index
        adata.obs['barcode'] = adata.obs.index
        adata.obs = adata.obs.merge(spatial_dataframe[["barcode", 'LabelNew']], on="barcode", how="left")
        adata.obs.index = index

        spatial_df_j = adata.copy()
        spatial_df_j.obs['LabelNew'] = spatial_df_j.obs['LabelNew'].fillna("Normal")

        # Create coordinate tuples for hexagonal grid navigation
        spatial_df_j.obs['xy'] = list(zip(spatial_df_j.obs['array_col'], spatial_df_j.obs['array_row']))
        bdy_obs = spatial_df_j.obs[spatial_df_j.obs['LabelNew'] == 'Bdy'].iloc[:, -4:]

        for level in range(1, levels + 1):
            for i in range(len(bdy_obs['xy'])):
                xy = bdy_obs['xy'].iloc[i]
                edges = BoundarySP_utility.hexagon_edges(x=int(xy[0]), y=int(xy[1]), n=level)
                unique_edges = list(
                    set(tuple(edge) for edge in (edge for edge_list in edges for edge in edge_list))
                )
                bdy_nbrs_barcode = []

                # Match neighbor barcodes based on array coordinate types
                if (spatial_df_j.obs['array_col'].dtype == 'int64') and \
                   (spatial_df_j.obs['array_row'].dtype == 'int64'):
                    for edge in unique_edges:
                        barcode_edge = spatial_df_j.obs['barcode'][
                            (spatial_df_j.obs['array_col'] == int(edge[0])) &
                            (spatial_df_j.obs['array_row'] == int(edge[1]))
                        ]
                        if not barcode_edge.empty:
                            bdy_nbrs_barcode.append(barcode_edge.values[0])
                elif (type(spatial_df_j.obs['array_col'].iloc[0]) == str) and \
                     (type(spatial_df_j.obs['array_row'].iloc[0]) == str):
                    for edge in unique_edges:
                        barcode_edge = spatial_df_j.obs['barcode'][
                            (spatial_df_j.obs['array_col'] == str(edge[0])) &
                            (spatial_df_j.obs['array_row'] == str(edge[1]))
                        ]
                        if not barcode_edge.empty:
                            bdy_nbrs_barcode.append(barcode_edge.values[0])

                spatial_df_j.obs['LabelOld'] = spatial_df_j.obs['LabelNew']
                spatial_df_j.obs['LabelNew'] = spatial_df_j.obs['LabelNew'].astype(str)

                mask = spatial_df_j.obs['barcode'].isin(bdy_nbrs_barcode)
                spatial_df_j.obs.loc[mask, 'LabelNew'] = spatial_df_j.obs.loc[
                    mask, 'LabelNew'
                ].replace({"Mal": f'Mal_{level}00', "Normal": f'Normal_{level}00'})

        return spatial_df_j