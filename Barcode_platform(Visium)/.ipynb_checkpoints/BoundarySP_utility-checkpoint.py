import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def compute_interspot_distances(position, scale_factor=1.05):
    """Calculate average inter-spot distances and spot radius using linear regression
    between spatial array coordinates and pixel coordinates.
    """
    required_cols = {"array_row", "array_col", "imagerow", "imagecol"}
    assert required_cols.issubset(
        position.columns
    ), "Some required columns are missing."

    dists = {}
    # Linear regression between array columns and image columns
    xdist_reg = linregress(position["array_col"], position["imagecol"])
    # Linear regression between array rows and image rows
    ydist_reg = linregress(position["array_row"], position["imagerow"])
    # Extract slopes
    dists["xdist"] = xdist_reg.slope
    dists["ydist"] = ydist_reg.slope
    # Calculate radius
    dists["radius"] = (
        abs(dists["xdist"]) + abs(dists["ydist"])
    ) * scale_factor
    return dists


def find_neighbors(position, radius, n=1, method="euclidean"):
    """Identify spatial neighbors for each spot within a scaled distance threshold."""
    if method == "manhattan":
        distance_metric = "cityblock"
    else:
        distance_metric = method
    pos_matrix = position[["imagecol", "imagerow"]].values
    dist_matrix = squareform(pdist(pos_matrix, metric=distance_metric))
    neighbors_dict = {index: [] for index in position.index}
    for i, (row_index, row_dist) in enumerate(zip(position.index, dist_matrix)):
        # Identify neighbor indices excluding self
        neighbor_indices = np.where(row_dist <= n * radius + n - 1)[0]
        if i in neighbor_indices:
            neighbor_indices = np.delete(
                neighbor_indices, np.where(neighbor_indices == i)[0]
            )
        # Store neighbor IDs
        neighbors_dict[row_index] = [
            position.index[j] for j in neighbor_indices
        ]

    return neighbors_dict


def find_mal_cells(sub_mal_cell_ids, sub_ci_mal, ci_normal, umap_embeddings):
    """Filter malignant cells based on relative Euclidean distances in UMAP space."""
    sub_mal_cells = []
    for cell_id in sub_mal_cell_ids:
        pos = umap_embeddings.loc[cell_id]
        rt = np.sqrt(np.sum((pos - sub_ci_mal) ** 2))
        rn = np.sqrt(np.sum((pos - ci_normal) ** 2))
        if rt < 1 / 3 * rn:
            sub_mal_cells.append(cell_id)
    return sub_mal_cells


def nbrs(df_j, MalCellIDAdd, CellIDRaw):
    """Retrieve adjacent neighbors of malignant cells, filtering out already categorized cells."""
    nbrs_of_Mal = {}
    for id in MalCellIDAdd:
        nbs = df_j.get(id, [])
        nbs = [n for n in nbs if n not in CellIDRaw]
        nbrs_of_Mal[id] = nbs
    return nbrs_of_Mal


def calculate_location(pos, sub_CiMal, ci_normal):
    """Determine cell state ('Mal' vs 'other') based on distance to malignant and normal centers."""
    rt = np.linalg.norm(np.array(pos) - np.array(sub_CiMal))
    rn = np.linalg.norm(np.array(pos) - np.array(ci_normal))
    return "Mal" if rt < 1 / 3 * rn else "other"


def init_tumor_anchor(adata, mal_ids, emb_key="emb_pca"):
    """Compute centroid and radius tolerance of tumor core in the embedding space."""
    X = adata.obsm[emb_key]
    spot_indices = pd.Series(np.arange(len(adata)), index=adata.obs.index)
    mal_idx = spot_indices.loc[
        spot_indices.index.intersection(mal_ids)
    ].values
    # Extract embeddings for tumor core
    mal_embs = X[mal_idx]
    # Calculate centroid
    tumor_centroid = np.mean(mal_embs, axis=0)
    # Calculate standard radius (95th percentile distance to centroid)
    distances_to_center = np.linalg.norm(mal_embs - tumor_centroid, axis=1)
    tumor_radius = np.percentile(distances_to_center, 95)
    anchor_dict = {"centroid": tumor_centroid, "radius": tumor_radius}
    return anchor_dict


def calculate_location_by_emb_deviation(
    anchor_dict,
    target_id,
    df_j,
    adata,
    emb_key="emb_pca",
    threshold_multiplier=0.5,
):
    """Classify spot location by embedding deviation from the tumor centroid."""
    # 1. Collect 1-hop and 2-hop neighbors
    hop1_neighbors = list(df_j.get(target_id, []))
    hop2_neighbors = []
    for n1 in hop1_neighbors:
        if n1 in df_j:
            hop2_neighbors.extend(df_j[n1])

    all_related_spots = list(
        set([target_id] + hop1_neighbors + hop2_neighbors)
    )
    valid_spots = [s for s in all_related_spots if s in adata.obs.index]
    if not valid_spots:
        valid_spots = [target_id]

    # 2. Extract embedding features
    idxs = [adata.obs.index.get_loc(spot) for spot in valid_spots]
    embs = adata.obsm[emb_key][idxs]
    # 3. Compute mean embedding for local neighborhood
    mean_emb = np.mean(embs, axis=0)
    # 4. Calculate Euclidean distance to tumor centroid
    dist_to_tumor = np.linalg.norm(mean_emb - anchor_dict["centroid"])
    # 5. Check tolerance boundary
    allowed_radius = anchor_dict["radius"] * threshold_multiplier
    if dist_to_tumor > allowed_radius:
        return "Bdy"
    else:
        return "Mal"


def ClusterUpdate(
    MalCellIDN=None,
    position=None,
    df_j=None,
    UMAPembeddings=None,
    NormalCellID=None,
    BdyCellID=None,
    MalCellID=None,
    x=None,
):
    """Update cluster assignments based on spatial neighborhood distances in UMAP space."""
    P = []
    for i in MalCellIDN:
        ncellID = pd.Series(df_j[i])
        cpos = UMAPembeddings.loc[i, :]

        ncellID_set = set(ncellID)
        valid_ncellID = ncellID_set.intersection(UMAPembeddings.index)
        npos = UMAPembeddings.loc[list(valid_ncellID)]

        nMalID = ncellID[ncellID.isin(MalCellID)].tolist()
        CiMal = pd.DataFrame(
            np.mean(np.vstack((npos.loc[nMalID], cpos)), axis=0)
        ).T
        CiMal.columns = ["x", "y"]
        rMal = [
            np.sqrt(np.sum((UMAPembeddings.loc[id] - CiMal) ** 2))
            for id in nMalID + [i]
        ]
        all_values = [value for series in rMal for value in series.values]
        max_value = max(all_values)

        nBdyID = ncellID[ncellID.isin(BdyCellID)].tolist()
        Neigh_unlabel = ncellID[
            ~ncellID.isin(MalCellID + BdyCellID + NormalCellID)
        ].tolist()

        if len(nBdyID) <= 1:
            p = [
                np.sqrt(np.sum((npos.loc[id] - CiMal) ** 2))
                for id in Neigh_unlabel
            ]
            d = pd.DataFrame({"cellID": Neigh_unlabel, "p1": p, "p2": p})
            d["cluster"] = [
                (
                    f"Mal{x}"
                    if any(
                        d.loc[d["cellID"] == id, "p1"].values[0]
                        <= 0.8 * max_value
                    )
                    else "Bdy"
                )
                for id in Neigh_unlabel
            ]
        else:
            CiBdy = pd.DataFrame(np.mean(npos.loc[nBdyID], axis=0)).T
            rBdy = [
                np.sqrt(np.sum((npos.loc[id] - CiBdy) ** 2)) for id in nBdyID
            ]
            all_rBdy = [value for series in rBdy for value in series.values]
            max_rBdy = max(all_rBdy)
            p1 = [
                np.sqrt(np.sum((npos.loc[id] - CiMal) ** 2))
                for id in Neigh_unlabel
            ]
            p2 = [
                np.sqrt(np.sum((npos.loc[id] - CiBdy) ** 2))
                for id in Neigh_unlabel
            ]
            d = pd.DataFrame({"cellID": Neigh_unlabel, "p1": p1, "p2": p2})
            d["cluster"] = [
                (
                    f"Mal{x}"
                    if all(
                        (
                            d.loc[d["cellID"] == id, "p1"].values[0]
                            <= 0.8 * max_value
                        )
                    )
                    and all(
                        (d.loc[d["cellID"] == id, "p2"].values[0] > max_rBdy)
                    )
                    else "Bdy"
                )
                for id in Neigh_unlabel
            ]

        P.append(d)
    if not P:
        return pd.DataFrame()
    PDF = pd.concat(P, ignore_index=True)
    PDF[["p1_x", "p1_y"]] = PDF["p1"].apply(pd.Series)
    PDF[["p2_x", "p2_y"]] = PDF["p2"].apply(pd.Series)
    PDF.drop(["p1", "p2"], axis=1, inplace=True)

    Cluster = PDF.groupby(["cellID", "cluster"]).size().unstack(fill_value=0)

    Cluster["Location"] = "Bdy"
    if "Mal" in Cluster.columns:
        Cluster["Location"] = Cluster.apply(
            lambda row: f"Mal{x}" if row[f"Mal{x}"] > 0 else "Bdy", axis=1
        )

    Cluster.drop(["Bdy", f"Mal{x}"], axis=1, inplace=True)

    ClusterNew = Cluster.reset_index()
    ClusterNew.columns = ["CellID", "Location"]
    return ClusterNew


def process_clusters(
    TumorST,
    Clustern,
    MalCellIDN,
    NormalCellID,
    BdyCellID,
    df_j,
    MalCellID,
    anchor_dict,
    n=1,
):
    """Iteratively expand and reclassify cluster boundaries using spatial neighbors and embedding anchors."""
    while True:
        if len(list(MalCellIDN)) < 3:
            print(f"Stop at n = {n - 1}")
            print("Stopped at Step 1")
            break
        else:
            nbrs_of_Mal = nbrs(
                df_j=df_j,
                MalCellIDAdd=list(MalCellIDN),
                CellIDRaw=list(MalCellID)
                + list(NormalCellID)
                + list(BdyCellID),
            )

            if (
                len(
                    set(
                        [
                            item
                            for sublist in nbrs_of_Mal.values()
                            for item in sublist
                        ]
                    )
                )
                < 3
            ):
                print(f"Stop at n = {n - 1}")
                print("Stopped at Step 2")
                break
            else:
                cells_to_keep = set()
                for sublist in nbrs_of_Mal.values():
                    cells_to_keep.update(sublist)
                cells_to_keep.update(MalCellID)
                cells_to_keep.update(NormalCellID)
                cells_to_keep.update(BdyCellID)

                TumorSTn = TumorST[list(cells_to_keep)].copy()
                TumorSTn.obs["barcode"] = TumorSTn.obs.index
                Clustern_unique = Clustern.drop_duplicates(
                    subset="CellID", keep="first"
                ).set_index("CellID")
                TumorSTn.obs["Label"] = TumorSTn.obs["barcode"].map(
                    Clustern_unique["Location"]
                )

                if n == 1:
                    TumorSTn.obs["Label"] = pd.Categorical(
                        TumorSTn.obs["Label"],
                        categories=["Normal", "Bdy", "Mal"],
                    )
                else:
                    TumorSTn.obs["Label"] = pd.Categorical(
                        TumorSTn.obs["Label"],
                        categories=["Normal", "Bdy", "Mal"]
                        + [f"Mal{i}" for i in range(1, n)],
                    )

                ClusterAdd = ClusterUpdate_test(
                    x=n,
                    anchor_dict=anchor_dict,
                    df_j=df_j,
                    adata=TumorST,
                    MalCellIDN=list(MalCellIDN),
                    BdyCellID=list(BdyCellID),
                    NormalCellID=list(NormalCellID),
                    MalCellID=list(MalCellID),
                )

                Clustern = pd.concat([Clustern, ClusterAdd])
                TumorSTn.obs["barcode"] = TumorSTn.obs.index
                Clustern_unique = Clustern.drop_duplicates(
                    subset="CellID", keep="first"
                ).set_index("CellID")
                TumorSTn.obs["LabelNew"] = TumorSTn.obs["barcode"].map(
                    Clustern_unique["Location"]
                )
                TumorSTn.obs["LabelNew"] = pd.Categorical(
                    TumorSTn.obs["LabelNew"],
                    categories=["Normal", "Bdy", "Mal"]
                    + [f"Mal{i}" for i in range(1, n + 1)],
                )

                # Update state for next iteration
                MalCellID = TumorSTn.obs_names[
                    TumorSTn.obs["LabelNew"].isin(
                        ["Mal"] + [f"Mal{i}" for i in range(1, n + 1)]
                    )
                ]
                MalCellIDN = TumorSTn.obs_names[
                    TumorSTn.obs["LabelNew"].isin([f"Mal{n}"])
                ]
                BdyCellID = TumorSTn.obs_names[
                    TumorSTn.obs["LabelNew"] == "Bdy"
                ]
                n += 1

        if n > 5:
            print(f"Stop at n = {n - 1}")
            print("Stopped at Step 3")
            return TumorSTn

    return TumorSTn


def calculate_line_points_xiexian(
    start_x, start_y, end_x, end_y, step_x, step_y
):
    """Calculate point coordinates along a diagonal line trajectory."""
    dx = end_x - start_x
    dy = end_y - start_y
    steps = max(abs(dx), abs(dy))
    points = [
        (int(start_x + i * step_x), int(start_y + i * step_y))
        for i in range(steps + 1)
    ]
    return points


def calculate_line_points_zhixian(start_x, start_y, end_x, end_y, step_x):
    """Calculate point coordinates along a straight horizontal line trajectory."""
    dx = end_x - start_x
    steps_x = int(abs(dx) / 2)
    points = [
        (int(start_x + i * step_x), int(start_y)) for i in range(steps_x + 1)
    ]
    return points


def hexagon_edges(x, y, n):
    """Generate discrete boundary line coordinates defining a symmetric hexagonal grid unit."""
    top_left = (x - n, y + n)
    top_right = (x + n, y + n)
    bottom_left = (x - n, y - n)
    bottom_right = (x + n, y - n)
    left = (x - 2 * n, y)
    right = (x + 2 * n, y)

    left_top_line = calculate_line_points_xiexian(
        left[0], left[1], top_left[0], top_left[1], 1, 1
    )
    top_left_to_top_right = calculate_line_points_zhixian(
        top_left[0], top_left[1], top_right[0], top_right[1], 2
    )
    top_right_to_right = calculate_line_points_xiexian(
        top_right[0], top_right[1], right[0], right[1], 1, -1
    )
    right_to_bottom_right = calculate_line_points_xiexian(
        right[0], right[1], bottom_right[0], bottom_right[1], -1, -1
    )
    bottom_right_to_bottom_left = calculate_line_points_zhixian(
        bottom_right[0],
        bottom_right[1],
        bottom_left[0],
        bottom_left[1],
        -2,
    )
    bottom_left_to_left = calculate_line_points_xiexian(
        bottom_left[0], bottom_left[1], left[0], left[1], -1, 1
    )
    return (
        left_top_line,
        top_left_to_top_right,
        top_right_to_right,
        right_to_bottom_right,
        bottom_right_to_bottom_left,
        bottom_left_to_left,
    )


def ClusterUpdate_test(
    anchor_dict,
    MalCellIDN=None,
    df_j=None,
    adata=None,
    NormalCellID=None,
    BdyCellID=None,
    MalCellID=None,
    x=None,
):
    """Test updated cluster mapping using embedding anchor deviation for unlabelled neighbors."""
    P = []
    for i in MalCellIDN:
        ncellID = pd.Series(df_j[i])
        Neigh_unlabel = ncellID[
            ~ncellID.isin(MalCellID + BdyCellID + NormalCellID)
        ].tolist()
        if not Neigh_unlabel:
            continue
        d_list = []
        for id in Neigh_unlabel:
            loc = calculate_location_by_emb_deviation(
                anchor_dict=anchor_dict, target_id=id, df_j=df_j, adata=adata
            )
            cluster_label = f"Mal{x}" if loc == "Mal" else "Bdy"
            d_list.append({"cellID": id, "cluster": cluster_label})
        if d_list:
            P.append(pd.DataFrame(d_list))
    if not P:
        return pd.DataFrame(columns=["CellID", "Location"])
    PDF = pd.concat(P, ignore_index=True)
    Cluster = PDF.groupby(["cellID", "cluster"]).size().unstack(fill_value=0)
    ClusterNew = pd.DataFrame(index=Cluster.index)
    if f"Mal{x}" in Cluster.columns:
        ClusterNew["Location"] = Cluster.apply(
            lambda row: f"Mal{x}" if row.get(f"Mal{x}", 0) > 0 else "Bdy", axis=1
        )
    else:
        ClusterNew["Location"] = "Bdy"
    ClusterNew = ClusterNew.reset_index()
    ClusterNew.columns = ["CellID", "Location"]
    return ClusterNew