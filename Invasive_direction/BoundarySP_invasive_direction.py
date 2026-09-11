import os
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import diopy
import commot as ct
import scipy.ndimage as ndi
from sklearn.preprocessing import normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import FancyArrowPatch

def LR_score(path,sample,distance,database="CellPhoneDB_v4.0"):
    adata = sc.read_h5ad(path+sample+".h5ad")
    LR_data = ct.pp.ligand_receptor_database(species='human', 
                                                 signaling_type='Secreted Signaling', 
                                                 database='CellPhoneDB_v4.0')
    ct.tl.spatial_communication(adata,
                                database_name='CellPhoneDB_v4.0', 
                                df_ligrec=LR_data, 
                                dis_thr=distance, 
                                heteromeric=True, 
                                pathway_sum=True)
    return(adata)
    
def input_group(group_path,sample,adata):
    group_meta = pd.read_csv(group_path+sample+".csv")
    if '_' not in str(group_meta['barcode']) and '_' not in str(adata.obs['barcode']):
        pass
    elif '_' not in str(group_meta['barcode']) and '_' in str(adata.obs['barcode']):
        group_meta['barcode'] = group_meta['barcode'] + '_' + sample
    elif '_' in str(group_meta['barcode']) and '_' not in str(adata.obs['barcode']):
        adata.obs['barcode'] = group_meta['barcode'] + '_' + sample
    else:
        pass
    group_meta.set_index("barcode",inplace=True)
    adata.obs= adata.obs.join(group_meta,how="inner")
    return(adata)


def vector_filed(adata,database="CellPhoneDB_v4.0",lr_pair=['TGFB1','TGFBR3']):
    lr_pair = lr_pair
    obsp_names = database+'-'+lr_pair[0]+'-'+lr_pair[1]

    # 导入信号矩阵
    S = adata.obsp['commot-'+obsp_names]    
    results = []  
    sigma = 50  
    S_lil = S.tolil()
    S_lil.max_col = np.full(S_lil.shape[0], np.nan, dtype=float)
    S_lil.max_val = np.full(S_lil.shape[0], np.nan, dtype=float)
    pos = adata.obsm['spatial']
    for i in range(S_lil.shape[0]):
        ## 计算邻居
        if adata.obs.iloc[i]["group"] not in ["Mac_02_APOE"]:
            continue
        neighbor_idx = S_lil.rows[i]
        fib_mask = [adata.obs.iloc[j]['group'] == "Fib_03_CXCL12" for j in neighbor_idx] # 不能再是Fib
        fib_idx = np.array(neighbor_idx)[fib_mask]
        
        if len(fib_idx) == 0:
            continue
        data = np.array(S_lil.data[i], float)[fib_mask]  # 信号强度S_ij
        positions = pos[fib_idx] - pos[i] 
        
        ## 归一化
        norm_directions = normalize(positions, norm='l2', axis=1)
        
        # 信号衰减
        distances = np.linalg.norm(positions, axis=1)
        decay_factors = np.exp(-distances**2 / (2 * sigma**2))  # 高斯衰减
        
        # 加权求和
        weighted_sum = np.sum(data[:, None] * decay_factors[:, None] * norm_directions, axis=0)
        
        # 归一化
        if np.linalg.norm(weighted_sum) > 0:
            V_i = weighted_sum / np.linalg.norm(weighted_sum)
        else:
            V_i = weighted_sum  # 零向量
        
        results.append({
            'mac_index': i,
            'fib_indices': fib_idx,
            'vector': V_i,
            'signal_strength': np.sum(data * decay_factors)
    })
    return(results)

def plot_vector_filed(results,adata,line_size=25,point_size=15,figsize=[8,8]):
    import matplotlib.pyplot as plt
    import numpy as np

    """
    巨噬细胞向成纤维细胞的向量场
    Par:
        results:'start_pos': 巨噬细胞起点坐标 (x, y)
            'vector': 方向向量 (dx, dy)
            'signal_strength': 信号强度值
        adata: 原始空转数据输入
        figsize:导出图像数据
        line_size:箭头长度。int。数值越大长度越小。
    
    """

    # 数据输入
    pos = adata.obsm['spatial'] 
    groups = adata.obs['group'].unique()

    mac_positions = pos[[item['mac_index'] for item in results]]  # 起点坐标 
    vectors = np.array([item['vector'] for item in results])      # 方向向量 
    strength = [item['signal_strength'] for item in results]      # 信号强度
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    # 画图
    plt.figure(figsize=figsize)
    for i, group_name in enumerate(groups):
        group_data = adata.obs[adata.obs['group'] == group_name]
        plt.scatter(
            x=group_data['x'], 
            y=group_data['y'],
            color=colors[i],
            label=group_name,
            s=point_size,  
            alpha=0.8
        )
    plt.quiver(
        mac_positions[:, 0], mac_positions[:, 1],   # 起点 (x, y)
        vectors[:, 0], vectors[:, 1],               # 向量分量 (U, V)
        strength,                                   # 用颜色映射信号强度
        scale=line_size,                                   # 箭头长度缩放因子
        cmap='viridis',                             # 颜色方案
        width=0.006,                                # 箭头宽度
        headwidth=3                                 # 箭头头部宽度
    )
    plt.colorbar(label='Signal Strength')
    plt.title("Macrophage Signaling Vectors to Fibroblasts")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


def calculate_divergence_field_paired(results, adata, grid_res=150, sigma=3.0):
    """
    基于 Sender(Mac) - Receiver(Fib) 对偶关系计算真实的信号源汇场与矢量场
    
    Parameters:
    -----------
    results: vector_filed() 返回的列表
    adata: AnnData 对象
    grid_res: 二维网格分辨率
    sigma: 高斯平滑核（单位为 grid 像素），控制空间扩散半径
    """
    pos = adata.obsm['spatial']
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()

    # 1. 创建二维网格及物理坐标映射
    grid_x, grid_y = np.mgrid[
        x_min : x_max : complex(0, grid_res),
        y_min : y_max : complex(0, grid_res)
    ]
    
    # 构建离散栅格阵列
    source_grid = np.zeros((grid_res, grid_res)) # 记录 Mac (Source, +)
    sink_grid = np.zeros((grid_res, grid_res))   # 记录 Fib (Sink, -)
    U_grid = np.zeros((grid_res, grid_res))      # 矢量 X 分量
    V_grid = np.zeros((grid_res, grid_res))      # 矢量 Y 分量

    # 辅助函数：将真实坐标 (x, y) 转换为网格索引 (ix, iy)
    def pos_to_grid(x, y):
        ix = int(np.clip((x - x_min) / (x_max - x_min) * (grid_res - 1), 0, grid_res - 1))
        iy = int(np.clip((y - y_min) / (y_max - y_min) * (grid_res - 1), 0, grid_res - 1))
        return ix, iy

    # 2. 遍历每一个通讯事件，建立 Mac -> Fib 的精准投射
    for item in results:
        mac_idx = item['mac_index']
        fib_indices = item['fib_indices']
        mac_pos = pos[mac_idx]
        
        mx, my = pos_to_grid(mac_pos[0], mac_pos[1])
        
        # 每一个 Mac 节点的发射矢量与总强度
        m_vec = item['vector']
        m_strength = item['signal_strength']
        
        # 累加 Mac 的源强度 (+Value)
        source_grid[mx, my] += m_strength
        
        # 累加矢量场
        U_grid[mx, my] += m_vec[0] * m_strength
        V_grid[mx, my] += m_vec[1] * m_strength
        
        # 将接收信号精准分配给对应的每一个 Fib Spot
        # 这里的信号强度按照距离分配给各个 Fib
        fib_positions = pos[fib_indices]
        dists = np.linalg.norm(fib_positions - mac_pos, axis=1)
        # 防止除零
        weights = 1.0 / (dists + 1e-5)
        weights /= weights.sum()  # 归一化权重
        
        for f_pos, w in zip(fib_positions, weights):
            fx, fy = pos_to_grid(f_pos[0], f_pos[1])
            # 累加 Fib 的汇强度 (-Value)
            sink_grid[fx, fy] -= m_strength * w

    # 3. 真实源汇场 = Source (+) + Sink (-)
    net_field = source_grid + sink_grid

    # 4. 高斯平滑（连续化，将离散 Spot 转换为连续微环境场）
    net_field_smooth = ndi.gaussian_filter(net_field, sigma=sigma)
    U_smooth = ndi.gaussian_filter(U_grid, sigma=sigma)
    V_smooth = ndi.gaussian_filter(V_grid, sigma=sigma)

    return grid_x, grid_y, net_field_smooth, U_smooth, V_smooth




def plot_flow(adata, grid_x, grid_y, net_field, U, V, figsize=(10, 8)):
    plt.figure(figsize=figsize)

    pos = adata.obsm['spatial']
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    vmax = np.max(np.abs(net_field))
    if vmax == 0:
        vmax = 1.0

    field = plt.pcolormesh(
        grid_x,
        grid_y,
        net_field,
        cmap='bwr',  # Blue - White - Red
        vmin=-vmax,
        vmax=vmax,
        shading='gouraud',
        alpha=0.8,
        zorder=0,
    )

    # 1. 绘制背景点：突出 Mac 和 Fib 细胞的位置
    labels = adata.obs['Label'].astype(str)

    mask_bdy = labels == 'Bdy'
    plt.scatter(
        pos[mask_bdy, 0],
        pos[mask_bdy, 1],
        c='#FD8C67',
        s=20,
        alpha=1,
        label='Bdy (Boundary)',
        zorder=1,
        rasterized=False
    )

    mask_mal = labels == 'Mal'
    plt.scatter(
        pos[mask_mal, 0],
        pos[mask_mal, 1],
        c='#F17172',
        s=5,
        alpha=0.5,
        label='Mal (Sender)',
        zorder=1,
        rasterized=False
    )

    mask_normal = labels == 'Normal'
    plt.scatter(
        pos[mask_normal, 0],
        pos[mask_normal, 1],
        c='#DBEFCB',
        s=10,
        alpha=0.5,
        label='Normal (Receiver)',
        zorder=1,
        rasterized=False
    )

    # 2. 过滤微弱信号噪声
    speed = np.sqrt(U**2 + V**2)
    threshold = np.percentile(speed, 50)  # 过滤后50%微弱信号噪点
    U_norm = np.where(speed > threshold, U, 0)
    V_norm = np.where(speed > threshold, V, 0)

    # 3. 生成流线，关闭内置箭头
    strm = plt.streamplot(
        grid_x[:, 0],
        grid_y[0, :],
        U_norm.T,
        V_norm.T,
        color='k',
        linewidth=1.2,
        density=1,  # 控制流线密度
        arrowsize=1,  # 关闭自带内置箭头
        zorder=3,
    )

    ax = plt.gca()

    # 提取多条小线段
    lines = strm.lines
    if hasattr(lines, 'get_segments'):
        segments = lines.get_segments()
    else:
        segments = lines

    # ================= 核心修复：适配大坐标尺度的拼接逻辑 =================
    merged_trajectories = []
    current_traj = []

    # 动态计算容差：取坐标跨度的 0.2%，自动适配任何大/小坐标系
    x_span = grid_x.max() - grid_x.min()
    tolerance = x_span * 0.002

    for seg in segments:
        if len(seg) < 2:
            continue
        if not current_traj:
            current_traj.extend(seg)
        else:
            # 判断当前片段起点与上一片段终点距离，低于动态容差则顺利接轨
            if (
                np.linalg.norm(np.array(current_traj[-1]) - np.array(seg[0]))
                < tolerance
            ):
                current_traj.append(seg[1])
            else:
                # 确定断开才保存旧长线，开启新长线
                merged_trajectories.append(current_traj)
                current_traj = list(seg)

    if current_traj:
        merged_trajectories.append(current_traj)

    # ================= 绘制每条整线末端的唯一大箭头 =================
    for traj in merged_trajectories:
        if len(traj) < 2:
            continue

        p1 = traj[-2]
        p2 = traj[-1]

        # 计算尾端向量长度，防止重合点
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        if dist < 1e-5:
            continue

        arrow = FancyArrowPatch(
            posA=p1,
            posB=p2,
            arrowstyle='->',
            mutation_scale=20,  # 箭头头部尺寸
            color='black',
            linewidth=2,
            zorder=4,
        )
        ax.add_patch(arrow)

    # 4. 图例与美化
    cbar = plt.colorbar(field, pad=0.02)
    cbar.set_label(
        'Signaling Intensity\n(Red: Mac Sender (+)  <--->  Blue: Fib Receiver'
        ' (-))',
        fontsize=10,
    )

    plt.title(
        'TGFB1-TGFBR3 Signaling Flow Field: Mac_02_APOE → Fib_03_CXCL12',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlim(x_min , x_max )
    plt.ylim(y_min , y_max)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right', frameon=True)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.tight_layout()


def plot_edge(adata, grid_x, grid_y, figsize=(10, 8)):
    plt.figure(figsize=figsize)

    pos = adata.obsm['spatial']

    # 1. 绘制背景点：突出 Mac 和 Fib 细胞的位置
    labels = adata.obs['Label'].astype(str)

    mask_bdy = labels == 'Bdy'
    plt.scatter(
        pos[mask_bdy, 0],
        pos[mask_bdy, 1],
        c='#FD8C67',
        s=20,
        alpha=1,
        label='Bdy (Boundary)',
        zorder=1,
        rasterized=False
    )

    mask_mal = labels == 'Mal'
    plt.scatter(
        pos[mask_mal, 0],
        pos[mask_mal, 1],
        c='#F17172',
        s=5,
        alpha=0.5,
        label='Mal (Sender)',
        zorder=1,
        rasterized=False
    )

    mask_normal = labels == 'Normal'
    plt.scatter(
        pos[mask_normal, 0],
        pos[mask_normal, 1],
        c='#DBEFCB',
        s=10,
        alpha=0.5,
        label='Normal (Receiver)',
        zorder=1,
        rasterized=False
    )

    plt.xlim(grid_x.min(), grid_x.max())
    plt.ylim(grid_y.min(),grid_y.max())
    plt.title(
        'TGFB1-TGFBR3 Signaling Flow Field: Mac_02_APOE → Fib_03_CXCL12',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right', frameon=True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def plot_cell(adata, grid_x, grid_y, net_field, U, V, figsize=(10, 8)):
    plt.figure(figsize=figsize)

    pos = adata.obsm['spatial']

    # 1. 绘制背景点：突出 Mac 和 Fib 细胞的位置
    is_mac = adata.obs['group'] == 'Mac_02_APOE'
    is_fib = adata.obs['group'] == 'Fib_03_CXCL12'

    # 其它细胞背景 (浅灰)
    plt.scatter(
        pos[~is_mac & ~is_fib, 0],
        pos[~is_mac & ~is_fib, 1],
        c='#E0E0E0',
        s=6,
        alpha=0.3,
        label='Other cells',
    )

    # 2. 绘制源-汇场 (Diverging Colorbar)
    vmax = np.max(np.abs(net_field))
    if vmax == 0:
        vmax = 1.0

    field = plt.pcolormesh(
        grid_x,
        grid_y,
        net_field,
        cmap='bwr',  # Blue - White - Red
        vmin=-vmax,
        vmax=vmax,
        shading='gouraud',
        alpha=0.65,
        zorder=2,
    )

    # 3. 叠加从 Mac 指向 Fib 的信号流线 (Streamlines 并在整条流线末端绘制唯一大箭头)
    speed = np.sqrt(U**2 + V**2)
    threshold = np.percentile(speed, 50)  # 过滤后50%微弱信号噪点
    U_norm = np.where(speed > threshold, U, 0)
    V_norm = np.where(speed > threshold, V, 0)

    # 生成流线，关闭默认箭头
    strm = plt.streamplot(
        grid_x[:, 0],
        grid_y[0, :],
        U_norm.T,
        V_norm.T,
        color='k',
        linewidth=1.0,
        density=1,  # 稍微降低密度，使图像更清爽
        arrowsize=0,  # 关闭自带内置箭头
        zorder=3,
    )

    ax = plt.gca()

    # 提取多条小线段
    lines = strm.lines
    if hasattr(lines, 'get_segments'):
        segments = lines.get_segments()
    else:
        segments = lines

    # ================= 核心修复：连接片段为完整长流线 =================
    merged_trajectories = []
    current_traj = []

    for seg in segments:
        if len(seg) < 2:
            continue
        if not current_traj:
            current_traj.extend(seg)
        else:
            # 判断当前片段起点是否与上一片段终点接轨
            if np.linalg.norm(current_traj[-1] - seg[0]) < 1e-3:
                current_traj.append(seg[1])
            else:
                # 不连续则保存上一条完整轨迹，开辟新轨迹
                merged_trajectories.append(current_traj)
                current_traj = list(seg)

    if current_traj:
        merged_trajectories.append(current_traj)

    # ================= 绘制每条整线末端的唯一大箭头 =================
    for traj in merged_trajectories:
        if len(traj) < 2:
            continue

        p1 = traj[-2]
        p2 = traj[-1]

        # 计算尾端向量长度，避免零长度点
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        if dist < 1e-4:
            continue

        arrow = FancyArrowPatch(
            posA=p1,
            posB=p2,
            arrowstyle='->',
            mutation_scale=18,  # 放大箭头头部的尺寸 (原 10 -> 现 18)
            color='black',
            linewidth=0.8,
            zorder=4,
        )
        ax.add_patch(arrow)

    # 4. 高亮真实 Mac 和 Fib 节点
    plt.scatter(
        pos[is_mac, 0],
        pos[is_mac, 1],
        c='red',
        s=12,
        label='Mac_02_APOE (Sender)',
        zorder=5,
    )
    plt.scatter(
        pos[is_fib, 0],
        pos[is_fib, 1],
        c='blue',
        s=12,
        label='Fib_03_CXCL12 (Receiver)',
        zorder=5,
    )

    # 图例与美化
    cbar = plt.colorbar(field, pad=0.02)
    cbar.set_label(
        'Signaling Intensity\n(Red: Mac Sender (+)  <--->  Blue: Fib Receiver (-))',
        fontsize=10,
    )

    plt.title(
        'TGFB1-TGFBR3 Signaling Flow Field: Mac_02_APOE → Fib_03_CXCL12',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right', frameon=True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()
