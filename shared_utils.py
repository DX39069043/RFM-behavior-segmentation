"""
shared_utils.py — 用户价值分层与精准营销模型共享指标计算模块

标准化策略（v2.0）：
  1. 对所有原始连续指标做 log1p（对数压缩，缓解长尾偏态）
  2. 再对压缩后的值做 StandardScaler（Z-score 标准化）

所有函数只要求传入「语义一致的列名」，不关心数据源头。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import display, HTML


# ═══════════════════════════════════════════════════
# 1. RFM 标准化得分
# ═══════════════════════════════════════════════════

def compute_rfm_scores(
    df: pd.DataFrame,
    r_raw_col: str,
    f_raw_col: str,
    m_raw_col: str,
    r_out: str = 'R_Score',
    f_out: str = 'F_Score',
    m_out: str = 'M_Score'
) -> pd.DataFrame:
    """原始值 → log1p 压缩 → StandardScaler(Z-score)

    R 特殊处理：log1p 后取负再标准化，使「天数大 → 得分低」。
    """
    r_log = np.log1p(df[r_raw_col].clip(lower=0))
    df[r_out] = StandardScaler().fit_transform((-r_log).values.reshape(-1, 1))
    f_log = np.log1p(df[f_raw_col].clip(lower=0))
    df[f_out] = StandardScaler().fit_transform(f_log.values.reshape(-1, 1))
    m_log = np.log1p(df[m_raw_col].clip(lower=0))
    df[m_out] = StandardScaler().fit_transform(m_log.values.reshape(-1, 1))
    return df


# ═══════════════════════════════════════════════════
# 2. 探索度 E_Score & 摩擦力 Friction
# ═══════════════════════════════════════════════════

def compute_engagement_metrics(
    df: pd.DataFrame,
    pages_col: str,
    time_col: str,
    freq_col: str,
    e_out: str = 'E_Score',
    friction_out: str = 'Friction'
) -> pd.DataFrame:
    """原始值 → log1p → StandardScaler → 加权平均

    E_Score = 0.5 * pages_z + 0.5 * time_z
    Friction = pages / (freq + 1)    ← 比率型，不做变换
    """
    pages_log = np.log1p(df[pages_col].clip(lower=0))
    pages_z = StandardScaler().fit_transform(pages_log.values.reshape(-1, 1)).flatten()
    time_log = np.log1p(df[time_col].clip(lower=0))
    time_z  = StandardScaler().fit_transform(time_log.values.reshape(-1, 1)).flatten()
    df[e_out] = 0.5 * pages_z + 0.5 * time_z
    df[friction_out] = df[pages_col] / (df[freq_col] + 1)
    return df


# ═══════════════════════════════════════════════════
# 3. 一维 GMM 阈值
# ═══════════════════════════════════════════════════

def calculate_1d_gmm_threshold(series: pd.Series, verbose: bool = True) -> float:
    """用 2 个高斯分布（GMM）拟合单维数据，取两簇均值的中点作为阈值。"""
    X = series.fillna(0).values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    means = gmm.means_.flatten()
    bound = (sorted(means)[0] + sorted(means)[1]) / 2
    if verbose:
        labels = gmm.predict(X)
        sil = silhouette_score(X, labels)
        pct_below = (series < bound).mean() * 100
        print(f"  silhouette={sil:.3f} | 阈值={bound:.3f} | 低于阈值={pct_below:.0f}%")
    return bound


# ═══════════════════════════════════════════════════
# 4. 3D 聚类版用户分层
# ═══════════════════════════════════════════════════

_RFM_SIGN_MAP = {
    (1, 1, 1): "重要价值用户",
    (1, 0, 1): "重要发展用户",
    (0, 1, 1): "重要保持用户",
    (0, 0, 1): "重要挽留用户",
    (1, 1, 0): "一般价值用户",
    (1, 0, 0): "一般发展用户",
    (0, 1, 0): "一般保持用户",
    (0, 0, 0): "低价值用户",
}


def _centroid_to_sign_pattern(centroid: np.ndarray) -> tuple:
    return tuple(1 if comp > 0 else 0 for comp in centroid)


def _label_map_report(centroids: np.ndarray) -> dict:
    """将 K-Means 簇形心映射到 8 个 RFM 标签。"""
    n = len(centroids)
    used_patterns = set()
    label_map = {}
    for i in range(n):
        pattern = _centroid_to_sign_pattern(centroids[i])
        label = _RFM_SIGN_MAP.get(pattern)
        if label and pattern not in used_patterns:
            label_map[i] = label
            used_patterns.add(pattern)
    remaining = [i for i in range(n) if i not in label_map]
    unassigned = [lbl for pat, lbl in _RFM_SIGN_MAP.items() if lbl not in label_map.values()]
    remaining.sort(key=lambda i: centroids[i].sum(), reverse=True)
    for i, lbl in zip(remaining, unassigned):
        label_map[i] = lbl
    return label_map


def classify_users_rfm_3d(
    df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42
) -> tuple:
    """3D 聚类版用户分层（默认 k=8）。

    原理：对 (R_Score, F_Score, M_Score) 做 K-Means 聚类，
    按每个簇形心的正负符号模式分配传统 RFM 标签（2³ = 8 类），
    再按逐簇 E_Score / Friction 阈值做精细化裂变。
    """
    # ── 1. 聚类诊断 ──
    print(f"\n=== RFM 3D 聚类 (n_clusters={n_clusters}) ===")
    print("Silhouette 诊断:")
    for k in range(3, min(11, len(df) // 50 + 3)):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(
            df[['R_Score', 'F_Score', 'M_Score']].values
        )
        sil = silhouette_score(df[['R_Score', 'F_Score', 'M_Score']].values, km.labels_)
        marker = " ← 当前选择" if k == n_clusters else ""
        print(f"  k={k}: silhouette={sil:.4f}{marker}")

    # ── 2. 执行聚类 ──
    rfm_3d = df[['R_Score', 'F_Score', 'M_Score']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(rfm_3d)
    df['Cluster_ID'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    label_map = _label_map_report(centroids)

    print("\n簇形心 → RFM 标签映射:")
    for cid in sorted(label_map.keys()):
        c = centroids[cid]
        pattern = _centroid_to_sign_pattern(c)
        arrow = ''.join('↑' if p else '↓' for p in pattern)
        print(f"  簇{cid}: R={c[0]:+.3f}  F={c[1]:+.3f}  M={c[2]:+.3f}  ({arrow}) → {label_map[cid]}")

    df['Base_Segment'] = df['Cluster_ID'].map(label_map)

    # ── 3. 逐簇 GMM 阈值 ──
    print("\n=== 逐簇 GMM 阈值 (E_Score / Friction) ===")
    cluster_thresholds = {}
    for cid in sorted(df['Cluster_ID'].unique()):
        mask = df['Cluster_ID'] == cid
        cluster_label = label_map[cid]
        n_users = mask.sum()
        needs_refine = cluster_label in ["重要价值用户", "重要发展用户",
                                         "重要保持用户", "重要挽留用户", "低价值用户"]
        if n_users < 20 or not needs_refine:
            e_b = calculate_1d_gmm_threshold(df.loc[mask, 'E_Score'], verbose=False)
            f_b = calculate_1d_gmm_threshold(df.loc[mask, 'Friction'], verbose=False)
        else:
            print(f"  簇{cid} ({cluster_label}, {n_users}人):")
            e_b = calculate_1d_gmm_threshold(df.loc[mask, 'E_Score'])
            f_b = calculate_1d_gmm_threshold(df.loc[mask, 'Friction'])
        cluster_thresholds[cid] = (e_b, f_b)

    # ── 4. 精细化裂变 ──
    def _refine_row(row):
        base = row['Base_Segment']
        e = row['E_Score']
        friction = row['Friction']
        final = base
        e_b, f_b = cluster_thresholds.get(row['Cluster_ID'], (0, 0))
        high_m = ["重要价值用户", "重要发展用户", "重要保持用户", "重要挽留用户"]
        if base in high_m:
            if friction >= f_b:
                final = "高危异动VIP"
            elif e >= e_b and friction < f_b:
                final = "深度互动用户"
            elif e < e_b and friction < f_b:
                final = "精准直购用户"
        elif base == "低价值用户" and e >= e_b and friction >= f_b:
            final = "高潜力用户"
        return pd.Series([base, final])

    df[['Base_Segment', 'User_Segment']] = df.apply(_refine_row, axis=1)

    # ── 5. 统计 ──
    stats = df.groupby(['Base_Segment', 'User_Segment']).agg(
        用户数=('user_id', 'count'),
        平均消费=('Total_Spending', 'mean'),
        平均购买频率=('Purchase_Frequency', 'mean'),
        平均意向分=('E_Score', 'mean'),
        平均摩擦力系数=('Friction', 'mean')
    ).round(2)
    stats['占比(%)'] = (stats['用户数'] / len(df) * 100).round(1)
    stats = stats.sort_values(['Base_Segment', '用户数'], ascending=[True, False])
    print("\n用户分层统计 (3D 聚类版):")
    display(HTML(stats.to_html()))
    return df, stats
