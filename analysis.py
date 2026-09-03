"""用户分层与针对性营销（Segments & Targeting）分析函数库。

核心产出：六类可解释用户标签——普通浏览 / 高潜力首购 / 常规已购 / 高价值直购 /
高价值深度互动 / 高价值高摩擦（跨月沉默），支撑对不同人群的差异化营销策略
（潜力培育 / 首购激励与购物车挽回 / 复购唤醒 / 快捷复购 / 会员运营 / 流失召回）。
方法主线：未购用户 → 首购潜力识别（探索度 + 加购未买 + 首购概率分）；
已购用户 → 价值分层与复购概率分（RFM 特征 + 复购基准）。
指标构建 → 特征聚合 → GMM 阈值分层 → 时间外验证检验 → 逻辑回归基准 → 概率分 Top-k 选人 → 名单导出。
数据分析工作流（数据加载、EDA 与可视化）在 main.ipynb 中组织。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import (BOOTSTRAP_N, CV_FOLDS, DEDUPE_EVENTS, LR_MAX_ITER,
                    MONTHS, PANEL_FILE, POOL_SEGMENTS, POOL_TOP_RATIO,
                    RANDOM_STATE, SESSION_GAP_SECONDS)


def fit_engagement_scalers(df: pd.DataFrame) -> dict:
    """拟合 E_Score 三个分量（log1p 后）的 Z 标准化器，供基期冻结复用。

    队列迁移分析中，若每月对 E_Score 重新标准化，"冻结阈值"就只冻结了
    阈值数值、没冻结"尺子"本身——同用户跨月 E_Score 不可严格比较。
    基期拟合一次、后续月 transform 可保证跨月可比。
    """
    # 需要做 Z 标准化的三个探索度原始指标列
    cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']
    # 空字典，最后返回"列名 -> 已拟合好的标准化器"
    scalers_dict = {}
    # 逐个指标单独拟合一个标准化器
    for col in cols:
        # 1. 取出该指标整列数据
        data_series = df[col]
        # 2. 裁剪：把所有小于 0 的值改为 0（确保后续对数变换不会出现负数）
        data_clipped = data_series.clip(lower=0)
        # 3. 对数变换 log(1+x)：压缩大数值，缓解长尾分布对标准差的拉扯
        data_logged = np.log1p(data_clipped)
        # 4. 转成 NumPy 数组并重塑为列向量 (行数, 1)：sklearn 要求二维输入
        data_reshaped = data_logged.to_numpy().reshape(-1, 1)
        # 5. 新建 StandardScaler 并拟合（记录该列的均值与标准差）
        scaler = StandardScaler()
        scaler.fit(data_reshaped)
        # 6. 把拟合好的标准化器存进字典
        scalers_dict[col] = scaler
    # 返回"列名 -> 标准化器"的映射
    return scalers_dict


def compute_engagement_metrics(df: pd.DataFrame, scalers: dict | None = None) -> pd.DataFrame:
    """
    构建探索度与摩擦指标，指标定义保持业务可解释。

    E_Score: 浏览页数、有效停留时长、会话数的对数标准化均值。
    Friction: 加购但未购买的去重商品数（Cart_Products - Purchased_Products，截断≥0），
              度量“加购了却没买”的购买意图受阻（未购人群首购潜力识别用）。
    已购人群的“高摩擦”不在此定义——它指跨月完全沉默（见 flag_buyer_silence）。

    scalers：可选 dict（见 fit_engagement_scalers）。为 None 时当月重新拟合
    （滚动验证等逐月独立建模场景）；传入时用既有标准化器 transform
    （队列迁移分析冻结 E_Score 的标准化参数，保证跨月可比）。
    """

    # 在副本上操作，避免污染调用方的 DataFrame
    out = df.copy()
    # 参与 E_Score 的三个探索度原始指标列
    cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']
    # z：依次存放三个指标标准化后的列（每列长度均为行数）
    z = []
    for col in cols:
        # 1. 取列 → 裁剪负数为 0 → log1p 压缩长尾
        col_clipped = out[col].clip(lower=0)
        col_logged = np.log1p(col_clipped)
        # 2. 转成二维列向量 (行数, 1)：sklearn 要求二维输入
        col_values = col_logged.to_numpy().reshape(-1, 1)
        if scalers is None:
            # 未传标准化器：当月重新拟合标准化（滚动验证等逐月独立建模场景）
            scaler = StandardScaler()
            standardized = scaler.fit_transform(col_values)
        else:
            # 传入基期标准化器：只做 transform（冻结"尺子"，保证跨月可比）
            standardized = scalers[col].transform(col_values)
        # ravel() 把列向量展平成一维，方便最后按行求均值
        z.append(standardized.ravel())
    # E_Score = 三个标准化分量的逐行均值（探索度综合分）
    out['E_Score'] = np.mean(z, axis=0)
    # Friction = 加购去重商品数 - 购买去重商品数（"加购了却没买"的意图受阻），截断 ≥ 0
    friction_raw = out['Cart_Products'] - out['Purchased_Products']
    out['Friction'] = friction_raw.clip(lower=0)
    # Log_Friction：对摩擦计数做 log1p，供 GMM 阈值分层使用
    out['Log_Friction'] = np.log1p(out['Friction'])
    return out


def build_features(events: pd.DataFrame,
                   observation_end: pd.Timestamp,
                   scalers: dict | None = None,
                   return_scalers: bool = False):
    """按会话计算有效停留时长，聚合为用户特征（含加购/购买的去重商品数）。

    scalers / return_scalers：E_Score 标准化器的冻结复用（队列迁移分析用）。
    基期调用传 return_scalers=True 取回标准化器，后续月传 scalers=基期标准化器，
    保证 E_Score 的 Z 标准化参数跨月不变（否则"冻结阈值"只冻结了阈值、
    没冻结"尺子"本身，跨月标签不可比）。默认（两者均不传）行为与旧版一致：
    返回单表、每月重新拟合标准化。
    """
    # ── 第 1 步：事件去重与排序 ──
    if DEDUPE_EVENTS:
        # 配置要求时，先去掉完全重复的事件行
        events = events.drop_duplicates()
    # 按 用户 → 会话 → 时间 排序：让同一会话内的事件在行上相邻
    # （后面才能用相邻行时间差来算会话内有效停留时长）
    events = events.sort_values(['user_id', 'user_session', 'event_time']).copy()

    # ── 第 2 步：向量化计算会话内有效停留时长 active_seconds ──
    # 判断"当前行与上一行是否属于同一个用户"
    same_user = events['user_id'].eq(events['user_id'].shift())
    # 判断"当前行与上一行是否属于同一个会话"
    same_session_row = events['user_session'].eq(events['user_session'].shift())
    # 两者同时成立 → 当前行与上一行是同一会话内的相邻事件
    same_session = same_user & same_session_row
    # 当前行与上一行的事件时间差（秒）；排序后同一会话内的差分才有意义
    time_diff = events['event_time'].diff().dt.total_seconds()
    # 会话内合法时间差范围：(0, SESSION_GAP_SECONDS] 之外一律视为无效
    valid_diff = same_session & time_diff.between(0, SESSION_GAP_SECONDS)
    # 无效位置填 0，首行（无上一行）的 NaN 也补 0 → 会话内有效停留秒数
    in_session_seconds = time_diff.where(valid_diff, 0)
    events['active_seconds'] = in_session_seconds.fillna(0)

    # ── 第 3 步：一次性生成各事件类型的布尔掩码与购买子集 ──
    is_purchase = events['event_type'].eq('purchase')
    is_view = events['event_type'].eq('view')
    is_cart = events['event_type'].eq('cart')
    # 只保留购买事件（后面多次使用，先取出来）
    purchase_ev = events.loc[is_purchase]

    # ── 第 4 步：用户级聚合——购买侧指标 ──
    # 最近购买时间（算沉默天数用）、购买次数、总消费，一次 groupby 完成
    purchases = purchase_ev.groupby('user_id').agg(
        Last_Purchase=('event_time', 'max'), Purchase_Frequency=('event_type', 'size'),
        Total_Spending=('price', 'sum'))

    # ── 第 5 步：用户级聚合——行为侧指标 ──
    # 5a. 浏览页数：每个用户产生的 view 事件条数
    view_events = events.loc[is_view]
    view_counts = view_events.groupby('user_id').size()
    views = view_counts.rename('Pages_Viewed')
    # 5b. 会话数：每个用户出现过的不同 user_session 个数
    session_counts = events.groupby('user_id')['user_session'].nunique()
    sessions = session_counts.rename('Session_Count')
    # 5c. 有效停留时长：每个用户的 active_seconds 求和
    duration_sums = events.groupby('user_id')['active_seconds'].sum()
    duration = duration_sums.rename('Estimated_Time')
    # 5d. 加购去重商品数：cart 事件里去重后的 product_id 个数
    cart_events = events.loc[is_cart]
    cart_nunique = cart_events.groupby('user_id')['product_id'].nunique()
    cart_products = cart_nunique.rename('Cart_Products')
    # 5e. 购买去重商品数：购买事件里去重后的 product_id 个数
    purchased_nunique = purchase_ev.groupby('user_id')['product_id'].nunique()
    purchased_products = purchased_nunique.rename('Purchased_Products')

    # ── 第 6 步：合并成宽表（一行一个用户）──
    # 以"面板中出现过的全部 user_id"为骨架建立空 DataFrame
    features = pd.DataFrame(index=events['user_id'].unique())
    features.index.name = 'user_id'
    # 把各聚合结果按 user_id 索引 join 进来（没有该行为的用户对应列为 NaN）
    parts = [purchases, views, sessions, duration, cart_products, purchased_products]
    joined = features.join(parts)
    # 把 user_id 从索引还原成普通列
    features = joined.reset_index()

    # ── 第 7 步：缺失值兜底与类型规整 ──
    # 需要把 NaN 补成 0 的数值列（没有购买/浏览等行为的用户）
    zero_fill_cols = ['Purchase_Frequency', 'Total_Spending', 'Pages_Viewed',
                      'Estimated_Time', 'Cart_Products', 'Purchased_Products']
    # 一次性把所有 NaN 填成 0
    features[zero_fill_cols] = features[zero_fill_cols].fillna(0)
    # 计数类列统一转成整数类型（fillna 之后可能是浮点）
    features['Purchase_Frequency'] = features['Purchase_Frequency'].astype(int)
    features['Pages_Viewed'] = features['Pages_Viewed'].astype(int)
    features['Cart_Products'] = features['Cart_Products'].astype(int)
    features['Purchased_Products'] = features['Purchased_Products'].astype(int)

    # ── 第 8 步：最近购买距观察期结束的天数（Recency_Days）──
    # 回退值：观察期结束距面板最早事件日期的天数 + 1（从未购买的用户用）
    fallback_recency = (observation_end - events['event_time'].min()).days + 1
    # 距最近一次购买的时间差（从未购买 → NaN）
    recency_timedelta = observation_end - features['Last_Purchase']
    # 时间差换算成天数
    recency_days = recency_timedelta.dt.days
    # NaN（从未购买）用回退值兜底，再转成整数
    recency_filled = recency_days.fillna(fallback_recency)
    features['Recency_Days'] = recency_filled.astype(int)

    # ── 第 9 步：去掉中间列并叠加探索度/摩擦指标 ──
    # Last_Purchase 只用于算 Recency_Days，之后不再需要
    feats = features.drop(columns='Last_Purchase')
    if return_scalers and scalers is None:
        # 调用方要求返回标准化器且未传入：在特征表上拟合 E_Score 的三个标准化器
        scalers = fit_engagement_scalers(feats)
    # 叠加 E_Score / Friction / Log_Friction（scalers 传入时复用、不重拟合）
    out = compute_engagement_metrics(feats, scalers=scalers)
    if return_scalers:
        # 队列迁移场景：返回 (特征表, 标准化器)
        return out, scalers
    # 默认场景：只返回特征表
    return out


def gmm_intersection_threshold(series: pd.Series, random_state: int = RANDOM_STATE) -> float:
    """
    两成分 GMM 的后验概率交点；拟合不稳定时退回中位数。

    阈值取两成分后验概率相等的位置，综合考虑分量方差与权重。
    零膨胀（大量 0 值，如“加购未买”计数）场景下，若交点落在 ≤0 且序列本身
    非负，则取最小正值作为阈值，语义即“有加购即高摩擦”。
    """
    # 先把正负无穷替换成 NaN，再丢弃 NaN，得到干净的观测值
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    x = cleaned.to_numpy()
    # 样本太少或取值种类太少时 GMM 拟合不可靠 → 退回中位数
    if len(x) < 50 or np.unique(x).size < 4:
        return float(np.nanmedian(x))
    # 拟合两成分高斯混合：一个成分通常是"多数普通值"，另一个是"少数高值"
    model = GaussianMixture(n_components=2, random_state=random_state, n_init=5)
    x_2d = x.reshape(-1, 1)
    model.fit(x_2d)
    # 在 [1% 分位, 99% 分位] 之间均匀取 2000 个点作为候选阈值
    low_q = np.quantile(x, 0.01)
    high_q = np.quantile(x, 0.99)
    grid = np.linspace(low_q, high_q, 2000)
    grid_2d = grid.reshape(-1, 1)
    # 每个网格点上成分 0 的后验概率（离 0.5 越近 = 两个成分概率越接近）
    posterior = model.predict_proba(grid_2d)
    posterior_comp0 = posterior[:, 0]
    posterior_gap = np.abs(posterior_comp0 - 0.5)
    # 取"两成分后验概率最接近相等"的网格点作为交点阈值
    cut = float(grid[np.argmin(posterior_gap)])
    # 序列的正数部分（零膨胀场景下代表"真正有正向行为"的取值）
    positive = x[x > 0]
    # 零膨胀退化兜底：交点 ≤ 0 而数据本身非负且存在正值 → 取最小正值
    # （语义即"只要有加购/正向行为就算高摩擦"）
    if cut <= 0 and x.min() >= 0 and positive.size > 0:
        return float(np.min(positive))
    return cut


def segment_users(features: pd.DataFrame, thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    生成基期可解释、互斥的运营人群。

    以“是否已购”为首要业务边界：已购用户按价值指数上四分位识别高价值，
    未购用户按行为指标识别首购潜力；每个指标只在语义匹配的人群内计算。

    未购人群：E_Score 与 Friction（加购未买）的 GMM 阈值识别高潜力首购。
    已购人群：高价值用户再按 E_Score 细分为深度互动 / 直购；
    “高价值高摩擦用户”（= 跨月完全沉默）由 flag_buyer_silence 结合观察月数据标记。

    参数 thresholds：传入基期拟合的阈值字典则冻结使用（队列迁移分析中
    保证跨月标签可比），为 None 时当月重新拟合。
    """
    # 在副本上操作，避免污染传入的特征表
    df = features.copy()
    # 是否已购：购买频次 > 0
    purchased = df['Purchase_Frequency'].gt(0)
    # 已购用户子集
    buyer = df.loc[purchased].copy()
    # 防御：没有任何购买用户时，分层没有意义
    if buyer.empty:
        raise ValueError('建模期没有购买用户，无法完成分层。')

    # ── 1. 手工 RFM 价值指数（消费 + 频次 - 沉默，全部对数化）──
    # log1p(总消费)：压缩金额长尾，只保留数量级差异
    spending_log = np.log1p(buyer['Total_Spending'])
    # log1p(购买频次)：次数越多越有价值
    frequency_log = np.log1p(buyer['Purchase_Frequency'])
    # log1p(最近购买天数)：越久没买价值越低（此项用减号）
    recency_log = np.log1p(buyer['Recency_Days'])
    # 价值指数只在已购用户上计算
    value_index = spending_log + frequency_log - recency_log
    # 先整列占位 NaN，再只给已购用户填值（未购用户保持空）
    df['Value_Index'] = np.nan
    df.loc[purchased, 'Value_Index'] = value_index
    # 第一层业务边界：已购 / 未购
    df['Base_Segment'] = np.where(purchased, '已购用户', '未购用户')

    # 未购用户的布尔掩码（后续反复使用）
    nonbuyer_mask = ~purchased

    # ── 2. 确定分层阈值：当月重新拟合 或 冻结基期阈值 ──
    if thresholds is None:
        # 默认：当月拟合全部阈值
        # 高价值线 = 价值指数的上四分位数（Q3）
        value_cut = float(value_index.quantile(0.75))
        # 未购人群的探索度 / 加购摩擦阈值：各自 GMM 两成分交点
        nonbuyer_escore_cut = gmm_intersection_threshold(df.loc[nonbuyer_mask, 'E_Score'])
        nonbuyer_friction_cut = gmm_intersection_threshold(df.loc[nonbuyer_mask, 'Log_Friction'])
    else:
        # 冻结阈值：沿用基期拟合的阈值重算标签（队列迁移分析，保证跨月标签可比）
        value_cut = thresholds['vip_value_index_cutoff']
        nonbuyer_escore_cut = thresholds['nonbuyer_e_score_cutoff']
        nonbuyer_friction_cut = thresholds['nonbuyer_log_friction_cutoff']

    # 高价值用户 = 已购且价值指数 ≥ 高价值线
    vip_mask = purchased & (df['Value_Index'] >= value_cut)
    if thresholds is None:
        # 高价值用户内部再按探索度 GMM 交点细分（当月拟合）
        vip_escore_cut = gmm_intersection_threshold(df.loc[vip_mask, 'E_Score'])
    else:
        # 冻结基期的高价值探索度阈值
        vip_escore_cut = thresholds['vip_e_score_cutoff']

    # ── 3. 生成互斥人群标签：先全部置默认，再逐条覆盖 ──
    # 默认标签：普通浏览用户
    df['User_Segment'] = '普通浏览用户'
    # 未购用户中，探索度与加购摩擦都过阈值 → 高潜力首购用户
    high_potential_mask = (nonbuyer_mask
                           & (df['E_Score'] >= nonbuyer_escore_cut)
                           & (df['Log_Friction'] >= nonbuyer_friction_cut))
    df.loc[high_potential_mask, 'User_Segment'] = '高潜力首购用户'
    # 已购但价值指数未达高价值线 → 常规已购用户
    regular_buyer_mask = purchased & ~vip_mask
    df.loc[regular_buyer_mask, 'User_Segment'] = '常规已购用户'
    # 高价值用户先统一记为"直购用户"（下单干脆、不太深逛）
    df.loc[vip_mask, 'User_Segment'] = '高价值直购用户'
    # 高价值且探索度也过阈值 → 覆盖为"高价值深度互动用户"
    deep_interaction_mask = vip_mask & (df['E_Score'] >= vip_escore_cut)
    df.loc[deep_interaction_mask, 'User_Segment'] = '高价值深度互动用户'

    # ── 4. 返回：分层结果 + 本次用到的全部阈值（供冻结复用）──
    metadata = {'vip_value_index_cutoff': value_cut,
                'nonbuyer_e_score_cutoff': nonbuyer_escore_cut,
                'nonbuyer_log_friction_cutoff': nonbuyer_friction_cut,
                'vip_e_score_cutoff': vip_escore_cut}
    return df, metadata


def flag_buyer_silence(segmented: pd.DataFrame, obs_events: pd.DataFrame) -> pd.DataFrame:
    """
    已购人群“跨月沉默”高摩擦标记（流失判定）。

    基期的高价值用户（高价值直购 / 深度互动）在观察月完全无任何事件
    （view / cart / purchase 都没有）→ 标记为“高价值高摩擦用户”（= 沉默/流失）。
    未购用户与其他人群不受影响。
    """
    # 在副本上操作，避免影响调用方
    out = segmented.copy()
    # 防御：观察月没有事件时无法判断用户"是否出现过"
    if obs_events.empty:
        raise ValueError('观察月没有事件数据，无法判定沉默。')
    # 观察月里有任何事件的 user_id 集合（view / cart / purchase 都算"有活动"）
    obs_users = set(obs_events['user_id'])
    # 只看基期的高价值人群：高价值直购 / 高价值深度互动
    vip_mask = out['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])
    # 高价值用户若在观察月完全没有事件 → 标记为"高价值高摩擦用户"（沉默/流失）
    silence_mask = vip_mask & ~out['user_id'].isin(obs_users)
    out.loc[silence_mask, 'User_Segment'] = '高价值高摩擦用户'
    return out


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    返回简历/汇报可直接引用的人群概览
    """
    # 按人群分组聚合：每组的人数与各项均值指标
    grouped = df.groupby('User_Segment', as_index=False).agg(
        用户数=('user_id', 'size'), 平均消费=('Total_Spending', 'mean'),
        平均购买频次=('Purchase_Frequency', 'mean'), 平均探索度=('E_Score', 'mean'),
        平均摩擦力=('Friction', 'mean'))
    # 用户占比 = 各组人数 / 全体人数（注意：未购用户的消费/频次均值本身为 0，展示上自然偏低）
    total_users = grouped['用户数'].sum()
    grouped['用户占比'] = grouped['用户数'] / total_users
    # 按人数从多到少排序，方便汇报里先看大盘
    summary = grouped.sort_values('用户数', ascending=False)
    return summary


def proportion_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% CI，避免小比例下的 Wald 区间失真。"""
    # total == 0：没有样本可估计 → 返回 NaN 占位
    if total == 0:
        return np.nan, np.nan
    # Wilson 区间用标准正态分布的 95% 分位数（约 1.96）
    z = 1.96
    # 观测到的成功比例
    p = successes / total
    # Wilson 公式的分母：1 + z² / n
    d = 1 + z**2 / total
    # 区间中心（对小样本做向 0.5 的收缩）
    centre = (p + z**2 / (2 * total)) / d
    # 区间半宽
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    # 返回 (下界, 上界)
    return centre - half, centre + half


def rate_test(nov: pd.DataFrame, treatment_ids: set, control_ids: set, title: str) -> dict:
    """两组 11 月购买率对比：卡方检验（Yates 校正）+ Wilson 95% CI。

    当期望频数含 0（样本过小或某组无事件）导致卡方失效时，回退 Fisher 精确检验。
    """
    # 验证月里实际发生购买的 user_id 集合
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    # 目标组（treatment）与对照组（control）中分别有多少人购买了
    a = len(treatment_ids & buyers)
    b = len(control_ids & buyers)
    # 两组总人数
    n_a = len(treatment_ids)
    n_b = len(control_ids)
    # 从标题里取组名（title 形如 "2019-10→2019-11 高潜力首购"）
    group_name = title.split('：')[0]
    if n_a == 0 or n_b == 0:
        # 空组无法计算比率与检验（滚动验证中某月可能没有该人群），返回 NaN 占位
        return {'目标组': group_name, '目标人数': n_a, '目标购买率': np.nan,
                '对照人数': n_b, '对照购买率': np.nan, '购买率差': np.nan, 'p值': np.nan}
    # 两组各自的购买率
    rate_a = a / n_a
    rate_b = b / n_b
    # 2×2 列联表：[[目标组买了, 目标组没买], [对照组买了, 对照组没买]]
    table = [[a, n_a - a], [b, n_b - b]]
    try:
        # 卡方独立性检验（Yates 连续校正，更适合小样本）
        _, pvalue, _, _ = chi2_contingency(table, correction=True)
    except ValueError:
        # 卡方失效（期望频数含 0）时回退 Fisher 精确检验
        _, pvalue = fisher_exact(table, alternative='two-sided')
    # 两组购买率的 Wilson 95% 置信区间
    lo_a, hi_a = proportion_ci(a, n_a)
    lo_b, hi_b = proportion_ci(b, n_b)
    # 打印结果，供 Notebook / 滚动验证直接查看
    print(f'\n{title}')
    print(f'  目标组: {rate_a:.2%} ({a}/{n_a}), 95% CI [{lo_a:.2%}, {hi_a:.2%}]')
    print(f'  对照组: {rate_b:.2%} ({b}/{n_b}), 95% CI [{lo_b:.2%}, {hi_b:.2%}]')
    print(f'  购买率差: {rate_a-rate_b:+.2%}; 卡方检验 p={pvalue:.3g}')
    # 返回结构化结果（供滚动验证统一汇总）
    return {'目标组': group_name, '目标人数': n_a, '目标购买率': rate_a,
            '对照人数': n_b, '对照购买率': rate_b, '购买率差': rate_a-rate_b, 'p值': pvalue}


def export_tracking(segmented: pd.DataFrame, path: Path | str | None = None,
                    segments: list | None = None) -> pd.DataFrame:
    """
    导出用户名单（全量分层名单 / A/B 候选抽样框）。

    默认导出**全部用户**的标签与概率分（运营按 User_Segment 筛选差异化策略：
    高潜力首购→首购激励、高价值高摩擦→流失召回、深度互动→会员运营、
    常规已购→复购唤醒、直购→快捷复购、普通浏览→潜力池）；
    传入 segments 时只导出指定标签（如候选触达名单 = 高潜力首购 + 高价值高摩擦）。

    排序层：名单附带未购用户首购概率分（First_Purchase_Prob / Rank / TopK_Flag）
    与已购用户复购概率分（Repurchase_Prob / Rank），运营可按预算取 rank ≤ k；
    解释层：保留 User_Segment / E_Score / Friction 供策略话术使用。
    """
    # 导出名单所需的固定列集合（排序层 + 解释层字段）
    cols = ['user_id', 'User_Segment', 'E_Score', 'Friction', 'Value_Index',
            'First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag',
            'Repurchase_Prob', 'Repurchase_Rank']
    if segments is None:
        # 默认：导出全部用户的标签与概率分（运营自行按 User_Segment 差异化运营）
        tracking = segmented.loc[:, cols].copy()
    else:
        # 只导出指定标签的用户（如候选触达名单 = 高潜力首购 + 高价值高摩擦）
        in_segments = segmented['User_Segment'].isin(segments)
        tracking = segmented.loc[in_segments, cols]
    if path is not None:
        # 写到 CSV（带 BOM，Excel 打开不乱码）
        tracking.to_csv(path, index=False, encoding='utf-8-sig')
    return tracking


def _nonbuyer_lr_features(nb: pd.DataFrame) -> np.ndarray:
    """未购用户的 LR 特征矩阵：log1p(页数 / 停留时长 / 会话数 / 加购商品数)。"""
    # 四个输入特征各自做 log1p 变换（压缩长尾、处理零值）
    pages_log = np.log1p(nb['Pages_Viewed'])
    time_log = np.log1p(nb['Estimated_Time'])
    sessions_log = np.log1p(nb['Session_Count'])
    cart_log = np.log1p(nb['Cart_Products'])
    # 把 4 个一维向量按列堆叠成 (样本数, 4) 的特征矩阵
    feature_columns = [pages_log, time_log, sessions_log, cart_log]
    x_matrix = np.column_stack(feature_columns)
    return x_matrix


def _buyer_lr_features(b: pd.DataFrame) -> np.ndarray:
    """已购用户的 LR 特征矩阵：log1p(RFM + 浏览/加购行为共 7 维)。"""
    # 前三个是 RFM 类特征：消费金额、购买频次、最近购买天数
    spending_log = np.log1p(b['Total_Spending'])
    frequency_log = np.log1p(b['Purchase_Frequency'])
    recency_log = np.log1p(b['Recency_Days'])
    # 后四个是行为类特征：页数、停留时长、会话数、加购商品数
    pages_log = np.log1p(b['Pages_Viewed'])
    time_log = np.log1p(b['Estimated_Time'])
    sessions_log = np.log1p(b['Session_Count'])
    cart_log = np.log1p(b['Cart_Products'])
    # 把 7 个一维向量按列堆叠成 (样本数, 7) 的特征矩阵
    feature_columns = [spending_log, frequency_log, recency_log,
                       pages_log, time_log, sessions_log, cart_log]
    x_matrix = np.column_stack(feature_columns)
    return x_matrix


_BUYER_FEATURE_NAMES = ['log1p(Total_Spending)', 'log1p(Purchase_Frequency)', 'log1p(Recency_Days)',
                        'log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)',
                        'log1p(Cart_Products)']


def _lr_pipeline() -> 'object':
    """标准化的 (StandardScaler + LogisticRegression) 流水线。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    # 流水线 = 先 Z 标准化再逻辑回归：统一特征尺度，利于模型收敛与系数可读
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    pipeline = make_pipeline(scaler, lr)
    return pipeline


def nonbuyer_baseline(segmented: pd.DataFrame, nov: pd.DataFrame,
                      random_state: int = RANDOM_STATE) -> dict:
    """
    未购用户首购基准：以 10 月未购用户为样本，用逻辑回归预测 11 月是否首购，
    量化规则分层（高潜力首购）的判别力与边际增量。

    返回 dict：{'metrics': {...}, 'preds': DataFrame(user_id, y, p_lr, rule)}，
    其中 p_lr 为 5 折 OOF 预测概率，供 Notebook 绘制校准/Top-k 提升曲线。
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # ── 1. 样本与标签：基期未购用户 → 验证月是否首购 ──
    # 取基期未购用户（购买频次 == 0）作为建模样本
    nb = segmented[segmented['Purchase_Frequency'].eq(0)].copy()
    # 防御：没有未购用户时无法建模
    if nb.empty:
        raise ValueError('没有未购用户，无法运行基准。')
    # 验证月里实际发生购买的 user_id 集合
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    # 标签 y：1 = 验证月完成首购，0 = 仍未购买
    # 先判断每个未购用户是否出现在验证月的购买名单里（得到 True/False 序列）
    is_buyer = nb['user_id'].isin(buyers)
    # 布尔值转成 1/0 整数
    y_int = is_buyer.astype(int)
    # 转成 NumPy 数组，供 sklearn 拟合使用
    y = y_int.to_numpy()

    # ── 2. 特征矩阵与规则标记 ──
    # 完整特征矩阵（4 维浏览/加购行为）
    x_full = _nonbuyer_lr_features(nb)
    # 只含前 3 维浏览特征（用于对比"仅浏览特征"能到多少 AUC）
    x_vol = x_full[:, :3]
    # 规则标记：规则分层命中的"高潜力首购用户"记为 1
    is_potential = nb['User_Segment'].eq('高潜力首购用户')
    rule_int = is_potential.astype(int)
    rule = rule_int.to_numpy()

    # ── 3. 5 折分层交叉验证：产出 OOF 预测 ──
    # 分层 K 折：保证每折正负样本比例与整体一致
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    # LR 流水线（标准化 + 逻辑回归）
    pipe = _lr_pipeline()
    # OOF：仅浏览特征预测首购概率
    p_lr_vol = cross_val_predict(pipe, x_vol, y, cv=cv, method='predict_proba')
    p_lr_vol = p_lr_vol[:, 1]
    auc_vol = roc_auc_score(y, p_lr_vol)
    # OOF：完整特征预测首购概率（主模型）
    p_lr = cross_val_predict(pipe, x_full, y, cv=cv, method='predict_proba')
    p_lr = p_lr[:, 1]
    auc_lr = roc_auc_score(y, p_lr)
    brier_lr = brier_score_loss(y, p_lr)
    # OOF：完整特征再拼上规则标记（看规则能否带来增量信息）
    x_full_rule = np.column_stack([x_full, rule])
    p_lr_rule = cross_val_predict(pipe, x_full_rule, y, cv=cv, method='predict_proba')
    p_lr_rule = p_lr_rule[:, 1]
    auc_lr_rule = roc_auc_score(y, p_lr_rule)
    # 规则本身当作一个"预测器"来评估：AUC 与 Brier
    auc_rule = roc_auc_score(y, rule)
    brier_rule = brier_score_loss(y, rule)

    # ── 4. Top-k 对比：规则人数 = k，双方都取前 k 名看实际购买率 ──
    # 规则圈出的人数
    k = int(rule.sum())
    # 按 LR 概率从高到低排序的用户下标
    order = np.argsort(-p_lr)
    # LR 概率最高的前 k 人的实际购买率
    top_lr_rate = float(y[order[:k]].mean())
    # 规则圈出的人群（按定义就是 top k）的实际购买率
    top_rule_rate = float(y[rule == 1].mean())

    # ── 5. 全量拟合一次，输出各特征系数（业务解释用）──
    pipe.fit(x_full, y)
    # 特征名与系数一一对应
    coef_names = ['log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)',
                  'log1p(Cart_Products)']
    coef_values = pipe.named_steps['logisticregression'].coef_[0]
    coef = {}
    for name, value in zip(coef_names, coef_values):
        coef[name] = float(value)

    # ── 6. Bootstrap：LR AUC − 规则 AUC 的 95% 置信区间 ──
    # 固定随机数种子，保证结果可复现
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(BOOTSTRAP_N):
        # 有放回地抽取与样本等长的下标（一次自举抽样）
        idx = rng.choice(len(y), size=len(y), replace=True)
        # 在同一批自举样本上分别算 LR 与规则的 AUC，再取差
        auc_lr_boot = roc_auc_score(y[idx], p_lr[idx])
        auc_rule_boot = roc_auc_score(y[idx], rule[idx])
        diffs.append(auc_lr_boot - auc_rule_boot)
    # 取 2.5% / 97.5% 分位数作为 95% 置信区间
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # ── 7. 汇总指标与逐用户预测结果 ──
    metrics = {
        '样本(未购用户)': int(len(nb)),
        '11月购买率': float(y.mean()),
        '规则人群规模': k,
        '规则 Top-k 购买率': top_rule_rate,
        'LR Top-k 购买率': top_lr_rate,
        '规则 AUC': auc_rule,
        'LR AUC (5折OOF)': auc_lr,
        'LR AUC 仅浏览特征': auc_vol,
        'LR AUC +规则标记': auc_lr_rule,
        'LR Brier (OOF)': brier_lr,
        '规则 Brier': brier_rule,
        'LR−规则 AUC 差 (bootstrap 95% CI)': (float(round(lo, 4)), float(round(hi, 4))),
        'coefficients': coef,
    }
    # 逐用户预测结果：供 Notebook 绘制校准曲线与 Top-k 提升曲线
    preds = pd.DataFrame({'user_id': nb['user_id'], 'y': y, 'p_lr': p_lr, 'rule': rule})
    return {'metrics': metrics, 'preds': preds}


def buyer_baseline(segmented: pd.DataFrame, nov: pd.DataFrame,
                   random_state: int = RANDOM_STATE) -> dict:
    """
    已购用户复购基准：以基期已购用户为样本，用逻辑回归预测验证月是否复购，
    检验“高价值高摩擦”（跨月完全沉默，需先经 flag_buyer_silence 标记）规则
    与手工 RFM 价值指数作为排序器的判别力。

    双视角解读：
    - Top-k：谁最可能复购（运营触达视角）；
    - Bottom-k：谁最可能流失（风险预警视角，与沉默规则的语义对应）。

    返回 dict：{'metrics': {...}, 'preds': DataFrame(user_id, y, p_lr, rule, value_index)}，
    其中 p_lr 为 5 折 OOF 复购概率，供 Notebook 绘制校准与 Top/Bottom-k 曲线。
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # ── 1. 样本与标签：基期已购用户 → 验证月是否复购 ──
    # 取基期已购用户（购买频次 > 0）作为建模样本
    buyer = segmented[segmented['Purchase_Frequency'].gt(0)].copy()
    # 防御：没有已购用户时无法建模
    if buyer.empty:
        raise ValueError('没有已购用户，无法运行复购基准。')
    # 验证月里实际发生购买的 user_id 集合
    buyers_nov = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    # 标签 y：1 = 验证月复购，0 = 未复购
    # 先判断每个已购用户是否出现在验证月的购买名单里（得到 True/False 序列）
    is_buyer = buyer['user_id'].isin(buyers_nov)
    # 布尔值转成 1/0 整数
    y_int = is_buyer.astype(int)
    # 转成 NumPy 数组，供 sklearn 拟合使用
    y = y_int.to_numpy()
    # 防御：验证期必须两类样本都有，否则无法拟合
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有复购样本，无法拟合复购基准。')

    # ── 2. 特征矩阵、规则标记与手工价值指数 ──
    # 完整特征矩阵（7 维：RFM + 行为）
    x_full = _buyer_lr_features(buyer)
    # 规则标记：被标记为"高价值高摩擦"（跨月沉默）的用户记为 1
    is_silent = buyer['User_Segment'].eq('高价值高摩擦用户')
    rule_int = is_silent.astype(int)
    rule = rule_int.to_numpy()
    # 手工 RFM 价值指数（作为对比排序器）
    value_score = buyer['Value_Index'].to_numpy()

    # ── 3. 5 折分层交叉验证：产出 OOF 预测 ──
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    pipe = _lr_pipeline()
    # OOF：完整特征预测复购概率（主模型）
    p_lr = cross_val_predict(pipe, x_full, y, cv=cv, method='predict_proba')
    p_lr = p_lr[:, 1]
    auc_lr = roc_auc_score(y, p_lr)
    brier_lr = brier_score_loss(y, p_lr)
    # OOF：完整特征 + 沉默规则标记（看规则能否带来增量信息）
    x_full_rule = np.column_stack([x_full, rule])
    p_lr_rule = cross_val_predict(pipe, x_full_rule, y, cv=cv, method='predict_proba')
    p_lr_rule = p_lr_rule[:, 1]
    auc_lr_rule = roc_auc_score(y, p_lr_rule)
    # 规则本身 / 手工价值指数各自作为排序器的 AUC
    auc_rule = roc_auc_score(y, rule)
    brier_rule = brier_score_loss(y, rule)
    auc_value = roc_auc_score(y, value_score)

    # ── 4. Top-k / Bottom-k 对比（双视角解读）──
    # 规则圈出的人数
    k = int(rule.sum())
    # 按 LR 概率从高到低排序的用户下标
    order = np.argsort(-p_lr)
    # LR 概率最高的前 k 人 = 最可能复购 → 实际复购率（运营触达视角）
    top_lr_rate = float(y[order[:k]].mean())
    # LR 概率最低的后 k 人 = 最可能流失 → 实际复购率（风险预警视角）
    bottom_lr_rate = float(y[order[-k:]].mean())
    # 规则圈出人群（按定义就是 top k by 沉默规则）的实际复购率
    top_rule_rate = float(y[rule == 1].mean())

    # ── 5. 全量拟合一次，输出各特征系数（业务解释用）──
    pipe.fit(x_full, y)
    coef_values = pipe.named_steps['logisticregression'].coef_[0]
    coef = {}
    for name, value in zip(_BUYER_FEATURE_NAMES, coef_values):
        coef[name] = float(value)

    # ── 6. Bootstrap：LR AUC − 规则 AUC 的 95% 置信区间 ──
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(BOOTSTRAP_N):
        # 有放回地抽取与样本等长的下标（一次自举抽样）
        idx = rng.choice(len(y), size=len(y), replace=True)
        # 在同一批自举样本上分别算 LR 与规则的 AUC，再取差
        auc_lr_boot = roc_auc_score(y[idx], p_lr[idx])
        auc_rule_boot = roc_auc_score(y[idx], rule[idx])
        diffs.append(auc_lr_boot - auc_rule_boot)
    # 取 2.5% / 97.5% 分位数作为 95% 置信区间
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # ── 7. 汇总指标与逐用户预测结果 ──
    metrics = {
        '样本(已购用户)': int(len(buyer)),
        '11月复购率': float(y.mean()),
        '规则人群规模': k,
        '规则 Top-k 复购率': top_rule_rate,
        'LR Top-k 复购率': top_lr_rate,
        'LR Bottom-k 复购率(风险视角)': bottom_lr_rate,
        '规则 AUC': auc_rule,
        '手工价值指数 AUC': auc_value,
        'LR AUC (5折OOF)': auc_lr,
        'LR AUC +规则标记': auc_lr_rule,
        'LR Brier (OOF)': brier_lr,
        '规则 Brier': brier_rule,
        'LR−规则 AUC 差 (bootstrap 95% CI)': (float(round(lo, 4)), float(round(hi, 4))),
        'coefficients': coef,
    }
    # 逐用户预测结果：供 Notebook 绘制校准曲线与 Top/Bottom-k 曲线
    preds = pd.DataFrame({'user_id': buyer['user_id'], 'y': y, 'p_lr': p_lr, 'rule': rule,
                          'value_index': value_score})
    return {'metrics': metrics, 'preds': preds}


def score_nonbuyers(segmented: pd.DataFrame, nov: pd.DataFrame,
                    pool_segments: list | None = None,
                    top_ratio: float | None = None) -> pd.DataFrame:
    """
    对未购用户全量拟合并输出连续首购概率分（排序层），
    返回带 First_Purchase_Prob / First_Purchase_Rank / TopK_Flag 的副本。

    规则圈池 + LR 池内排序：候选池 = 规则认可的人群（默认 POOL_SEGMENTS，
    未购人群里即"高潜力首购用户"，低意向的"普通浏览用户"不入池）；
    LR 概率分只在池内排名，TopK_Flag 标记池内前 top_ratio（默认 50%）的人优先触达
    ——k = ceil(池内人数 × top_ratio)，不再是"目标群体的总人数"。
    非池用户保留概率分（供查看），但 Rank / TopK_Flag 为 NaN。
    注：本列为全量拟合的排序分，用于选人；预期触达效果以 nonbuyer_baseline
    的 OOF 评估为准。上线时需用滚动历史窗口训练、未来月验证并定期重校准。
    """
    # 参数默认值：未显式传入时用 config 里的候选池与 top 比例
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    if top_ratio is None:
        top_ratio = POOL_TOP_RATIO
    # 在副本上叠加概率分，避免污染调用方
    out = segmented.copy()
    # 未购用户掩码（只有他们能拿到首购概率分）
    nb_mask = out['Purchase_Frequency'].eq(0)
    # 未购用户子集
    nb = out.loc[nb_mask]
    # 防御：没有未购用户时无法打分
    if nb.empty:
        raise ValueError('没有未购用户，无法打分。')
    # 验证月实际购买的 user_id 集合 → 训练标签
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    # 判断每个未购用户是否出现在验证月的购买名单里（得到 True/False 序列）
    is_buyer = nb['user_id'].isin(buyers)
    # 布尔值转成 1/0 整数，再转 NumPy 数组供 sklearn 使用
    y_int = is_buyer.astype(int)
    y = y_int.to_numpy()
    # 防御：验证期必须两类样本都有，否则拟合不出有意义的概率分
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有购买样本，无法拟合首购概率分。')
    # 全量拟合 LR 并预测未购用户的购买概率
    pipe = _lr_pipeline()
    nb_features = _nonbuyer_lr_features(nb)
    pipe.fit(nb_features, y)
    proba_full = pipe.predict_proba(nb_features)
    # 取"购买"那一类的概率（第 2 列）
    proba = proba_full[:, 1]

    # 三列先占位 NaN，再按人群填充
    out['First_Purchase_Prob'] = np.nan
    out['First_Purchase_Rank'] = np.nan
    out['TopK_Flag'] = np.nan
    # 所有未购用户都保留概率分（供查看），非未购用户保持 NaN
    out.loc[nb_mask, 'First_Purchase_Prob'] = proba

    # 只在候选池内排序：候选池 = 未购用户中规则认可的人群（低意向的普通浏览不入池）
    pool_mask = nb_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        # 概率分转成以 nb 行为索引的 Series，再只取池内用户
        proba_series = pd.Series(proba, index=nb.index)
        pool_proba = proba_series.loc[pool_idx]
        # 池内按概率降序排名（method='min'：并列给相同的最小名次）
        ranks = pool_proba.rank(ascending=False, method='min')
        # 优先触达人数 k = ceil(池内人数 × top_ratio)
        k = int(np.ceil(len(pool_idx) * top_ratio))
        out.loc[pool_idx, 'First_Purchase_Rank'] = ranks
        # 池内前 k 名标记为 TopK（1），其余为 0
        topk_mask = ranks <= k
        out.loc[pool_idx, 'TopK_Flag'] = topk_mask.astype(int)
    return out


def score_buyers(segmented: pd.DataFrame, nov: pd.DataFrame,
                 pool_segments: list | None = None) -> pd.DataFrame:
    """
    对已购用户全量拟合并输出连续复购概率分（排序层），
    返回带 Repurchase_Prob / Repurchase_Rank 的副本（未购用户为 NaN）。

    与 score_nonbuyers 对应：规则圈池 + LR 池内排序——复购排名只在
    候选池内计算（默认 POOL_SEGMENTS，即规则认可的人群，供触达排序
    Top-k / 流失预警 Bottom-k 按预算取用）；非池用户保留概率分但 Rank 为 NaN。
    注：本列为全量拟合的排序分；预期效果以 buyer_baseline 的 OOF 评估为准。
    上线时需用滚动历史窗口训练、未来月验证并定期重校准。
    """
    # 参数默认值：未显式传入时用 config 里的候选池
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    # 在副本上叠加概率分，避免污染调用方
    out = segmented.copy()
    # 已购用户掩码（只有他们能拿到复购概率分）
    buyer_mask = out['Purchase_Frequency'].gt(0)
    # 已购用户子集
    buyer = out.loc[buyer_mask]
    # 防御：没有已购用户时无法打分
    if buyer.empty:
        raise ValueError('没有已购用户，无法打分。')
    # 验证月实际购买的 user_id 集合 → 训练标签
    buyers_nov = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    # 判断每个已购用户是否出现在验证月的购买名单里（得到 True/False 序列）
    is_buyer = buyer['user_id'].isin(buyers_nov)
    # 布尔值转成 1/0 整数，再转 NumPy 数组供 sklearn 使用
    y_int = is_buyer.astype(int)
    y = y_int.to_numpy()
    # 防御：验证期必须两类样本都有，否则拟合不出有意义的概率分
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有复购样本，无法拟合复购概率分。')
    # 全量拟合 LR 并预测已购用户的复购概率
    pipe = _lr_pipeline()
    buyer_features = _buyer_lr_features(buyer)
    pipe.fit(buyer_features, y)
    proba_full = pipe.predict_proba(buyer_features)
    # 取"复购"那一类的概率（第 2 列）
    proba = proba_full[:, 1]

    # 两列先占位 NaN，再按人群填充
    out['Repurchase_Prob'] = np.nan
    out['Repurchase_Rank'] = np.nan
    # 所有已购用户都保留概率分（供查看），未购用户保持 NaN
    out.loc[buyer_mask, 'Repurchase_Prob'] = proba

    # 只在候选池内排序：池 = 已购用户中规则认可的人群
    pool_mask = buyer_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        # 概率分转成以 buyer 行为索引的 Series，再只取池内用户
        proba_series = pd.Series(proba, index=buyer.index)
        pool_proba = proba_series.loc[pool_idx]
        # 池内按概率降序排名（method='min'：并列给相同的最小名次）
        ranks = pool_proba.rank(ascending=False, method='min')
        out.loc[pool_idx, 'Repurchase_Rank'] = ranks
    return out


def _next_month(m: str) -> str:
    """'2019-10' -> '2019-11'（月份字符串递增）。"""
    # 拆出年份与月份两个整数（m 形如 '2019-10'）
    year = int(m[:4])
    month = int(m[5:7])
    if month == 12:
        # 12 月 → 下一年 1 月
        return f'{year + 1:04d}-01'
    # 其余月份：年份不变、月份 +1，保持两位补零格式
    return f'{year:04d}-{month + 1:02d}'


def _history_train_samples(panel: pd.DataFrame, score_month: str, buyer: bool,
                           cache: dict | None = None):
    """构建打分月之前的所有 (特征月 → 次月标签) 训练样本（未购/已购人群），返回 (X, y)。

    训练样本对象 = m 月用户（m < score_month），打分对象 = score_month 用户——
    模型从未见过打分对象的结果（次月购买），严格避免"偷看答案"。
    cache：可选 dict（month → 特征表 / '_by_month' → 按月分片），跨打分月复用。
    """
    # ── 1. 按月分片（一次性），避免对 2000 万+ 行面板反复全表布尔过滤 ──
    if cache is not None and '_by_month' in cache:
        # 缓存里已有按月分片 → 直接复用
        by_month = cache['_by_month']
    else:
        # 首次：把 panel 按 month 列切分成 {月名 -> 该月事件表} 的字典
        by_month = {}
        for m, g in panel.groupby('month'):
            by_month[m] = g
        if cache is not None:
            cache['_by_month'] = by_month
    # 按月名排序（'YYYY-MM' 字典序即时间顺序）
    months = sorted(by_month)
    # X_parts：各月特征矩阵片段；y_parts：各月标签片段（之后纵向拼接）
    X_parts = []
    y_parts = []
    for m in months:
        # 当前特征月的下一个自然月（标签月）
        nm = _next_month(m)
        # 若标签月已经超过打分月，后续月份只会更晚 → 终止循环
        if nm > score_month:
            break
        ev_m = by_month.get(m)
        ev_nm = by_month.get(nm)
        # 特征月或标签月缺数据 / 为空 → 该月对无法构成训练样本，跳过
        if ev_m is None or ev_nm is None or ev_m.empty or ev_nm.empty:
            continue
        if cache is not None and m in cache:
            # 该月特征已缓存 → 直接复用
            feats = cache[m]
        else:
            # 用当月事件构建用户特征（观察期终点 = 当月最后一条事件时间）
            feats = build_features(ev_m, ev_m['event_time'].max())
            if cache is not None:
                cache[m] = feats
        # 标签月里实际购买的 user_id 集合（打分时模型看不到这份"答案"）
        buyers_nm = set(ev_nm.loc[ev_nm['event_type'].eq('purchase'), 'user_id'])
        if buyer:
            # 已购人群分支：样本 = 当月已购用户，标签 = 次月是否复购
            sub = feats[feats['Purchase_Frequency'].gt(0)].copy()
            # 判断当月已购用户是否在次月又购买（True/False → 1/0）
            is_buyer = sub['user_id'].isin(buyers_nm)
            y_int = is_buyer.astype(int)
            y = y_int.to_numpy()
            X_parts.append(_buyer_lr_features(sub))
        else:
            # 未购人群分支：样本 = 当月未购用户，标签 = 次月是否首购
            sub = feats[feats['Purchase_Frequency'].eq(0)].copy()
            # 判断当月未购用户是否在次月完成首购（True/False → 1/0）
            is_buyer = sub['user_id'].isin(buyers_nm)
            y_int = is_buyer.astype(int)
            y = y_int.to_numpy()
            X_parts.append(_nonbuyer_lr_features(sub))
        y_parts.append(y)
    # 防御：打分月之前没有任何训练数据（10 月是最早建模月，不需要选人）
    if not X_parts:
        raise ValueError(
            f'{score_month} 无法给出概率分：没有更早月份的训练数据。'
            '（10 月是最早建模月，不需要选人；LR 需要"上上个月/上个月"的历史行为训练，'
            '打分月最早从 11 月开始）')
    # 纵向拼接所有月份：特征行数总和 = 标签长度
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    return X_train, y_train


def score_nonbuyers_history(panel: pd.DataFrame, score_month: str,
                            pool_segments: list | None = None,
                            top_ratio: float | None = None,
                            cache: dict | None = None) -> pd.DataFrame:
    """
    LR 历史窗口打分（未购人群，严格无泄漏）——替代 score_nonbuyers 的演示版全量拟合。

    时间窗口（3 个月）：
    - **训练**：score_month 之前所有 (特征月 → 次月标签) 的未购用户样本
      （如打分 12 月 = 10→11 与 11→12 两批，即"上上个月 + 上个月"的活动）；
    - **打分**：score_month 未购用户（当月特征），规则圈池 + LR 池内排序，
      取池内前 top_ratio（默认 50%）；
    - **结果**：score_month 的次月购买（训练时不可见）。

    ⚠️ 备注：**10 月无法给出概率分**——10 月是最早的建模/EDA 月，没有
    更早月份的历史行为可用于训练；10 月本身也不需要选人。打分最早从
    11 月开始（用 10 月特征 + 11 月标签训练）。
    cache：可选 dict（month → 特征表），跨打分月复用 build_features 结果。
    """
    # 参数默认值：未显式传入时用 config 里的候选池与 top 比例
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    if top_ratio is None:
        top_ratio = POOL_TOP_RATIO

    # ── 1. 用打分月之前的历史样本训练 LR（严格无泄漏）──
    # 训练样本 = 各 (特征月 → 次月标签) 的未购用户；打分月当月的结果不可见
    X_train, y_train = _history_train_samples(panel, score_month, buyer=False, cache=cache)
    pipe = _lr_pipeline()
    pipe.fit(X_train, y_train)

    # ── 2. 取打分月事件并构建当月特征 ──
    if cache is not None and '_by_month' in cache:
        # 复用缓存里的按月分片
        by_month = cache['_by_month']
    else:
        # 首次：把 panel 按月切分成 {月名 -> 事件表}
        by_month = {}
        for m, g in panel.groupby('month'):
            by_month[m] = g
        if cache is not None:
            cache['_by_month'] = by_month
    ev_sc = by_month.get(score_month)
    if cache is not None and score_month in cache:
        # 该月特征已缓存 → 直接复用
        feats_sc = cache[score_month]
    else:
        # 用打分月事件构建特征（观察期终点 = 当月最后一条事件时间）
        feats_sc = build_features(ev_sc, ev_sc['event_time'].max())
        if cache is not None:
            cache[score_month] = feats_sc

    # ── 3. 对打分月分层并取出未购用户 ──
    seg_sc, _ = segment_users(feats_sc)
    out = seg_sc.copy()
    nb_mask = out['Purchase_Frequency'].eq(0)
    nb = out.loc[nb_mask]
    # 防御：打分月没有未购用户
    if nb.empty:
        raise ValueError(f'{score_month} 没有未购用户，无法打分。')

    # ── 4. 打分：所有未购用户都给出首购概率分 ──
    proba_full = pipe.predict_proba(_nonbuyer_lr_features(nb))
    # 取"购买"那一类的概率（第 2 列）
    proba = proba_full[:, 1]

    # 三列先占位 NaN，再按人群填充
    out['First_Purchase_Prob'] = np.nan
    out['First_Purchase_Rank'] = np.nan
    out['TopK_Flag'] = np.nan
    out.loc[nb_mask, 'First_Purchase_Prob'] = proba

    # 规则圈池：候选池 = 未购用户中规则认可的人群；Rank / TopK 只在池内计算
    pool_mask = nb_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        # 概率分转成以 nb 行为索引的 Series，再只取池内用户
        proba_series = pd.Series(proba, index=nb.index)
        pool_proba = proba_series.loc[pool_idx]
        # 池内按概率降序排名（method='min'：并列给相同的最小名次）
        ranks = pool_proba.rank(ascending=False, method='min')
        # 优先触达人数 k = ceil(池内人数 × top_ratio)
        k = int(np.ceil(len(pool_idx) * top_ratio))
        out.loc[pool_idx, 'First_Purchase_Rank'] = ranks
        # 池内前 k 名标记为 TopK（1），其余为 0
        topk_mask = ranks <= k
        out.loc[pool_idx, 'TopK_Flag'] = topk_mask.astype(int)
    return out


def score_buyers_history(panel: pd.DataFrame, score_month: str,
                         pool_segments: list | None = None,
                         segmented: pd.DataFrame | None = None,
                         cache: dict | None = None) -> pd.DataFrame:
    """
    LR 历史窗口打分（已购人群，严格无泄漏）——替代 score_buyers 的演示版全量拟合。

    时间窗口与 score_nonbuyers_history 一致（训练 = score_month 之前的
    (特征月 → 次月复购标签) 已购样本；打分 = score_month 已购用户，池内排名）。
    segmented：可选——传入 score_nonbuyers_history 的结果可在其上叠加复购分列
    （否则内部重新对 score_month 分层）。
    cache：可选 dict（month → 特征表），跨打分月复用 build_features 结果。
    ⚠️ 备注：10 月无法给出概率分（无更早训练数据；10 月为建模月不需要选人）。
    """
    # 参数默认值：未显式传入时用 config 里的候选池
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS

    # ── 1. 用打分月之前的历史样本训练 LR（严格无泄漏）──
    # 训练样本 = 各 (特征月 → 次月复购标签) 的已购用户
    X_train, y_train = _history_train_samples(panel, score_month, buyer=True, cache=cache)
    pipe = _lr_pipeline()
    pipe.fit(X_train, y_train)

    # ── 2. 准备打分月的分层结果（未传入时内部重新分层）──
    if segmented is None:
        # 调用方没传已分层结果 → 重新走"按月分片 → 构建特征 → 分层"流程
        if cache is not None and '_by_month' in cache:
            # 复用缓存里的按月分片
            by_month = cache['_by_month']
        else:
            # 首次：把 panel 按月切分成 {月名 -> 事件表}
            by_month = {}
            for m, g in panel.groupby('month'):
                by_month[m] = g
            if cache is not None:
                cache['_by_month'] = by_month
        ev_sc = by_month.get(score_month)
        if cache is not None and score_month in cache:
            # 该月特征已缓存 → 直接复用
            feats_sc = cache[score_month]
        else:
            # 用打分月事件构建特征（观察期终点 = 当月最后一条事件时间）
            feats_sc = build_features(ev_sc, ev_sc['event_time'].max())
            if cache is not None:
                cache[score_month] = feats_sc
        seg_sc, _ = segment_users(feats_sc)
        out = seg_sc.copy()
    else:
        # 传入已有的 score_nonbuyers_history 结果：直接在其上叠加复购分列
        out = segmented.copy()

    # ── 3. 取出已购用户并打复购概率分 ──
    buyer_mask = out['Purchase_Frequency'].gt(0)
    buyer = out.loc[buyer_mask]
    # 防御：打分月没有已购用户
    if buyer.empty:
        raise ValueError(f'{score_month} 没有已购用户，无法打分。')
    proba_full = pipe.predict_proba(_buyer_lr_features(buyer))
    # 取"复购"那一类的概率（第 2 列）
    proba = proba_full[:, 1]

    # 两列先占位 NaN，再按人群填充
    out['Repurchase_Prob'] = np.nan
    out['Repurchase_Rank'] = np.nan
    out.loc[buyer_mask, 'Repurchase_Prob'] = proba

    # 规则圈池：复购排名只在候选池内计算（非池用户保留概率分但 Rank 为 NaN）
    pool_mask = buyer_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        # 概率分转成以 buyer 行为索引的 Series，再只取池内用户
        proba_series = pd.Series(proba, index=buyer.index)
        pool_proba = proba_series.loc[pool_idx]
        # 池内按概率降序排名（method='min'：并列给相同的最小名次）
        ranks = pool_proba.rank(ascending=False, method='min')
        out.loc[pool_idx, 'Repurchase_Rank'] = ranks
    return out


def load_panel(path: Path | str = PANEL_FILE, months: list | None = None,
               columns: list | None = None) -> pd.DataFrame:
    """读取用户面板（parquet 或 CSV），解析 event_time 并确保 month 列（'YYYY-MM'）存在。

    months：若指定（如 ['2019-10', '2019-11']），parquet 用行过滤只读这些月份
    （predicate pushdown，大幅减少 IO）；CSV 则读取后过滤。为 None 时读全量。
    columns：若指定（如分析必需列子集），只读这些列，降低 IO 与内存占用
    （面板 2000 万+ 行，字符串列如 category_code / brand 占用显著）。
    """
    # 统一成 Path，便于判断文件类型
    path = Path(path)
    if path.suffix == '.parquet':
        # ── parquet：用 filters / columns 做下推读取，减少 IO 与内存 ──
        try:
            # 传给 pd.read_parquet 的关键字参数（按需追加）
            kwargs = {}
            if months is not None:
                # 只读指定月份的行（predicate pushdown，大幅减少 IO）
                kwargs['filters'] = [('month', 'in', months)]
            if columns is not None:
                # 只读需要的列，降低内存占用
                kwargs['columns'] = columns
            panel = pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            # pyarrow 未安装时给出可操作的安装提示
            raise ImportError(
                f'读取面板 {path.name} 需要 parquet 引擎（pyarrow）。请先安装依赖：'
                f'pip install pyarrow  （或 pip install -r requirements.txt）') from exc
    else:
        # ── CSV：先整体读入，再在内存里过滤 ──
        panel = pd.read_csv(path, parse_dates=['event_time'])
        if columns is not None:
            # 只保留调用方要求、且确实存在的列（保持原顺序）
            keep_cols = []
            for c in columns:
                if c in panel.columns:
                    keep_cols.append(c)
            panel = panel[keep_cols]
        if months is not None:
            # 用 event_time 的前 7 个字符（'YYYY-MM'）匹配目标月份
            event_month = panel['event_time'].astype(str).str[:7]
            in_months = event_month.isin(months)
            panel = panel[in_months]
    if 'event_time' in panel.columns:
        # 确保 event_time 统一成 datetime 类型
        panel['event_time'] = pd.to_datetime(panel['event_time'])
    if 'month' not in panel.columns:
        # 从 event_time 派生 month 列（'YYYY-MM'），供按月分片使用
        event_month = panel['event_time'].astype(str).str[:7]
        panel['month'] = event_month
    return panel


# 各臂基准的指标键名映射（用于滚动验证统一汇总）
_ARM_METRICS_KEYS = {
    '未购人群(首购)': {'样本': '样本(未购用户)', '结果率': '11月购买率', 'topk': 'LR Top-k 购买率'},
    '已购人群(复购)': {'样本': '样本(已购用户)', '结果率': '11月复购率', 'topk': 'LR Top-k 复购率'},
}


def rolling_validation(panel: pd.DataFrame, months: list | None = None) -> dict:
    """
    滚动时间外验证（已购人群为跨月沉默定义）。

    - 实验一（未购人群）：基期月 t 特征分层 → 观察月 t+1 购买（2 月对）；
    - 实验二（已购人群）：基期月 t 高价值买家 → 观察月 t+1 完全沉默（高价值高摩擦）
       → 验证月 t+2 是否购买（3 月组，避免“沉默月=结果月”的循环定义）；
    - 两套 LR 基准：未购人群预测 t+1 首购；已购人群以“沉默”为规则标记、预测 t+2 复购。

    返回 {'验证表': DataFrame, '基准表': DataFrame}。
    """
    if months is None:
        months = MONTHS
    # 一次性按月分片（面板 2000 万+ 行，避免后续每对月份都全表扫描）
    by_month = {}
    for m, g in panel.groupby('month'):
        by_month[m] = g
    # rate_rows：两组购买率对比结果行；auc_rows：LR 基准指标行
    rate_rows = []
    auc_rows = []
    # 按月滑动：基期月 t → 观察月 t+1（实验二还要用到 t+2）
    for t in range(len(months) - 1):
        base_m = months[t]
        obs_m = months[t + 1]
        base_events = by_month.get(base_m)
        obs_events = by_month.get(obs_m)
        # 该月对任一方向缺数据 → 跳过（不构成可用实验窗口）
        if base_events is None or obs_events is None or base_events.empty or obs_events.empty:
            print(f'跳过 {base_m}→{obs_m}（无数据）')
            continue
        # 基期月构建特征 + 当月分层
        features = build_features(base_events, base_events['event_time'].max())
        segmented, _ = segment_users(features)

        # ── 实验一（未购人群）：基期特征 → 观察月购买 ──
        # 规则人群 = 高潜力首购用户；对照组 = 普通浏览用户
        potential = set(segmented.loc[segmented['User_Segment'].eq('高潜力首购用户'), 'user_id'])
        nonbuyer_control = set(segmented.loc[segmented['User_Segment'].eq('普通浏览用户'), 'user_id'])
        # 两组在观察月的购买率对比 + 显著性检验
        res1 = rate_test(obs_events, potential, nonbuyer_control, f'{base_m}→{obs_m} 高潜力首购')
        row1 = {'训练月': base_m, '沉默月': '', '验证月': obs_m, '实验': '高潜力首购'}
        # 把 rate_test 返回的各列指标并入这一行
        row1.update(res1)
        rate_rows.append(row1)

        # ── 未购人群 LR 基准 ──
        try:
            base = nonbuyer_baseline(segmented, obs_events)
            m = base['metrics']
            # 本臂指标键名与统一汇总列的映射
            keys = _ARM_METRICS_KEYS['未购人群(首购)']
            auc_row = {
                '训练月': base_m, '沉默月': '', '验证月': obs_m, '人群': '未购人群(首购)',
                '样本': m[keys['样本']], '全体未购用户平均首购率': m[keys['结果率']],
                '高潜力首购用户购买率': m['规则 Top-k 购买率'],
                '规则人群规模': m['规则人群规模'], '规则 AUC': m['规则 AUC'],
                'LR AUC (OOF)': m['LR AUC (5折OOF)'], 'LR Top-k 率': m[keys['topk']],
            }
            auc_rows.append(auc_row)
        except ValueError as exc:
            # 该月样本不足以拟合基准（如未购用户太少）→ 记录后跳过
            print(f'  [未购人群] {base_m}→{obs_m} 跳过: {exc}')

        # ── 实验二（已购人群·跨月沉默）：基期高价值 → 观察月沉默 → 验证月购买（需第 3 个月）──
        if t + 2 < len(months):
            out_m = months[t + 2]
            out_events = by_month.get(out_m)
            # 验证月缺数据 → 该三周窗口不成立
            if out_events is None or out_events.empty:
                print(f'跳过 {base_m}→{obs_m}→{out_m}（验证月无数据）')
                continue
            # 用观察月事件标记"跨月完全沉默"的高价值用户（= 高价值高摩擦）
            flagged = flag_buyer_silence(segmented, obs_events)
            # 沉默组：高价值高摩擦用户；对照组：观察月仍活跃的高价值用户
            silent_vip = set(flagged.loc[flagged['User_Segment'].eq('高价值高摩擦用户'), 'user_id'])
            active_vip_mask = flagged['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])
            active_vip = set(flagged.loc[active_vip_mask, 'user_id'])
            # 两组在验证月（t+2）的购买率对比（避免"沉默月 = 结果月"的循环定义）
            res2 = rate_test(out_events, silent_vip, active_vip,
                             f'{base_m}(基期)→{obs_m}(沉默)→{out_m}(验证) 高价值高摩擦')
            row2 = {'训练月': base_m, '沉默月': obs_m, '验证月': out_m,
                    '实验': '高价值高摩擦(沉默)'}
            # 把 rate_test 返回的各列指标并入这一行
            row2.update(res2)
            rate_rows.append(row2)

            # ── 已购人群 LR 基准：规则标记 = 沉默，预测验证月复购 ──
            try:
                base = buyer_baseline(flagged, out_events)
                m = base['metrics']
                # 本臂指标键名与统一汇总列的映射
                keys = _ARM_METRICS_KEYS['已购人群(复购)']
                auc_row = {
                    '训练月': base_m, '沉默月': obs_m, '验证月': out_m, '人群': '已购人群(复购)',
                    '样本': m[keys['样本']], '全体已购用户平均复购率': m[keys['结果率']],
                    '高价值高摩擦用户复购率': m['规则 Top-k 复购率'],
                    '规则人群规模': m['规则人群规模'], '规则 AUC': m['规则 AUC'],
                    'LR AUC (OOF)': m['LR AUC (5折OOF)'], 'LR Top-k 率': m[keys['topk']],
                }
                auc_rows.append(auc_row)
            except ValueError as exc:
                # 该三周样本不足以拟合基准 → 记录后跳过
                print(f'  [已购人群] {base_m}→{obs_m}→{out_m} 跳过: {exc}')
    # 汇总：验证表（率对比）+ 基准表（LR 指标）
    validation_table = pd.DataFrame(rate_rows)
    baseline_table = pd.DataFrame(auc_rows)
    return {'验证表': validation_table, '基准表': baseline_table}


def cohort_migration(panel: pd.DataFrame, base_month: str, months: list | None = None) -> pd.DataFrame:
    """
    队列迁移分析（标签视角）：固定基期月分层 → 用**冻结的基期阈值**逐月重算
    同一批用户的标签，回答“10 月标签在后续月份是保持还是转化”。

    阈值只在基期拟合一次，后续月沿用（冻结）——若每月重拟合，阈值漂移会污染
    标签变化（无法区分“用户变了”还是“尺子变了”）。冻结后跨月标签才可比。

    返回 DataFrame：基期标签 × 月份 × 冻结标签占比（行内归一）。
    '无任何活动' = 当月完全无任何事件的用户。
    """
    if months is None:
        months = MONTHS
    # 一次性按月分片（避免逐月全表扫描），后续都从这个字典取数
    by_month = {}
    for m, g in panel.groupby('month'):
        by_month[m] = g
    base_events = by_month.get(base_month)
    # 防御：基期月必须存在且非空
    if base_events is None or base_events.empty:
        raise ValueError(f'基期月 {base_month} 没有数据。')

    # ── 基期：拟合阈值 + 拟合 E_Score 标准化器 ──
    # 标准化器"冻结"给后续月复用——否则每月重新标准化会让 E_Score 的"尺子"
    # 每月漂移、破坏跨月可比（那就只冻结了阈值数值、没冻结尺子本身）
    feats_base, scalers = build_features(base_events, base_events['event_time'].max(),
                                         return_scalers=True)
    # 基期分层：得到基期标签与本次拟合的全部阈值
    base_seg, thresholds = segment_users(feats_base)
    # 只追踪基期之后的月份
    after = []
    for m in months:
        if m > base_month:
            after.append(m)
    # 基期用户的 user_id（后续所有月份都只追踪这批人）
    base_user_ids = base_seg['user_id'].to_numpy()

    # ── 逐月重算冻结标签：当月特征（基期尺子）+ 冻结阈值 ──
    # label_parts：每个月的 (user_id, month, frozen_label) 片段
    label_parts = []
    for m in after:
        ev = by_month.get(m)
        if ev is None or ev.empty:
            continue
        # 下推过滤：只保留基期用户，后续各月特征聚合计算量大幅减少。
        # E_Score 用冻结标准化器、阈值冻结，过滤不改变任何标签判定（语义等价）。
        ev = ev[ev['user_id'].isin(base_user_ids)]
        if ev.empty:
            continue
        # 当月特征（用基期标准化器 transform，不重拟合）
        feats_m = build_features(ev, ev['event_time'].max(), scalers=scalers)
        # 当月标签（用冻结的基期阈值重算）
        seg_m, _ = segment_users(feats_m, thresholds=thresholds)
        # user_id → 当月标签 的映射（不在当月出现 → NaN）
        lab_map = seg_m.set_index('user_id')['User_Segment']
        # 以"全部基期用户 × 该月"为骨架生成明细表
        tmp = pd.DataFrame({'user_id': base_user_ids, 'month': m})
        # 当月无任何事件的基期用户标记为"无任何活动"
        frozen_label = tmp['user_id'].map(lab_map)
        tmp['frozen_label'] = frozen_label.fillna('无任何活动')
        label_parts.append(tmp)
    # 纵向拼接所有月份的明细
    fl = pd.concat(label_parts, ignore_index=True)

    # ── 汇总：基期标签 × 月份 × 冻结标签 的组合计数 ──
    # 把基期标签（base_seg 里的 User_Segment）合并回明细
    merged = fl.merge(base_seg[['user_id', 'User_Segment']], on='user_id')
    # 按 (基期标签, 月份, 冻结标签) 三维分组计数
    counts = merged.groupby(['User_Segment', 'month', 'frozen_label'], observed=True).size()
    # 计数组改名为列 'n'，再把分组键还原成列
    counts_named = counts.rename('n')
    flg = counts_named.reset_index()
    # 占比 = 每个 (基期标签, 月份) 组内计数 / 该组总数（行内归一）
    group_totals = flg.groupby(['User_Segment', 'month'], observed=True)['n'].transform('sum')
    flg['占比'] = flg['n'] / group_totals

    # 透视成宽表：行 = (基期标签, 月份)，列 = 冻结标签，值 = 占比
    pivoted = flg.pivot_table(index=['User_Segment', 'month'], columns='frozen_label', values='占比')
    # 没出现的冻结标签组合占比补 0，再把分组键还原成列
    pivoted = pivoted.fillna(0)
    pivoted = pivoted.reset_index()
    # 列名 User_Segment 改名为"基期标签"，语义更清晰
    result = pivoted.rename(columns={'User_Segment': '基期标签'})
    return result
