
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (BOOTSTRAP_N, CV_FOLDS, DEDUPE_EVENTS, LR_MAX_ITER,
                    MONTHS, PANEL_FILE, RANDOM_STATE, SESSION_GAP_SECONDS)


def F_and_E_scalers(df: pd.DataFrame) -> dict:


    cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']

    scalers_dict = {}
    # 逐个指标单独拟合一个标准化器
    for col in cols:
        data_series = df[col]
        # 把所有小于 0 的值改为 0
        data_clipped = data_series.clip(lower=0)
        # 对数变换 log(1+x)
        data_logged = np.log1p(data_clipped)
        # 转成 NumPy 数组并重塑为列向量 (行数, 1)：sklearn 要求二维输入
        data_reshaped = data_logged.to_numpy().reshape(-1, 1)
        # 新建 StandardScaler 并拟合
        scaler = StandardScaler()
        scaler.fit(data_reshaped)
        # 把拟合好的标准化器存进字典
        scalers_dict[col] = scaler
    # 返回"列名 -> 标准化器"的映射
    return scalers_dict


def Friction_and_Exploration(df: pd.DataFrame, scalers: dict | None = None) -> pd.DataFrame:
    """
    构建探索度与摩擦指标，指标定义保持业务可解释。

    Exploration: 浏览页数、有效停留时长、会话数的对数标准化均值。
    Friction: 加购但未购买的去重商品数（Cart_Products - Purchased_Products，截断≥0），
              度量“加购了却没买”的购买意图受阻（未购人群首购潜力识别用）。
    已购人群的“高摩擦”不在此定义——它指跨月完全沉默（见 silent_buyer）。

    scalers：可选 dict（见 F_and_E_scalers）。为 None 时当月重新拟合
    """

    # 在副本上操作，避免污染调用方的 DataFrame
    out = df.copy()
    # 参与 Exploration 的三个探索度原始指标列
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
    # Exploration = 三个标准化分量的逐行均值（探索度综合分）
    out['Exploration'] = np.mean(z, axis=0)

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

    # 第 1 步：去重和排序
    if DEDUPE_EVENTS:
        # 配置要求时，先去掉完全重复的事件行。可在config.py中更改
        events = events.drop_duplicates()

    # 按 用户 → 会话 → 时间 排序：让同一会话内的事件在行上相邻，后面才能用相邻行时间差来算会话内有效停留时长
    events = events.sort_values(['user_id', 'user_session', 'event_time']).copy()

    #第 2 步：计算会话内有效停留时长 active_seconds

    # 判断"当前行与上一行是否属于同一个用户"
    same_user = events['user_id'] == events['user_id'].shift()
    # 判断"当前行与上一行是否属于同一个会话"
    same_session_row = events['user_session'] == events['user_session'].shift()

    # 两者同时成立 → 当前行与上一行是同一会话内的相邻事件
    same_session = same_user & same_session_row

    # 计算当前行与上一行的事件时间差，并转化成秒做单位
    time_diff = events['event_time'].diff().dt.total_seconds()

    # 会话内合法时间差范围：(0, SESSION_GAP_SECONDS] 之外一律视为无效
    valid_diff = same_session & time_diff.between(0, SESSION_GAP_SECONDS)

    # 无效位置填 0，首行（无上一行）的 NaN 也补 0 → 会话内有效停留秒数
    in_session_seconds = time_diff.where(valid_diff, 0)
    events['active_seconds'] = in_session_seconds.fillna(0)

    # 第 3 步：提取各种行为事件

    purchase_events = events[events['event_type'] == 'purchase']
    view_events = events[events['event_type'] == 'view']
    cart_events = events[events['event_type'] == 'cart']

    # 第 4 步：将购买数据按照用户聚合，并计算 RFM 指标

    # 最近购买时间（算沉默天数用）、购买次数、总消费，一次 groupby 完成
    purchases = purchase_events.groupby('user_id').agg(
        Last_Purchase=('event_time', 'max'),
        Purchase_Frequency=('event_type', 'size'),
        Total_Spending=('price', 'sum')
    )

    # 第 5 步：计算行为相关指标
    # 5a. 浏览页数：每个用户产生的 view 事件条数
    view_counts = view_events.groupby('user_id').size()
    # view_counts 是一个 Series，索引是user_id,值为每个用户对应的 view_events 行数，
    views = view_counts.rename('Pages_Viewed')

    # 5b. 会话数：每个用户出现过的不同 user_session 个数
    session_counts = events.groupby('user_id')['user_session'].nunique()
    sessions = session_counts.rename('Session_Count')

    # 5c. 有效停留时长：每个用户的 active_seconds 求和
    duration_sums = events.groupby('user_id')['active_seconds'].sum()
    duration = duration_sums.rename('Estimated_Time')

    # 5d. 加购去重商品数：cart 事件里去重后的 product_id 个数
    cart_nunique = cart_events.groupby('user_id')['product_id'].nunique()
    cart_products = cart_nunique.rename('Cart_Products')

    # 5e. 购买去重商品数：购买事件里去重后的 product_id 个数
    purchased_nunique = purchase_events.groupby('user_id')['product_id'].nunique()
    purchased_products = purchased_nunique.rename('Purchased_Products')

    # 第 6 步：合并成宽表，一行一个用户

    features = pd.DataFrame(index=events['user_id'].unique())
    features.index.name = 'user_id'
    # 把各聚合结果按 user_id 索引 join 进来（没有该行为的用户对应列为 NaN）
    parts = [purchases, views, sessions, duration, cart_products, purchased_products]
    joined = features.join(parts)
    # 把 user_id 从索引还原成普通列
    features = joined.reset_index()


    # 第 7 步：处理缺失值与数据类型

    zero_fill_cols = ['Purchase_Frequency', 'Total_Spending', 'Pages_Viewed',
                      'Estimated_Time', 'Cart_Products', 'Purchased_Products']
    # 一次性把所有 NaN 填成 0
    features[zero_fill_cols] = features[zero_fill_cols].fillna(0)

    # 计数类列统一转成整数类型（fillna 之后可能是浮点）
    features['Purchase_Frequency'] = features['Purchase_Frequency'].astype(int)
    features['Pages_Viewed'] = features['Pages_Viewed'].astype(int)
    features['Cart_Products'] = features['Cart_Products'].astype(int)
    features['Purchased_Products'] = features['Purchased_Products'].astype(int)

    # 第 8 步：最近购买距观察期结束的天数（Recency_Days）
    # 观察期结束距面板最早事件日期的天数 + 1（从未购买的用户用）
    fallback_recency = (observation_end - events['event_time'].min()).days + 1
    # 距最近一次购买的时间差（从未购买 → NaN）
    recency_timedelta = observation_end - features['Last_Purchase']
    # 时间差换算成天数
    recency_days = recency_timedelta.dt.days
    # NaN（从未购买）用回退值兜底，再转成整数
    recency_filled = recency_days.fillna(fallback_recency)
    features['Recency_Days'] = recency_filled.astype(int)

    # 第 9 步：去掉中间列并叠加探索度/摩擦力指标
    # Last_Purchase 只用于算 Recency_Days，之后不再需要
    feats = features.drop(columns='Last_Purchase')

    if return_scalers and scalers is None:
        # 调用方要求返回标准化器且未传入：在特征表上拟合 Exploration 的三个标准化器
        scalers = F_and_E_scalers(feats)


    # 叠加 Exploration / Friction / Log_Friction（scalers 传入时复用、不重拟合）
    out = Friction_and_Exploration(feats, scalers=scalers)
    if return_scalers:
        # 队列迁移场景：返回 (特征表, 标准化器)
        return out, scalers
    # 默认场景：只返回特征表
    return out


def gmm(series: pd.Series,
        random_state: int = RANDOM_STATE) -> float:
    """
    两成分 GMM 的后验概率交点；数据太少无法训练则返回中位数。

    """
    # 先把正负无穷替换成 NaN，再丢弃 NaN，得到干净的观测值
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    x = cleaned.to_numpy()
    # 样本太少或取值种类太少时 GMM 拟合不可靠 → 退回中位数
    if len(x) < 50 or np.unique(x).size < 4:
        return float(np.nanmedian(x))

    # 拟合两成分高斯混合：一个成分通常是"多数普通值"，另一个是"少数高值"
    model = GaussianMixture(n_components=2, random_state=random_state, n_init=5)
    # n_init=5：独立运行 5 次，每次用不同的初始化，最后选对数似然最高的那个模型。

    x_2d = x.reshape(-1, 1)
    model.fit(x_2d)
    # 在 [1% 分位, 99% 分位] 之间均匀取 2000 个点作为候选阈值
    low_q = np.quantile(x, 0.01)
    high_q = np.quantile(x, 0.99)
    grid = np.linspace(low_q, high_q, 2000)
    grid_2d = grid.reshape(-1, 1)

    # 使用训练好的模型，预测这些候选阈值点
    posterior = model.predict_proba(grid_2d)
    # 使用predict_proba 返回软概率
    # predict返回的是硬概率

    # 每个网格点上成分 0 的后验概率（离 0.5 越近 = 两个成分概率越接近）
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


def build_user_segment(features: pd.DataFrame,
                       thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    对用户进行精准分层
    以“是否已购”为首要业务边界：
    - 已购用户按价值指数上四分位识别高价值，
    - 未购用户按行为指标识别首购潜力；
    每个指标只在语义匹配的人群内计算。

    未购人群：Exploration 与 Friction 的 GMM 阈值识别高潜力首购。
    已购人群：高价值用户再按 Exploration 细分为深度互动 / 直购；
    “高价值高摩擦用户”（= 跨月完全沉默）由 silent_buyer 结合观察月数据标记。

    参数 thresholds：传入基期拟合的阈值字典（即F_and_E_scalers的输出结果）则冻结使用，为 None 时当月重新拟合。
    """
    # 在副本上操作，避免污染传入的特征表
    df = features.copy()

    # 已购用户子集
    purchased = df['Purchase_Frequency'].gt(0)
    buyer = df[df['Purchase_Frequency'] > 0].copy()

    # 防御：没有任何购买用户时，分层没有意义
    if buyer.empty:
        raise ValueError('建模期没有购买用户，无法完成分层。')

    #  ── 1. 将RFM三个指标压缩为一个价值指标 ──
    spending_log = np.log1p(buyer['Total_Spending'])
    frequency_log = np.log1p(buyer['Purchase_Frequency'])
    recency_log = np.log1p(buyer['Recency_Days'])

    # 价值指数只在已购用户上计算
    value_index = spending_log + frequency_log - recency_log

    # 先整列占位 NaN，再只给已购用户填值（未购用户保持空）
    df['Value_Index'] = np.nan
    df.loc[purchased, 'Value_Index'] = value_index

    # 区分已购用户和未购用户
    df['Base_Segment'] = np.where(purchased, '已购用户', '未购用户')

    # 未购用户的布尔 Series（后续反复使用）
    unpurchased = ~purchased

    # ── 2. 确定分层阈值：当月重新拟合 或 冻结基期阈值 ──
    if thresholds is None:
        # 默认：当月拟合全部阈值

        # 高价值线 = 价值指数的上四分位数（Q3）
        value_cut = float(value_index.quantile(0.75))


        # 未购人群的探索度 / 加购摩擦阈值：各自 GMM 两成分交点
        nonbuyer_exploration_cut = gmm(df.loc[unpurchased, 'Exploration'])
        nonbuyer_friction_cut = gmm(df.loc[unpurchased, 'Log_Friction'])
    else:
        # 冻结阈值：沿用基期拟合的阈值重算标签（队列迁移分析，保证跨月标签可比）
        value_cut = thresholds['vip_value_index_cutoff']
        nonbuyer_exploration_cut = thresholds['nonbuyer_exploration_cutoff']
        nonbuyer_friction_cut = thresholds['nonbuyer_log_friction_cutoff']

    # 高价值用户 = 已购且价值指数 ≥ 高价值线
    vip_mask = purchased & (df['Value_Index'] >= value_cut)
    if thresholds is None:
        # 高价值用户内部再按探索度 GMM 交点细分（当月拟合）
        vip_exploration_cut = gmm(df.loc[vip_mask, 'Exploration'])
    else:
        # 冻结基期的高价值探索度阈值
        vip_exploration_cut = thresholds['vip_exploration_cutoff']

    # ── 3. 生成互斥人群标签：先全部置默认，再逐条覆盖 ──
    # 默认标签：普通浏览用户
    df['User_Segment'] = '普通浏览用户'
    # 未购用户中，探索度与加购摩擦都过阈值 → 高潜力首购用户
    high_potential_mask = (unpurchased
                           & (df['Exploration'] >= nonbuyer_exploration_cut)
                           & (df['Log_Friction'] >= nonbuyer_friction_cut))
    df.loc[high_potential_mask, 'User_Segment'] = '高潜力首购用户'

    # 已购但价值指数未达高价值线 → 常规已购用户
    regular_buyer_mask = purchased & ~vip_mask
    df.loc[regular_buyer_mask, 'User_Segment'] = '常规已购用户'

    # 高价值用户先统一记为"直购用户"（下单干脆、不太深逛）
    df.loc[vip_mask, 'User_Segment'] = '高价值直购用户'

    # 高价值且探索度也过阈值 → 覆盖为"高价值深度互动用户"
    deep_interaction_mask = vip_mask & (df['Exploration'] >= vip_exploration_cut)
    df.loc[deep_interaction_mask, 'User_Segment'] = '高价值深度互动用户'

    # ── 4. 返回：分层结果 + 本次用到的全部阈值（供冻结复用）──
    metadata = {'vip_value_index_cutoff': value_cut,
                'nonbuyer_exploration_cutoff': nonbuyer_exploration_cut,
                'nonbuyer_log_friction_cutoff': nonbuyer_friction_cut,
                'vip_exploration_cutoff': vip_exploration_cut}
    return df, metadata


def silent_buyer(segmented: pd.DataFrame, active_user_ids) -> pd.DataFrame:
    """
    已购人群“跨月沉默”高摩擦标记（流失判定）。

    基期的高价值用户（高价值直购 / 深度互动）在观察月完全无任何事件
    （view / cart / purchase 都没有）→ 标记为“高价值高摩擦用户”（= 沉默/流失）。
    未购用户与其他人群不受影响。

    active_user_ids：观察月里出现过（有任何事件）的 user_id 集合。
    """
    # 在副本上操作，避免影响调用方
    out = segmented.copy()
    # 观察月里有任何事件的 user_id 集合（view / cart / purchase 都算"有活动"）
    active = set(active_user_ids)
    # 防御：观察月一个人都没出现时，无法判断"谁没出现"
    if len(active) == 0:
        raise ValueError('观察月没有活跃用户，无法判定沉默。')
    # 只看基期的高价值人群：高价值直购 / 高价值深度互动
    vip_mask = out['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])
    # 高价值用户若在观察月完全没有事件 → 标记为"高价值高摩擦用户"（沉默/流失）
    silence_mask = vip_mask & ~out['user_id'].isin(active)
    out.loc[silence_mask, 'User_Segment'] = '高价值高摩擦用户'
    return out

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


def rolling_validation_rate_test(purchased_ids: set,
                                 treatment_ids: set,
                                 control_ids: set,
                                 title: str) -> dict:
    """两组验证月购买率对比：卡方检验（Yates 校正）+ Wilson 95% CI。

    purchased_ids：验证月里实际发生购买的用户集合。
    当期望频数含 0（样本过小或某组无事件）导致卡方失效时，回退 Fisher 精确检验。
    """
    # 验证月里实际发生购买的 user_id 集合
    buyers = set(purchased_ids)
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
    解释层：保留 User_Segment / Exploration / Friction 供策略话术使用。
    """
    # 导出名单所需的固定列集合（排序层 + 解释层字段）
    cols = ['user_id', 'User_Segment', 'Exploration', 'Friction', 'Value_Index',
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


def _lr_features_matrix(frame: pd.DataFrame) -> np.ndarray:
    """LR 的统一特征矩阵：log1p(原始字段聚合，共 7 维)。

    只用原始数据里的字段聚合——消费额 / 购买频次 / 最近购买天数，
    以及浏览页数 / 有效停留时长 / 会话数 / 加购去重商品数；
    不使用项目自造的复合指标（Exploration / Friction / Value_Index）。

    注：对未购用户，消费 / 频次恒为 0、近度为观察期兜底常量，标准化后恒为 0，
    对模型没有影响——因此一套特征可以服务所有人群，打分时无需判断人群。
    """
    # 前三个：原始购买字段的聚合
    spending_log = np.log1p(frame['Total_Spending'])
    frequency_log = np.log1p(frame['Purchase_Frequency'])
    recency_log = np.log1p(frame['Recency_Days'])
    # 后四个：原始行为字段的聚合
    pages_log = np.log1p(frame['Pages_Viewed'])
    time_log = np.log1p(frame['Estimated_Time'])
    sessions_log = np.log1p(frame['Session_Count'])
    cart_log = np.log1p(frame['Cart_Products'])
    # 按固定列序堆叠成 (样本数, 7)
    feature_columns = [spending_log, frequency_log, recency_log,
                       pages_log, time_log, sessions_log, cart_log]
    x_matrix = np.column_stack(feature_columns)
    return x_matrix


def nonbuyer_baseline(segmented: pd.DataFrame, nov: pd.DataFrame,
                      random_state: int = RANDOM_STATE) -> dict:
    """
    未购用户首购基准：以 10 月未购用户为样本，用逻辑回归预测 11 月是否首购，
    量化规则分层（高潜力首购）的判别力与边际增量。

    返回 dict：{'metrics': {...}, 'preds': DataFrame(user_id, y, p_lr, rule)}，
    其中 p_lr 为 5 折 OOF 预测概率，供 Notebook 绘制校准/Top-k 提升曲线。

    注：本函数是独立的评估工具，已不在 rolling_validation 主流程中调用。
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
    # 完整特征矩阵（统一 7 维：原始购买字段 + 原始行为字段）
    x_full = _lr_features_matrix(nb)
    # 只含浏览三指标（第 4~6 列，用于对比"仅浏览特征"能到多少 AUC）
    x_vol = x_full[:, 3:6]
    # 规则标记：规则分层命中的"高潜力首购用户"记为 1
    is_potential = nb['User_Segment'].eq('高潜力首购用户')
    rule_int = is_potential.astype(int)
    rule = rule_int.to_numpy()

    # ── 3. 5 折分层交叉验证：产出 OOF 预测 ──
    # 分层 K 折：保证每折正负样本比例与整体一致
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    # LR 模型 = 先 Z 标准化再逻辑回归（统一特征尺度，利于收敛与系数可读）
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    pipe = make_pipeline(scaler, logistic)
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
    # 特征名与系数一一对应（列序与 _lr_features 一致）
    coef_names = ['log1p(Total_Spending)', 'log1p(Purchase_Frequency)', 'log1p(Recency_Days)',
                  'log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)',
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
    检验“高价值高摩擦”（跨月完全沉默，需先经 silent_buyer 标记）规则
    与手工 RFM 价值指数作为排序器的判别力。

    双视角解读：
    - Top-k：谁最可能复购（运营触达视角）；
    - Bottom-k：谁最可能流失（风险预警视角，与沉默规则的语义对应）。

    返回 dict：{'metrics': {...}, 'preds': DataFrame(user_id, y, p_lr, rule, value_index)}，
    其中 p_lr 为 5 折 OOF 复购概率，供 Notebook 绘制校准与 Top/Bottom-k 曲线。

    注：本函数是独立的评估工具，已不在 rolling_validation 主流程中调用。
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
    # 完整特征矩阵（统一 7 维）
    x_full = _lr_features_matrix(buyer)
    # 规则标记：被标记为"高价值高摩擦"（跨月沉默）的用户记为 1
    is_silent = buyer['User_Segment'].eq('高价值高摩擦用户')
    rule_int = is_silent.astype(int)
    rule = rule_int.to_numpy()
    # 手工 RFM 价值指数（作为对比排序器）
    value_score = buyer['Value_Index'].to_numpy()

    # ── 3. 5 折分层交叉验证：产出 OOF 预测 ──
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    # LR 模型 = 先 Z 标准化再逻辑回归（统一特征尺度，利于收敛与系数可读）
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    pipe = make_pipeline(scaler, logistic)
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
    # 特征名与系数一一对应（列序与 _lr_features 一致）
    coef_names = ['log1p(Total_Spending)', 'log1p(Purchase_Frequency)', 'log1p(Recency_Days)',
                  'log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)',
                  'log1p(Cart_Products)']
    for name, value in zip(coef_names, coef_values):
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


def build_month_tables(panel: pd.DataFrame,
                       months: list,
                       cache_dir: Path | str | None = None) -> dict:
    """把事件面板整理成"每月一张表"：返回 {月份: 该月用户表}。

    每张表 = 该月用户特征（12 列）+ 该月"当月独立分层"标签（Value_Index /
    Base_Segment / User_Segment），一个月一个文件（month_YYYY-MM.parquet）。

    cache_dir 为 None 时只算不落盘（合成数据 / 命令行入口用）；
    传入目录时：有缓存文件直接读，没有才现算并落盘（首次约 40 秒/月）。

    注意：这里的分层是"每月独立拟合阈值"的口径；队列迁移用的是"冻结基期阈值 +
    冻结标准化器"的另一套，两者不可混用。
    """
    tables = {}
    for m in months:
        # 该月事件（面板本身带 month 列，按列过滤取一个月）
        events = panel[panel['month'] == m]
        # 防御：面板里没有这个月 → 跳过（下游各自判断"缺月"）
        if events.empty:
            print(f'  {m}: 面板里没有事件，跳过')
            continue

        # 缓存文件路径：cache_dir 为 None 时不落盘
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f'month_{m}.parquet'
            # 有缓存就直接读，跳过特征构建与分层
            if path.exists():
                tables[m] = pd.read_parquet(path)
                print(f'  {m}: 月度表（缓存）')
                continue

        # 该月用户特征（与全项目同一套口径，单一事实来源）
        feats = build_features(events, events['event_time'].max())
        # 叠加该月"当月独立分层"标签（第二个返回值是阈值，这里不需要）
        table, _ = build_user_segment(feats)
        if path is not None:
            # 首次运行：落盘，下次直接读（目录不存在就创建）
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path)
            print(f'  {m}: 月度表（新建，已落盘）')
        else:
            print(f'  {m}: 月度表（新建）')
        tables[m] = table
    return tables


def LR_predict_rank(month_tables: dict,
                    score_month: str,
                    target_user_ids) -> pd.DataFrame:
    """纯工具：预测指定用户在『预测月』的购买概率，并在这些用户内部按概率排名。

    输入
    ----
    month_tables    : {月份: 该月表}（见 build_month_tables：表里同时有特征与标签）
    score_month     : 训练结果月（本轮）
    target_user_ids : 要预测的用户名单（由调用方决定；函数不筛选、不判断人群、不扩大范围）

    行为
    ----
    - 训练：更早月份的「训练行为月 → 训练结果月」样本（该月全体用户，标签 = 次月是否购买），
      特征统一为原始字段聚合（见 _lr_features），一个 LR 模型；
    - 打分：只对 target_user_ids 打分；
    - 排名：名次 = 这些用户内部按概率降序（1 ~ N，并列取最小名次）。

    返回：DataFrame[user_id, Prob, Rank]，按 Rank 升序。
    """
    # 目标用户去重
    seen = set()
    target_ids = []
    for uid in target_user_ids:
        if uid not in seen:
            seen.add(uid)
            target_ids.append(uid)

    if len(target_ids) == 0:
        raise ValueError('target_user_ids 为空，无法打分。')

    # ── 1. 构造训练集：更早月份的「训练行为月 → 训练结果月」全体用户样本 ──
    # X_parts / y_parts：各月份对片段，最后纵向拼接
    X_parts = []
    y_parts = []
    for m in sorted(month_tables):
        # 训练行为月的下一个自然月 = 训练结果月（提供"次月是否购买"标签）
        nm = _next_month(m)
        # 防泄漏护栏：训练结果月不得晚于本轮（否则等于用本轮结果训练）
        if nm > score_month:
            break
        # 该月份对缺表 → 无法构成训练样本，跳过
        if nm not in month_tables:
            continue
        table_m = month_tables[m]
        table_nm = month_tables[nm]
        if table_m.empty or table_nm.empty:
            continue
        # 训练标签：该月用户在训练结果月是否购买
        # （买过 ⇔ 该月表 Purchase_Frequency > 0，与事件表口径等价）
        buyers_nm = set(table_nm.loc[table_nm['Purchase_Frequency'] > 0, 'user_id'])
        is_buyer = table_m['user_id'].isin(buyers_nm)
        y_parts.append(is_buyer.astype(int).to_numpy())
        # 训练样本 = 该月全体用户（不按人群拆分：统一一套特征、一个模型）
        X_parts.append(_lr_features_matrix(table_m))

    # 防御：没有更早的月份可用（10 月是最早的建模月）
    if not X_parts:
        raise ValueError(
            f'{score_month} 无法给出概率分：没有更早月份的训练数据。'
            '（10 月是最早的建模月，没有更早的"训练行为月 → 训练结果月"样本对可用；'
            '因此最早从 11 月开始打分）')
    # 纵向拼接：特征行数总和 = 标签长度
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    # ── 2. 训练：先 Z 标准化再逻辑回归（统一特征尺度，利于收敛与系数可读）──
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    pipe = make_pipeline(scaler, logistic)
    pipe.fit(X_train, y_train)

    # ── 3. 取训练结果月的月度表 ──
    if score_month not in month_tables:
        raise ValueError(f'{score_month} 没有月度表，无法打分。')
    feats_sc = month_tables[score_month]
    # 防御：该月没有数据
    if feats_sc.empty:
        raise ValueError(f'{score_month} 没有数据，无法打分。')

    # ── 4. 只保留目标用户；有用户不在该月 → 直接报错，避免静默少打分 ──
    target_set = set(target_ids)
    is_target = feats_sc['user_id'].isin(target_set)
    sub = feats_sc.loc[is_target]
    missing = len(target_set) - len(sub)
    if missing > 0:
        raise ValueError(f'{missing} 个目标用户在 {score_month} 的月度表里不存在，无法打分。')

    # ── 5. 打分：只对目标用户输出概率 ──
    proba = pipe.predict_proba(_lr_features_matrix(sub))[:, 1]

    # ── 6. 组内排名：名次只在这些目标用户内部计算（1 ~ N）──
    out = pd.DataFrame({'user_id': sub['user_id'].to_numpy(), 'Prob': proba})
    out['Rank'] = out['Prob'].rank(ascending=False, method='min')
    return out.sort_values('Rank').reset_index(drop=True)


def load_panel(path: Path | str = PANEL_FILE,
               months: list | None = None,
               columns: list | None = None) -> pd.DataFrame:
    """读取用户面板（parquet 或 CSV），解析 event_time 并确保 month 列（'YYYY-MM'）存在。
    months：若指定（如 ['2019-10', '2019-11']），parquet 用行过滤只读这些月份；CSV 则读取后过滤。为 None 时读全量。
    columns：若指定（如分析必需列子集），只读这些列，降低 IO 与内存占用
    """
    # 统一成 Path对象，便于判断文件类型
    path = Path(path)
    if path.suffix == '.parquet':
        # 如果是parquet文件（支持筛选读取）
        try:
            # kwargs表示传给 pd.read_parquet 的关键字参数（按需追加）
            kwargs = {}
            if months is not None:
                # 只读指定月份的行
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
        # 如果是csv文件
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


def rolling_validation(month_tables: dict,
                       months: list | None = None) -> dict:
    """
    滚动时间外验证（已购人群为跨月沉默定义）。

    - 实验一（未购人群）：基期月 t 分层 → 观察月 t+1 购买（2 月对）；
    - 实验二（已购人群）：基期月 t 高价值买家 → 观察月 t+1 完全沉默（高价值高摩擦）
       → 验证月 t+2 是否购买（3 月组，避免“沉默月 = 验证月”的循环定义）；
    返回 {'验证表': DataFrame}。
    注：LR 基准评估已移出本函数（需要时单独调用 nonbuyer_baseline / buyer_baseline）。
    month_tables：{月份: 该月表}（见 build_month_tables）。
    """
    if months is None:
        months = MONTHS

    # 收集每个月份对 / 月组的购买率对比结果，最终组成“验证表”
    rate_rows = []
    # 按月滑动：基期月 t → 观察月 t+1（实验二还要用到 t+2）
    for t in range(len(months) - 1):
        base_m = months[t]
        obs_m = months[t + 1]
        base_table = month_tables.get(base_m)
        obs_table = month_tables.get(obs_m)
        # 该月对任一方向缺表 → 跳过（不构成可用实验窗口）
        if base_table is None or obs_table is None or base_table.empty or obs_table.empty:
            print(f'跳过 {base_m}→{obs_m}（无数据）')
            continue

        # ── 实验一（未购人群）：基期特征 → 观察月购买 ──
        # 规则人群 = 高潜力首购用户；对照组 = 普通浏览用户（基期表里已带标签）
        potential = set(base_table.loc[base_table['User_Segment'] == '高潜力首购用户', 'user_id'])
        nonbuyer_control = set(base_table.loc[base_table['User_Segment'] == '普通浏览用户', 'user_id'])
        # 观察月买过的人（买过 ⇔ 该月表 Purchase_Frequency > 0）
        obs_buyers = set(obs_table.loc[obs_table['Purchase_Frequency'] > 0, 'user_id'])
        # 两组在观察月的购买率对比 + 显著性检验
        res1 = rolling_validation_rate_test(obs_buyers, potential, nonbuyer_control,
                                            f'{base_m}→{obs_m} 高潜力首购')
        row1 = {'训练月': base_m, '沉默月': '', '验证月': obs_m, '实验': '高潜力首购'}
        # 把 rolling_validation_rate_test 返回的各列指标并入这一行
        row1.update(res1)
        rate_rows.append(row1)

        # ── 实验二（已购人群·跨月沉默）：基期高价值 → 观察月沉默 → 验证月购买（需第 3 个月）──
        if t + 2 < len(months):
            out_m = months[t + 2]
            out_table = month_tables.get(out_m)
            # 验证月缺表 → 该三周窗口不成立
            if out_table is None or out_table.empty:
                print(f'跳过 {base_m}→{obs_m}→{out_m}（验证月无数据）')
                continue
            # 用观察月的活跃用户标记"跨月完全沉默"的高价值用户（= 高价值高摩擦）
            obs_active = set(obs_table['user_id'])
            flagged = silent_buyer(base_table, obs_active)
            # 沉默组：高价值高摩擦用户；对照组：观察月仍活跃的高价值用户
            silent_vip = set(flagged.loc[flagged['User_Segment'] == '高价值高摩擦用户', 'user_id'])
            active_vip_mask = flagged['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])
            active_vip = set(flagged.loc[active_vip_mask, 'user_id'])
            # 验证月买过的人
            out_buyers = set(out_table.loc[out_table['Purchase_Frequency'] > 0, 'user_id'])
            # 两组在验证月（t+2）的购买率对比（避免"沉默月 = 验证月"的循环定义）
            res2 = rolling_validation_rate_test(out_buyers, silent_vip, active_vip,
                             f'{base_m}(基期)→{obs_m}(沉默)→{out_m}(验证) 高价值高摩擦')
            row2 = {'训练月': base_m, '沉默月': obs_m, '验证月': out_m,
                    '实验': '高价值高摩擦(沉默)'}
            # 把 rolling_validation_rate_test 返回的各列指标并入这一行
            row2.update(res2)
            rate_rows.append(row2)

    # 汇总成一张表：规则标签在各月份对 / 月组上的购买率对比
    validation_table = pd.DataFrame(rate_rows)
    return {'验证表': validation_table}


def cohort_migration(panel: pd.DataFrame, base_month: str,
                     months: list | None = None) -> pd.DataFrame:
    """
    队列迁移分析（标签视角）：固定基期月分层 → 用**冻结的基期阈值**逐月重算
    同一批用户的标签，回答“10 月标签在后续月份是保持还是转化”。

    阈值只在基期拟合一次，后续月沿用（冻结）——若每月重拟合，阈值漂移会污染
    标签变化（无法区分“用户变了”还是“尺子变了”）。冻结后跨月标签才可比。

    返回 DataFrame：基期标签 × 月份 × 冻结标签占比（行内归一）。
    '无任何活动' = 当月完全无任何事件的用户。

    注：本函数要用冻结尺子**重算特征**，必须有原始事件，因此直接收面板，
    不使用"月度表"（月度表是用户级聚合结果，无法还原事件）。
    """
    if months is None:
        months = MONTHS
    # 面板按月切片（只切一次，避免逐月全表扫描）
    by_month = {}
    for m, g in panel.groupby('month'):
        by_month[m] = g
    base_events = by_month.get(base_month)
    # 防御：基期月必须存在且非空
    if base_events is None or base_events.empty:
        raise ValueError(f'基期月 {base_month} 没有数据。')

    # ── 基期：拟合阈值 + 拟合 Exploration 标准化器 ──
    # 标准化器"冻结"给后续月复用——否则每月重新标准化会让 Exploration 的"尺子"
    # 每月漂移、破坏跨月可比（那就只冻结了阈值数值、没冻结尺子本身）
    feats_base, scalers = build_features(base_events, base_events['event_time'].max(),
                                         return_scalers=True)
    # 基期分层：得到基期标签与本次拟合的全部阈值
    base_seg, thresholds = build_user_segment(feats_base)
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
        # 注意：过滤后 observation_end 取的是**子集**的最大事件时间，与"整月口径"算出的
        # Recency_Days 存在细微差异（既有行为，保持原样以免改动历史结论）。
        ev = ev[ev['user_id'].isin(base_user_ids)]
        if ev.empty:
            continue
        # 当月特征（用基期标准化器 transform，不重拟合）
        feats_m = build_features(ev, ev['event_time'].max(), scalers=scalers)
        # 当月标签（用冻结的基期阈值重算）
        seg_m, _ = build_user_segment(feats_m, thresholds=thresholds)
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
