"""用户价值与购买潜力分层（UserValue&Potential）分析函数库。

未购用户 → 首购潜力识别（探索度 + 加购未买 + 首购概率分）；
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
    cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']
    return {col: StandardScaler().fit(np.log1p(df[col].clip(lower=0)).to_numpy().reshape(-1, 1))
            for col in cols}


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
    out = df.copy()
    cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']
    z = []
    for col in cols:
        values = np.log1p(out[col].clip(lower=0)).to_numpy().reshape(-1, 1)
        if scalers is None:
            z.append(StandardScaler().fit_transform(values).ravel())
        else:
            z.append(scalers[col].transform(values).ravel())
    out['E_Score'] = np.mean(z, axis=0)
    out['Friction'] = (out['Cart_Products'] - out['Purchased_Products']).clip(lower=0)
    out['Log_Friction'] = np.log1p(out['Friction'])
    return out


def build_features(events: pd.DataFrame, observation_end: pd.Timestamp,
                   scalers: dict | None = None, return_scalers: bool = False):
    """按会话计算有效停留时长，聚合为用户特征（含加购/购买的去重商品数）。

    scalers / return_scalers：E_Score 标准化器的冻结复用（队列迁移分析用）。
    基期调用传 return_scalers=True 取回标准化器，后续月传 scalers=基期标准化器，
    保证 E_Score 的 Z 标准化参数跨月不变（否则"冻结阈值"只冻结了阈值、
    没冻结"尺子"本身，跨月标签不可比）。默认（两者均不传）行为与旧版一致：
    返回单表、每月重新拟合标准化。
    """
    if DEDUPE_EVENTS:
        events = events.drop_duplicates()
    events = events.sort_values(['user_id', 'user_session', 'event_time']).copy()
    # 向量化计算会话内相邻事件时间差：排序后仅同一 user_id 且同一 user_session 的
    # 相邻行计入（等价于 groupby(['user_id','user_session']).diff()，但避免维护
    # 数十万分组对象，每月数百万行下快数倍）
    same_session = (events['user_id'].eq(events['user_id'].shift())
                    & events['user_session'].eq(events['user_session'].shift()))
    time_diff = events['event_time'].diff().dt.total_seconds()
    events['active_seconds'] = time_diff.where(
        same_session & time_diff.between(0, SESSION_GAP_SECONDS), 0).fillna(0)
    # 布尔掩码一次切分事件类型（避免多次 query 全表扫描）
    is_purchase = events['event_type'].eq('purchase')
    is_view = events['event_type'].eq('view')
    is_cart = events['event_type'].eq('cart')
    purchase_ev = events.loc[is_purchase]
    purchases = purchase_ev.groupby('user_id').agg(
        Last_Purchase=('event_time', 'max'), Purchase_Frequency=('event_type', 'size'),
        Total_Spending=('price', 'sum'))
    views = events.loc[is_view].groupby('user_id').size().rename('Pages_Viewed')
    sessions = events.groupby('user_id')['user_session'].nunique().rename('Session_Count')
    duration = events.groupby('user_id')['active_seconds'].sum().rename('Estimated_Time')
    cart_products = events.loc[is_cart].groupby('user_id')['product_id'].nunique().rename('Cart_Products')
    purchased_products = purchase_ev.groupby('user_id')['product_id'].nunique().rename('Purchased_Products')
    features = pd.DataFrame(index=events['user_id'].unique())
    features.index.name = 'user_id'
    features = features.join([purchases, views, sessions, duration, cart_products, purchased_products]).reset_index()
    features[['Purchase_Frequency', 'Total_Spending', 'Pages_Viewed', 'Estimated_Time',
              'Cart_Products', 'Purchased_Products']] = (
        features[['Purchase_Frequency', 'Total_Spending', 'Pages_Viewed', 'Estimated_Time',
                  'Cart_Products', 'Purchased_Products']].fillna(0))
    features['Purchase_Frequency'] = features['Purchase_Frequency'].astype(int)
    features['Pages_Viewed'] = features['Pages_Viewed'].astype(int)
    features['Cart_Products'] = features['Cart_Products'].astype(int)
    features['Purchased_Products'] = features['Purchased_Products'].astype(int)
    fallback_recency = (observation_end - events['event_time'].min()).days + 1
    features['Recency_Days'] = (observation_end - features['Last_Purchase']).dt.days.fillna(fallback_recency).astype(int)
    feats = features.drop(columns='Last_Purchase')
    if return_scalers and scalers is None:
        scalers = fit_engagement_scalers(feats)
    out = compute_engagement_metrics(feats, scalers=scalers)
    return (out, scalers) if return_scalers else out


def gmm_intersection_threshold(series: pd.Series, random_state: int = RANDOM_STATE) -> float:
    """
    两成分 GMM 的后验概率交点；拟合不稳定时退回中位数。

    阈值取两成分后验概率相等的位置，综合考虑分量方差与权重。
    零膨胀（大量 0 值，如“加购未买”计数）场景下，若交点落在 ≤0 且序列本身
    非负，则取最小正值作为阈值，语义即“有加购即高摩擦”。
    """
    x = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(x) < 50 or np.unique(x).size < 4:
        return float(np.nanmedian(x))
    model = GaussianMixture(n_components=2, random_state=random_state, n_init=5).fit(x.reshape(-1, 1))
    grid = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), 2000)
    posterior_gap = np.abs(model.predict_proba(grid.reshape(-1, 1))[:, 0] - 0.5)
    cut = float(grid[np.argmin(posterior_gap)])
    positive = x[x > 0]
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
    df = features.copy()
    purchased = df['Purchase_Frequency'].gt(0)
    buyer = df.loc[purchased].copy()
    if buyer.empty:
        raise ValueError('建模期没有购买用户，无法完成分层。')

    value_index = (
        np.log1p(buyer['Total_Spending'])
        + np.log1p(buyer['Purchase_Frequency'])
        - np.log1p(buyer['Recency_Days'])
    )
    df['Value_Index'] = np.nan
    df.loc[purchased, 'Value_Index'] = value_index
    df['Base_Segment'] = np.where(purchased, '已购用户', '未购用户')

    nonbuyer_mask = ~purchased

    if thresholds is None:
        # 默认：当月拟合全部阈值
        value_cut = float(value_index.quantile(0.75))
        nonbuyer_escore_cut = gmm_intersection_threshold(df.loc[nonbuyer_mask, 'E_Score'])
        nonbuyer_friction_cut = gmm_intersection_threshold(df.loc[nonbuyer_mask, 'Log_Friction'])
    else:
        # 冻结阈值：沿用基期拟合的阈值重算标签（队列迁移分析用，保证跨月标签可比）
        value_cut = thresholds['vip_value_index_cutoff']
        nonbuyer_escore_cut = thresholds['nonbuyer_e_score_cutoff']
        nonbuyer_friction_cut = thresholds['nonbuyer_log_friction_cutoff']

    vip_mask = purchased & (df['Value_Index'] >= value_cut)
    if thresholds is None:
        vip_escore_cut = gmm_intersection_threshold(df.loc[vip_mask, 'E_Score'])
    else:
        vip_escore_cut = thresholds['vip_e_score_cutoff']

    df['User_Segment'] = '普通浏览用户'
    df.loc[nonbuyer_mask & (df['E_Score'] >= nonbuyer_escore_cut)
           & (df['Log_Friction'] >= nonbuyer_friction_cut), 'User_Segment'] = '高潜力首购用户'

    df.loc[purchased & ~vip_mask, 'User_Segment'] = '常规已购用户'
    df.loc[vip_mask, 'User_Segment'] = '高价值直购用户'
    df.loc[vip_mask & (df['E_Score'] >= vip_escore_cut), 'User_Segment'] = '高价值深度互动用户'

    metadata = {'vip_value_index_cutoff': value_cut, 'nonbuyer_e_score_cutoff': nonbuyer_escore_cut,
                'nonbuyer_log_friction_cutoff': nonbuyer_friction_cut, 'vip_e_score_cutoff': vip_escore_cut}
    return df, metadata


def flag_buyer_silence(segmented: pd.DataFrame, obs_events: pd.DataFrame) -> pd.DataFrame:
    """
    已购人群“跨月沉默”高摩擦标记（流失判定）。

    基期的高价值用户（高价值直购 / 深度互动）在观察月完全无任何事件
    （view / cart / purchase 都没有）→ 标记为“高价值高摩擦用户”（= 沉默/流失）。
    未购用户与其他人群不受影响。
    """
    out = segmented.copy()
    if obs_events.empty:
        raise ValueError('观察月没有事件数据，无法判定沉默。')
    obs_users = set(obs_events['user_id'])
    vip_mask = out['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])
    out.loc[vip_mask & ~out['user_id'].isin(obs_users), 'User_Segment'] = '高价值高摩擦用户'
    return out


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    返回简历/汇报可直接引用的人群概览
    """
    return (df.groupby('User_Segment', as_index=False)
              .agg(用户数=('user_id', 'size'), 平均消费=('Total_Spending', 'mean'),
                   平均购买频次=('Purchase_Frequency', 'mean'), 平均探索度=('E_Score', 'mean'),
                   平均摩擦力=('Friction', 'mean'))
              .assign(用户占比=lambda x: x['用户数'] / x['用户数'].sum())
              .sort_values('用户数', ascending=False))


def proportion_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% CI，避免小比例下的 Wald 区间失真。"""
    if total == 0:
        return np.nan, np.nan
    z = 1.96
    p = successes / total
    d = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / d
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return centre - half, centre + half


def rate_test(nov: pd.DataFrame, treatment_ids: set, control_ids: set, title: str) -> dict:
    """两组 11 月购买率对比：卡方检验（Yates 校正）+ Wilson 95% CI。

    当期望频数含 0（样本过小或某组无事件）导致卡方失效时，回退 Fisher 精确检验。
    """
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    a, b = len(treatment_ids & buyers), len(control_ids & buyers)
    n_a, n_b = len(treatment_ids), len(control_ids)
    if n_a == 0 or n_b == 0:
        # 空组无法计算比率与检验（滚动验证中某月可能没有该人群），返回 NaN
        return {'目标组': title.split('：')[0], '目标人数': n_a, '目标购买率': np.nan,
                '对照人数': n_b, '对照购买率': np.nan, '购买率差': np.nan, 'p值': np.nan}
    rate_a, rate_b = a / n_a, b / n_b
    table = [[a, n_a - a], [b, n_b - b]]
    try:
        _, pvalue, _, _ = chi2_contingency(table, correction=True)
    except ValueError:
        _, pvalue = fisher_exact(table, alternative='two-sided')
    lo_a, hi_a = proportion_ci(a, n_a)
    lo_b, hi_b = proportion_ci(b, n_b)
    print(f'\n{title}')
    print(f'  目标组: {rate_a:.2%} ({a}/{n_a}), 95% CI [{lo_a:.2%}, {hi_a:.2%}]')
    print(f'  对照组: {rate_b:.2%} ({b}/{n_b}), 95% CI [{lo_b:.2%}, {hi_b:.2%}]')
    print(f'  购买率差: {rate_a-rate_b:+.2%}; 卡方检验 p={pvalue:.3g}')
    return {'目标组': title.split('：')[0], '目标人数': n_a, '目标购买率': rate_a,
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
    cols = ['user_id', 'User_Segment', 'E_Score', 'Friction', 'Value_Index',
            'First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag',
            'Repurchase_Prob', 'Repurchase_Rank']
    if segments is None:
        tracking = segmented.loc[:, cols].copy()
    else:
        tracking = segmented.loc[segmented['User_Segment'].isin(segments), cols]
    if path is not None:
        tracking.to_csv(path, index=False, encoding='utf-8-sig')
    return tracking


def _nonbuyer_lr_features(nb: pd.DataFrame) -> np.ndarray:
    """未购用户的 LR 特征矩阵：log1p(页数 / 停留时长 / 会话数 / 加购商品数)。"""
    return np.column_stack([
        np.log1p(nb['Pages_Viewed']), np.log1p(nb['Estimated_Time']),
        np.log1p(nb['Session_Count']), np.log1p(nb['Cart_Products']),
    ])


def _buyer_lr_features(b: pd.DataFrame) -> np.ndarray:
    """已购用户的 LR 特征矩阵：log1p(RFM + 浏览/加购行为共 7 维)。"""
    return np.column_stack([
        np.log1p(b['Total_Spending']), np.log1p(b['Purchase_Frequency']),
        np.log1p(b['Recency_Days']), np.log1p(b['Pages_Viewed']),
        np.log1p(b['Estimated_Time']), np.log1p(b['Session_Count']),
        np.log1p(b['Cart_Products']),
    ])


_BUYER_FEATURE_NAMES = ['log1p(Total_Spending)', 'log1p(Purchase_Frequency)', 'log1p(Recency_Days)',
                        'log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)',
                        'log1p(Cart_Products)']


def _lr_pipeline() -> 'object':
    """标准化的 (StandardScaler + LogisticRegression) 流水线。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE))


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

    nb = segmented[segmented['Purchase_Frequency'].eq(0)].copy()
    if nb.empty:
        raise ValueError('没有未购用户，无法运行基准。')
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    y = nb['user_id'].isin(buyers).astype(int).to_numpy()

    x_full = _nonbuyer_lr_features(nb)
    x_vol = x_full[:, :3]
    rule = nb['User_Segment'].eq('高潜力首购用户').astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    pipe = _lr_pipeline()
    auc_vol = roc_auc_score(y, cross_val_predict(pipe, x_vol, y, cv=cv, method='predict_proba')[:, 1])
    p_lr = cross_val_predict(pipe, x_full, y, cv=cv, method='predict_proba')[:, 1]
    auc_lr = roc_auc_score(y, p_lr)
    brier_lr = brier_score_loss(y, p_lr)
    auc_lr_rule = roc_auc_score(
        y, cross_val_predict(pipe, np.column_stack([x_full, rule]), y, cv=cv, method='predict_proba')[:, 1])
    auc_rule = roc_auc_score(y, rule)
    brier_rule = brier_score_loss(y, rule)

    k = int(rule.sum())
    order = np.argsort(-p_lr)
    top_lr_rate = float(y[order[:k]].mean())
    top_rule_rate = float(y[rule == 1].mean())

    pipe.fit(x_full, y)
    coef = {k: float(v) for k, v in zip(
        ['log1p(Pages_Viewed)', 'log1p(Estimated_Time)', 'log1p(Session_Count)', 'log1p(Cart_Products)'],
        pipe.named_steps['logisticregression'].coef_[0])}

    rng = np.random.default_rng(random_state)
    diffs = [roc_auc_score(y[idx], p_lr[idx]) - roc_auc_score(y[idx], rule[idx])
             for idx in (rng.choice(len(y), size=len(y), replace=True) for _ in range(BOOTSTRAP_N))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])

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

    buyer = segmented[segmented['Purchase_Frequency'].gt(0)].copy()
    if buyer.empty:
        raise ValueError('没有已购用户，无法运行复购基准。')
    buyers_nov = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    y = buyer['user_id'].isin(buyers_nov).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有复购样本，无法拟合复购基准。')

    x_full = _buyer_lr_features(buyer)
    rule = buyer['User_Segment'].eq('高价值高摩擦用户').astype(int).to_numpy()
    value_score = buyer['Value_Index'].to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    pipe = _lr_pipeline()
    p_lr = cross_val_predict(pipe, x_full, y, cv=cv, method='predict_proba')[:, 1]
    auc_lr = roc_auc_score(y, p_lr)
    brier_lr = brier_score_loss(y, p_lr)
    auc_lr_rule = roc_auc_score(
        y, cross_val_predict(pipe, np.column_stack([x_full, rule]), y, cv=cv, method='predict_proba')[:, 1])
    auc_rule = roc_auc_score(y, rule)
    brier_rule = brier_score_loss(y, rule)
    auc_value = roc_auc_score(y, value_score)

    k = int(rule.sum())
    order = np.argsort(-p_lr)
    top_lr_rate = float(y[order[:k]].mean())
    bottom_lr_rate = float(y[order[-k:]].mean())
    top_rule_rate = float(y[rule == 1].mean())

    pipe.fit(x_full, y)
    coef = {k: float(v) for k, v in zip(_BUYER_FEATURE_NAMES,
                                        pipe.named_steps['logisticregression'].coef_[0])}

    rng = np.random.default_rng(random_state)
    diffs = [roc_auc_score(y[idx], p_lr[idx]) - roc_auc_score(y[idx], rule[idx])
             for idx in (rng.choice(len(y), size=len(y), replace=True) for _ in range(BOOTSTRAP_N))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])

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
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    if top_ratio is None:
        top_ratio = POOL_TOP_RATIO
    out = segmented.copy()
    nb_mask = out['Purchase_Frequency'].eq(0)
    nb = out.loc[nb_mask]
    if nb.empty:
        raise ValueError('没有未购用户，无法打分。')
    buyers = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    y = nb['user_id'].isin(buyers).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有购买样本，无法拟合首购概率分。')
    pipe = _lr_pipeline()
    pipe.fit(_nonbuyer_lr_features(nb), y)
    proba = pipe.predict_proba(_nonbuyer_lr_features(nb))[:, 1]

    out['First_Purchase_Prob'] = np.nan
    out['First_Purchase_Rank'] = np.nan
    out['TopK_Flag'] = np.nan
    out.loc[nb_mask, 'First_Purchase_Prob'] = proba
    # 只在候选池内排序：池 = 未购用户中规则认可的人群
    pool_mask = nb_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        ranks = pd.Series(proba, index=nb.index).loc[pool_idx].rank(ascending=False, method='min')
        k = int(np.ceil(len(pool_idx) * top_ratio))
        out.loc[pool_idx, 'First_Purchase_Rank'] = ranks
        out.loc[pool_idx, 'TopK_Flag'] = (ranks <= k).astype(int)
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
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    out = segmented.copy()
    buyer_mask = out['Purchase_Frequency'].gt(0)
    buyer = out.loc[buyer_mask]
    if buyer.empty:
        raise ValueError('没有已购用户，无法打分。')
    buyers_nov = set(nov.loc[nov['event_type'].eq('purchase'), 'user_id'])
    y = buyer['user_id'].isin(buyers_nov).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError('验证期没有复购样本，无法拟合复购概率分。')
    pipe = _lr_pipeline()
    pipe.fit(_buyer_lr_features(buyer), y)
    proba = pipe.predict_proba(_buyer_lr_features(buyer))[:, 1]

    out['Repurchase_Prob'] = np.nan
    out['Repurchase_Rank'] = np.nan
    out.loc[buyer_mask, 'Repurchase_Prob'] = proba
    # 只在候选池内排序：池 = 已购用户中规则认可的人群
    pool_mask = buyer_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        ranks = pd.Series(proba, index=buyer.index).loc[pool_idx].rank(ascending=False, method='min')
        out.loc[pool_idx, 'Repurchase_Rank'] = ranks
    return out


def _next_month(m: str) -> str:
    """'2019-10' -> '2019-11'（月份字符串递增）。"""
    y, mo = int(m[:4]), int(m[5:7])
    return f'{y + 1:04d}-01' if mo == 12 else f'{y:04d}-{mo + 1:02d}'


def _history_train_samples(panel: pd.DataFrame, score_month: str, buyer: bool,
                           cache: dict | None = None):
    """构建打分月之前的所有 (特征月 → 次月标签) 训练样本（未购/已购人群），返回 (X, y)。

    训练样本对象 = m 月用户（m < score_month），打分对象 = score_month 用户——
    模型从未见过打分对象的结果（次月购买），严格避免"偷看答案"。
    cache：可选 dict（month → 特征表 / '_by_month' → 按月分片），跨打分月复用。
    """
    if cache is not None and '_by_month' in cache:
        by_month = cache['_by_month']
    else:
        # 一次性按月分片，避免对 2000 万+ 行的 panel 反复全表布尔过滤（主要耗时点）
        by_month = {m: g for m, g in panel.groupby('month')}
        if cache is not None:
            cache['_by_month'] = by_month
    months = sorted(by_month)
    X_parts, y_parts = [], []
    for m in months:
        nm = _next_month(m)
        if nm > score_month:
            break
        ev_m = by_month.get(m)
        ev_nm = by_month.get(nm)
        if ev_m is None or ev_nm is None or ev_m.empty or ev_nm.empty:
            continue
        if cache is not None and m in cache:
            feats = cache[m]
        else:
            feats = build_features(ev_m, ev_m['event_time'].max())
            if cache is not None:
                cache[m] = feats
        buyers_nm = set(ev_nm.loc[ev_nm['event_type'].eq('purchase'), 'user_id'])
        if buyer:
            sub = feats[feats['Purchase_Frequency'].gt(0)].copy()
            y = sub['user_id'].isin(buyers_nm).astype(int).to_numpy()
            X_parts.append(_buyer_lr_features(sub))
        else:
            sub = feats[feats['Purchase_Frequency'].eq(0)].copy()
            y = sub['user_id'].isin(buyers_nm).astype(int).to_numpy()
            X_parts.append(_nonbuyer_lr_features(sub))
        y_parts.append(y)
    if not X_parts:
        raise ValueError(
            f'{score_month} 无法给出概率分：没有更早月份的训练数据。'
            '（10 月是最早建模月，不需要选人；LR 需要"上上个月/上个月"的历史行为训练，'
            '打分月最早从 11 月开始）')
    return np.vstack(X_parts), np.concatenate(y_parts)


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
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    if top_ratio is None:
        top_ratio = POOL_TOP_RATIO
    X_train, y_train = _history_train_samples(panel, score_month, buyer=False, cache=cache)
    pipe = _lr_pipeline()
    pipe.fit(X_train, y_train)

    if cache is not None and '_by_month' in cache:
        by_month = cache['_by_month']
    else:
        by_month = {m: g for m, g in panel.groupby('month')}
        if cache is not None:
            cache['_by_month'] = by_month
    ev_sc = by_month.get(score_month)
    if cache is not None and score_month in cache:
        feats_sc = cache[score_month]
    else:
        feats_sc = build_features(ev_sc, ev_sc['event_time'].max())
        if cache is not None:
            cache[score_month] = feats_sc
    seg_sc, _ = segment_users(feats_sc)
    out = seg_sc.copy()
    nb_mask = out['Purchase_Frequency'].eq(0)
    nb = out.loc[nb_mask]
    if nb.empty:
        raise ValueError(f'{score_month} 没有未购用户，无法打分。')
    proba = pipe.predict_proba(_nonbuyer_lr_features(nb))[:, 1]

    out['First_Purchase_Prob'] = np.nan
    out['First_Purchase_Rank'] = np.nan
    out['TopK_Flag'] = np.nan
    out.loc[nb_mask, 'First_Purchase_Prob'] = proba
    pool_mask = nb_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        ranks = pd.Series(proba, index=nb.index).loc[pool_idx].rank(ascending=False, method='min')
        k = int(np.ceil(len(pool_idx) * top_ratio))
        out.loc[pool_idx, 'First_Purchase_Rank'] = ranks
        out.loc[pool_idx, 'TopK_Flag'] = (ranks <= k).astype(int)
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
    if pool_segments is None:
        pool_segments = POOL_SEGMENTS
    X_train, y_train = _history_train_samples(panel, score_month, buyer=True, cache=cache)
    pipe = _lr_pipeline()
    pipe.fit(X_train, y_train)

    if segmented is None:
        if cache is not None and '_by_month' in cache:
            by_month = cache['_by_month']
        else:
            by_month = {m: g for m, g in panel.groupby('month')}
            if cache is not None:
                cache['_by_month'] = by_month
        ev_sc = by_month.get(score_month)
        if cache is not None and score_month in cache:
            feats_sc = cache[score_month]
        else:
            feats_sc = build_features(ev_sc, ev_sc['event_time'].max())
            if cache is not None:
                cache[score_month] = feats_sc
        seg_sc, _ = segment_users(feats_sc)
        out = seg_sc.copy()
    else:
        out = segmented.copy()
    buyer_mask = out['Purchase_Frequency'].gt(0)
    buyer = out.loc[buyer_mask]
    if buyer.empty:
        raise ValueError(f'{score_month} 没有已购用户，无法打分。')
    proba = pipe.predict_proba(_buyer_lr_features(buyer))[:, 1]

    out['Repurchase_Prob'] = np.nan
    out['Repurchase_Rank'] = np.nan
    out.loc[buyer_mask, 'Repurchase_Prob'] = proba
    pool_mask = buyer_mask & out['User_Segment'].isin(pool_segments)
    pool_idx = out.loc[pool_mask].index
    if len(pool_idx) > 0:
        ranks = pd.Series(proba, index=buyer.index).loc[pool_idx].rank(ascending=False, method='min')
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
    path = Path(path)
    if path.suffix == '.parquet':
        try:
            kwargs = {}
            if months is not None:
                kwargs['filters'] = [('month', 'in', months)]
            if columns is not None:
                kwargs['columns'] = columns
            panel = pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            raise ImportError(
                f'读取面板 {path.name} 需要 parquet 引擎（pyarrow）。请先安装依赖：'
                f'pip install pyarrow  （或 pip install -r requirements.txt）') from exc
    else:
        panel = pd.read_csv(path, parse_dates=['event_time'])
        if columns is not None:
            panel = panel[[c for c in columns if c in panel.columns]]
        if months is not None:
            panel = panel[panel['event_time'].astype(str).str[:7].isin(months)]
    if 'event_time' in panel.columns:
        panel['event_time'] = pd.to_datetime(panel['event_time'])
    if 'month' not in panel.columns:
        panel['month'] = panel['event_time'].astype(str).str[:7]
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
    # 一次性按月份分片，避免后续每对月份都全表扫描 panel（数据 2000 万+ 行）
    by_month = {m: g for m, g in panel.groupby('month')}
    rate_rows, auc_rows = [], []
    for t in range(len(months) - 1):
        base_m, obs_m = months[t], months[t + 1]
        base_events = by_month.get(base_m)
        obs_events = by_month.get(obs_m)
        if base_events is None or obs_events is None or base_events.empty or obs_events.empty:
            print(f'跳过 {base_m}→{obs_m}（无数据）')
            continue
        features = build_features(base_events, base_events['event_time'].max())
        segmented, _ = segment_users(features)

        # ── 实验一（未购人群）：基期特征 → 观察月购买 ──
        potential = set(segmented.loc[segmented['User_Segment'].eq('高潜力首购用户'), 'user_id'])
        nonbuyer_control = set(segmented.loc[segmented['User_Segment'].eq('普通浏览用户'), 'user_id'])
        res1 = rate_test(obs_events, potential, nonbuyer_control, f'{base_m}→{obs_m} 高潜力首购')
        rate_rows.append({'训练月': base_m, '沉默月': '', '验证月': obs_m, '实验': '高潜力首购', **res1})

        # ── 未购人群 LR 基准 ──
        try:
            base = nonbuyer_baseline(segmented, obs_events)
            m = base['metrics']
            keys = _ARM_METRICS_KEYS['未购人群(首购)']
            auc_rows.append({
                '训练月': base_m, '沉默月': '', '验证月': obs_m, '人群': '未购人群(首购)',
                '样本': m[keys['样本']], '全体未购用户平均首购率': m[keys['结果率']],
                '高潜力首购用户购买率': m['规则 Top-k 购买率'],
                '规则人群规模': m['规则人群规模'], '规则 AUC': m['规则 AUC'],
                'LR AUC (OOF)': m['LR AUC (5折OOF)'], 'LR Top-k 率': m[keys['topk']],
            })
        except ValueError as exc:
            print(f'  [未购人群] {base_m}→{obs_m} 跳过: {exc}')

        # ── 实验二（已购人群·跨月沉默）：基期高价值 → 观察月沉默 → 验证月购买（需第 3 个月）──
        if t + 2 < len(months):
            out_m = months[t + 2]
            out_events = by_month.get(out_m)
            if out_events is None or out_events.empty:
                print(f'跳过 {base_m}→{obs_m}→{out_m}（验证月无数据）')
                continue
            flagged = flag_buyer_silence(segmented, obs_events)
            silent_vip = set(flagged.loc[flagged['User_Segment'].eq('高价值高摩擦用户'), 'user_id'])
            active_vip = set(flagged.loc[flagged['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户']), 'user_id'])
            res2 = rate_test(out_events, silent_vip, active_vip,
                             f'{base_m}(基期)→{obs_m}(沉默)→{out_m}(验证) 高价值高摩擦')
            rate_rows.append({'训练月': base_m, '沉默月': obs_m, '验证月': out_m,
                              '实验': '高价值高摩擦(沉默)', **res2})

            # ── 已购人群 LR 基准：规则标记=沉默，预测验证月复购 ──
            try:
                base = buyer_baseline(flagged, out_events)
                m = base['metrics']
                keys = _ARM_METRICS_KEYS['已购人群(复购)']
                auc_rows.append({
                    '训练月': base_m, '沉默月': obs_m, '验证月': out_m, '人群': '已购人群(复购)',
                    '样本': m[keys['样本']], '全体已购用户平均复购率': m[keys['结果率']],
                    '高价值高摩擦用户复购率': m['规则 Top-k 复购率'],
                    '规则人群规模': m['规则人群规模'], '规则 AUC': m['规则 AUC'],
                    'LR AUC (OOF)': m['LR AUC (5折OOF)'], 'LR Top-k 率': m[keys['topk']],
                })
            except ValueError as exc:
                print(f'  [已购人群] {base_m}→{obs_m}→{out_m} 跳过: {exc}')
    return {'验证表': pd.DataFrame(rate_rows), '基准表': pd.DataFrame(auc_rows)}


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
    # 一次性按月份分片（避免逐月全表扫描），并只保留基期及之后月份
    by_month = {m: g for m, g in panel.groupby('month')}
    base_events = by_month.get(base_month)
    if base_events is None or base_events.empty:
        raise ValueError(f'基期月 {base_month} 没有数据。')
    # 基期：拟合阈值 + 拟合 E_Score 标准化器（后者冻结给后续月复用，
    # 否则每月重新标准化会让 E_Score 的"尺子"每月漂移、破坏跨月可比）
    feats_base, scalers = build_features(base_events, base_events['event_time'].max(),
                                         return_scalers=True)
    base_seg, thresholds = segment_users(feats_base)
    after = [m for m in months if m > base_month]
    base_user_ids = base_seg['user_id'].to_numpy()

    # 逐月：当月特征（用基期标准化器）+ 冻结的基期阈值 → 重算标签（当月无任何事件的用户记为"无任何活动"）
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
        feats_m = build_features(ev, ev['event_time'].max(), scalers=scalers)
        seg_m, _ = segment_users(feats_m, thresholds=thresholds)
        lab_map = seg_m.set_index('user_id')['User_Segment']
        tmp = pd.DataFrame({'user_id': base_user_ids, 'month': m})
        tmp['frozen_label'] = tmp['user_id'].map(lab_map).fillna('无任何活动')
        label_parts.append(tmp)
    fl = pd.concat(label_parts, ignore_index=True)
    flg = (fl.merge(base_seg[['user_id', 'User_Segment']], on='user_id')
             .groupby(['User_Segment', 'month', 'frozen_label'], observed=True).size()
             .rename('n').reset_index())
    flg['占比'] = flg.groupby(['User_Segment', 'month'], observed=True)['n'].transform(lambda s: s / s.sum())
    return (flg.pivot_table(index=['User_Segment', 'month'], columns='frozen_label', values='占比')
                .fillna(0).reset_index()
                .rename(columns={'User_Segment': '基期标签'}))
