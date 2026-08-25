# -*- coding: utf-8 -*-
"""核心逻辑单元测试（合成数据，不依赖原始 CSV）。

运行：在项目根目录执行  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analysis import (build_features, buyer_baseline, cohort_migration,
                      compute_engagement_metrics, export_tracking, flag_buyer_silence,
                      gmm_intersection_threshold, load_panel, rate_test, rolling_validation,
                      score_buyers, score_nonbuyers, segment_users)


def make_events(n_users: int = 6) -> pd.DataFrame:
    """合成事件表：约 1/3 用户未购（其中部分加购未买），其余已购。"""
    rows = []
    for uid in range(n_users):
        t = pd.Timestamp('2019-10-01 10:00:00') + pd.Timedelta(minutes=uid * 5)
        session = f's{uid}'
        if uid % 3 == 0:
            # 未购用户：只浏览；uid % 6 == 0 的再加购未买（高潜力候选）
            rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                         'event_type': 'view', 'product_id': 101, 'price': 0.0})
            t += pd.Timedelta(minutes=1)
            rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                         'event_type': 'view', 'product_id': 102, 'price': 0.0})
            if uid % 6 == 0:
                t += pd.Timedelta(minutes=1)
                rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                             'event_type': 'cart', 'product_id': 101, 'price': 10.0})
        else:
            for p, etype in [(101, 'view'), (102, 'view'), (101, 'cart'), (101, 'purchase')]:
                rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                             'event_type': etype, 'product_id': p, 'price': 10.0})
                t += pd.Timedelta(minutes=1)
    return pd.DataFrame(rows)


class TestGmmIntersectionThreshold(unittest.TestCase):
    def test_constant_series_returns_median(self):
        s = pd.Series([5.0] * 60)
        self.assertEqual(gmm_intersection_threshold(s), 5.0)

    def test_small_sample_returns_median(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(gmm_intersection_threshold(s), float(np.median([1, 2, 3, 4])))

    def test_zero_inflated_guard_positive(self):
        # 大量 0 值 + 少量正值的零膨胀序列：阈值应 > 0（有加购即高摩擦）
        s = pd.Series([0.0] * 100 + [0.693, 1.099, 1.386, 1.609])
        cut = gmm_intersection_threshold(s)
        self.assertGreater(cut, 0.0)
        self.assertLessEqual(cut, min(x for x in s if x > 0))


class TestComputeEngagementMetrics(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'Pages_Viewed': [0, 100, 50],
            'Estimated_Time': [0, 5000, 3000],
            'Session_Count': [1, 5, 3],
            'Purchase_Frequency': [0, 4, 2],
            'Cart_Products': [3, 10, 6],
            'Purchased_Products': [0, 6, 2],
        })

    def test_friction_definition(self):
        out = compute_engagement_metrics(self.df)
        np.testing.assert_allclose(out['Friction'], [3, 4, 4])      # max(加购-购买, 0)
        self.assertTrue(np.allclose(out['Log_Friction'], np.log1p(out['Friction'])))
        self.assertNotIn('Friction_1', out.columns)

    def test_escore_definition(self):
        # E_Score = log1p 后总体标准差(z-score, ddof=0)的等权均值
        out = compute_engagement_metrics(self.df)
        vals = [np.log1p(self.df[c]).to_numpy() for c in
                ['Pages_Viewed', 'Estimated_Time', 'Session_Count']]
        z = [(v - v.mean()) / v.std(ddof=0) for v in vals]
        np.testing.assert_allclose(out['E_Score'], np.mean(z, axis=0), atol=1e-12)


class TestBuildFeatures(unittest.TestCase):
    def test_columns_and_counts(self):
        events = make_events()
        obs_end = events['event_time'].max()
        feats = build_features(events, obs_end)
        for col in ['user_id', 'Purchase_Frequency', 'Total_Spending', 'Pages_Viewed',
                    'Estimated_Time', 'Recency_Days', 'Session_Count',
                    'Cart_Products', 'Purchased_Products', 'E_Score', 'Friction',
                    'Log_Friction']:
            self.assertIn(col, feats.columns)
        self.assertNotIn('Friction_1', feats.columns)
        buyer = feats[feats['user_id'] == 1].iloc[0]
        self.assertEqual(int(buyer['Purchase_Frequency']), 1)
        self.assertEqual(int(buyer['Pages_Viewed']), 2)
        self.assertEqual(int(buyer['Cart_Products']), 1)
        self.assertEqual(int(buyer['Purchased_Products']), 1)
        self.assertEqual(float(buyer['Total_Spending']), 10.0)

    def test_dedupe_exact_duplicates(self):
        events = make_events()
        dup = events[(events['user_id'] == 1) & (events['event_type'] == 'purchase')].iloc[[0]]
        events = pd.concat([events, dup], ignore_index=True)
        feats = build_features(events, events['event_time'].max())
        buyer = feats[feats['user_id'] == 1].iloc[0]
        self.assertEqual(int(buyer['Purchase_Frequency']), 1)   # 重复购买行被去重
        self.assertEqual(int(buyer['Cart_Products']), 1)


class TestSegmentUsers(unittest.TestCase):
    def test_segments_exhaustive(self):
        events = make_events(8)
        feats = build_features(events, events['event_time'].max())
        seg, meta = segment_users(feats)
        self.assertEqual(len(seg), len(feats))
        self.assertTrue(seg['User_Segment'].isin(
            ['高潜力首购用户', '普通浏览用户', '常规已购用户', '高价值直购用户',
             '高价值深度互动用户']).all())
        for key in ['vip_value_index_cutoff', 'nonbuyer_e_score_cutoff',
                    'nonbuyer_log_friction_cutoff', 'vip_e_score_cutoff']:
            self.assertIn(key, meta)
        self.assertNotIn('vip_log_friction1_cutoff', meta)


class TestFlagBuyerSilence(unittest.TestCase):
    def test_silent_vip_flagged(self):
        events = make_events(12)
        feats = build_features(events, events['event_time'].max())
        seg, _ = segment_users(feats)
        # 观察月只包含部分买家（1,2,4,5），其余买家（7,8,10,11）完全沉默
        obs_ids = [1, 2, 4, 5]
        obs = pd.DataFrame({'user_id': obs_ids, 'event_type': ['view'] * len(obs_ids),
                            'event_time': [pd.Timestamp('2019-11-01')] * len(obs_ids)})
        flagged = flag_buyer_silence(seg, obs)
        silent = flagged[flagged['User_Segment'] == '高价值高摩擦用户']
        self.assertGreater(len(silent), 0)
        # 沉默者 = 基期高价值用户且不在观察月
        vip_base = set(seg[seg['User_Segment'].isin(['高价值直购用户', '高价值深度互动用户'])]['user_id'])
        self.assertTrue(set(silent['user_id']) <= (vip_base - set(obs_ids)))
        # 未购人群不受影响
        self.assertEqual(set(seg[seg['Base_Segment'] == '未购用户']['User_Segment']),
                         set(flagged[flagged['Base_Segment'] == '未购用户']['User_Segment']))


class TestRateTest(unittest.TestCase):
    def test_known_contingency(self):
        nov = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'event_type': ['purchase', 'purchase', 'view', 'view', 'view'],
        })
        res = rate_test(nov, {1, 2, 3}, {4, 5}, '测试组：购买率验证')
        self.assertEqual(res['目标人数'], 3)
        self.assertEqual(res['对照人数'], 2)
        self.assertAlmostEqual(res['目标购买率'], 2 / 3)
        self.assertAlmostEqual(res['对照购买率'], 0.0)
        self.assertTrue(res['p值'] >= 0)


class TestScoreNonbuyers(unittest.TestCase):
    def test_columns_and_topk_flag(self):
        events = make_events(12)
        feats = build_features(events, events['event_time'].max())
        seg, _ = segment_users(feats)
        nov = pd.DataFrame({'user_id': [0, 1], 'event_type': ['purchase', 'purchase']})
        out = score_nonbuyers(seg, nov)
        for col in ['First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag']:
            self.assertIn(col, out.columns)
        self.assertEqual(int(out['TopK_Flag'].sum()),
                         int((out['User_Segment'] == '高潜力首购用户').sum()))
        nb = out[out['Purchase_Frequency'].eq(0)]
        ranks = nb['First_Purchase_Rank'].dropna().to_numpy()
        self.assertEqual(ranks.min(), 1)                     # min-rank：并列取同 rank
        self.assertLessEqual(ranks.max(), len(ranks))        # 末尾并列时 max 可 < n
        self.assertTrue(np.allclose(ranks, ranks.astype(int)))


class TestScoreBuyers(unittest.TestCase):
    def test_columns_and_ranks(self):
        rng = np.random.default_rng(7)
        n = 20
        freq = rng.integers(1, 6, n)
        freq[0] = 0
        df = pd.DataFrame({
            'user_id': np.arange(n),
            'Purchase_Frequency': freq,
            'Total_Spending': rng.uniform(50, 5000, n),
            'Recency_Days': rng.integers(1, 31, n),
            'Pages_Viewed': rng.integers(1, 200, n),
            'Estimated_Time': rng.uniform(100, 20000, n),
            'Session_Count': rng.integers(1, 10, n),
            'Cart_Products': rng.integers(0, 8, n),
            'User_Segment': ['高潜力首购用户'] * n,
        })
        nov_ids = df[df['Purchase_Frequency'] > 0]['user_id'].sample(frac=0.5, random_state=1)
        nov = pd.DataFrame({'user_id': nov_ids, 'event_type': ['purchase'] * len(nov_ids)})

        out = score_buyers(df, nov)
        for col in ['Repurchase_Prob', 'Repurchase_Rank']:
            self.assertIn(col, out.columns)
        buyer = out[out['Purchase_Frequency'] > 0]
        ranks = buyer['Repurchase_Rank'].dropna().to_numpy()
        self.assertEqual(ranks.min(), 1)
        self.assertLessEqual(ranks.max(), len(ranks))
        self.assertTrue(out[out['Purchase_Frequency'].eq(0)]['Repurchase_Prob'].isna().all())


class TestExportTracking(unittest.TestCase):
    def test_columns(self):
        events = make_events(10)
        feats = build_features(events, events['event_time'].max())
        seg, _ = segment_users(feats)
        nov = pd.DataFrame({'user_id': [0, 1], 'event_type': ['purchase', 'purchase']})
        seg = flag_buyer_silence(seg, nov)
        seg = score_nonbuyers(seg, nov)
        seg = score_buyers(seg, nov)
        track = export_tracking(seg)
        for col in ['user_id', 'User_Segment', 'E_Score', 'Friction', 'Value_Index',
                    'First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag',
                    'Repurchase_Prob', 'Repurchase_Rank']:
            self.assertIn(col, track.columns)
        self.assertTrue(set(track['User_Segment']) <= {'高潜力首购用户', '高价值高摩擦用户'})


class TestBuyerBaseline(unittest.TestCase):
    def test_metrics_and_preds(self):
        # 手构已购用户表：确保存在高价值高摩擦用户、且验证期复购标签两类齐全
        rng = np.random.default_rng(7)
        n = 20
        freq = rng.integers(1, 6, n)
        freq[0] = 0
        freq[2] = 0
        df = pd.DataFrame({
            'user_id': np.arange(n),
            'Purchase_Frequency': freq,
            'Total_Spending': rng.uniform(50, 5000, n),
            'Recency_Days': rng.integers(1, 31, n),
            'Pages_Viewed': rng.integers(1, 200, n),
            'Estimated_Time': rng.uniform(100, 20000, n),
            'Session_Count': rng.integers(1, 10, n),
            'Cart_Products': rng.integers(0, 8, n),
            'Value_Index': rng.uniform(0, 8, n),
            'User_Segment': ['高价值高摩擦用户' if i in (1, 3) else '常规已购用户' for i in range(n)],
        })
        buyers = df[df['Purchase_Frequency'] > 0]
        nov_ids = buyers['user_id'].sample(frac=0.5, random_state=1)
        nov = pd.DataFrame({'user_id': nov_ids, 'event_type': ['purchase'] * len(nov_ids)})

        res = buyer_baseline(df, nov)
        m = res['metrics']
        self.assertGreater(m['样本(已购用户)'], 0)
        self.assertGreater(m['规则人群规模'], 0)
        for key in ['规则 Top-k 复购率', 'LR Top-k 复购率', 'LR Bottom-k 复购率(风险视角)',
                    '手工价值指数 AUC', 'LR AUC (5折OOF)']:
            self.assertIn(key, m)
        self.assertEqual(len(m['coefficients']), 7)      # RFM + 行为 7 维
        for col in ['user_id', 'y', 'p_lr', 'rule', 'value_index']:
            self.assertIn(col, res['preds'].columns)


class TestRollingValidation(unittest.TestCase):
    def test_rolling_validation_synthetic(self):
        frames = []
        for mi, month in enumerate(['2019-10', '2019-11', '2019-12', '2020-01']):
            ev = make_events(10)
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            frames.append(ev)
        panel = pd.concat(frames, ignore_index=True)
        res = rolling_validation(panel, months=['2019-10', '2019-11', '2019-12', '2020-01'])
        rates = res['验证表']
        # 实验一：3 个月对（2 月窗口）；实验二：2 个月组（基期→沉默→验证，3 月窗口）
        exp1 = rates[rates['实验'] == '高潜力首购']
        exp2 = rates[rates['实验'] == '高价值高摩擦(沉默)']
        self.assertEqual(len(exp1), 3)
        self.assertEqual(len(exp2), 2)
        self.assertEqual(set(exp1['训练月']), {'2019-10', '2019-11', '2019-12'})
        self.assertEqual(set(exp2['沉默月']), {'2019-11', '2019-12'})
        for col in ['训练月', '沉默月', '验证月', '实验', '目标人数', '目标购买率',
                    '对照人数', '对照购买率', '购买率差', 'p值']:
            self.assertIn(col, rates.columns)
        self.assertIsInstance(res['基准表'], pd.DataFrame)

    def test_load_panel(self):
        ev = make_events(6)
        ev['month'] = '2019-10'
        path = os.path.join(tempfile.gettempdir(), 'test_panel.parquet')
        ev.to_parquet(path, index=False)
        try:
            panel = load_panel(path)
            self.assertIn('month', panel.columns)
            self.assertEqual(len(panel), len(ev))
        finally:
            os.remove(path)


class TestCohortMigration(unittest.TestCase):
    """队列迁移分析：固定基期分层 → 逐月追踪同一批用户（合成面板）。"""

    def _panel(self):
        frames = []
        for mi, month in enumerate(['2019-10', '2019-11', '2019-12', '2020-01']):
            ev = make_events(10)
            if mi > 0:                              # 后续月份只保留一半用户 → 另一半沉默
                ev = ev[ev['user_id'] % 2 == 0]
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            frames.append(ev)
        return pd.concat(frames, ignore_index=True)

    def test_cohort_migration_outputs(self):
        # cohort_migration 返回冻结标签占比表（基期标签 × 月份 × 冻结标签）
        panel = self._panel()
        months = ['2019-10', '2019-11', '2019-12', '2020-01']
        frozen = cohort_migration(panel, '2019-10', months=months)
        n_after = 3
        self.assertEqual(len(frozen), frozen['基期标签'].nunique() * n_after)
        self.assertIn('基期标签', frozen.columns)
        self.assertIn('month', frozen.columns)
        # 当月无任何事件的用户应记为"无任何活动"
        self.assertIn('无任何活动', frozen.columns)
        # 每个基期标签 × 月份的行内占比之和应为 1
        rows = frozen.set_index(['基期标签', 'month'])
        self.assertTrue(np.allclose(rows.sum(axis=1).to_numpy(), 1.0, atol=1e-9))

    def test_frozen_thresholds_consistency(self):
        # 冻结阈值 = 用 10 月阈值重算 11 月标签，等价于直接调用 segment_users(thresholds=...)
        panel = self._panel()
        oct_events = panel[panel['month'].eq('2019-10')]
        nov_events = panel[panel['month'].eq('2019-11')]
        seg_oct, thr = segment_users(build_features(oct_events, oct_events['event_time'].max()))
        seg_nov_frozen, _ = segment_users(build_features(nov_events, nov_events['event_time'].max()),
                                          thresholds=thr)
        self.assertNotIn('vip_log_friction1_cutoff', thr)
        self.assertEqual(len(seg_nov_frozen), len(nov_events['user_id'].unique()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
