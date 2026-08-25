"""项目集中配置：路径、月份、随机种子、会话阈值等。

analysis.py / main.ipynb 共用，避免魔法数字散落。
"""
from pathlib import Path

# ── 路径 ──
ROOT = Path(__file__).resolve().parent
OUTPUT_VALIDATION = ROOT / 'validation_results.csv'
OUTPUT_TRACKING = ROOT / 'tracked_users_list_Nov.csv'
PANEL_FILE = ROOT / 'data' / 'panel_7months.parquet'   # 按用户抽样的 7 个月面板
OUTPUT_ROLLING = ROOT / 'rolling_validation_results.csv'
OUTPUT_ROLLING_BASELINE = ROOT / 'rolling_baseline_results.csv'
OUTPUT_COHORT_LABELS = ROOT / 'cohort_frozen_labels.csv'   # 冻结阈值重算的标签占比（标签保持/转化）

# ── 时间窗口 ──
MODEL_MONTH = 10          # 建模月（特征构建 + 阈值拟合）
VALIDATION_MONTH = 11     # 验证月（独立时间外验证）
SESSION_GAP_SECONDS = 1800  # 会话内相邻事件间隔上限（30 分钟）

# ── 多个月用户面板（滚动时间外验证 / 队列迁移分析用）──
MONTHS = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
BASE_MONTH = '2019-10'    # 队列迁移分析的固定基期

# ── 数据质量 ──
DEDUPE_EVENTS = True      # 构建特征前是否去除完全重复事件

# ── 随机性与模型 ──
RANDOM_STATE = 42         # 全局随机种子（GMM / 交叉验证 / bootstrap / LR）
LR_MAX_ITER = 2000
CV_FOLDS = 5              # logistic_baseline 交叉验证折数
BOOTSTRAP_N = 200         # AUC 差异 bootstrap 次数
