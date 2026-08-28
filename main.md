# main.ipynb 逐 cell 详解（复习指南）

> **本文档做什么**：把 `main.ipynb`（35 个 cell）按章节结构逐 cell 复制，并在每个代码 cell 后附**逐行级详细解释**——函数/变量/方法的作用、设计动机、以及"为什么这样做而不是其他方案"。用于快速回忆和复习整个项目。
>
> **配套阅读**：`README.md`（项目总览）、`数据分析报告.md`（结果解读）、`analysis.py`（所有计算逻辑的单一事实来源）。
>
> **代码组织原则**（整个项目最重要的一条设计）：
> **Notebook 只做"加载数据 + 调用 analysis.py 的函数 + 画图 + 展示结果"，所有计算逻辑（特征构建、分层、检验、模型）都在 `analysis.py` 里**。好处：可复用（`run_rolling.py` / `run_cohort.py` / 单元测试共用同一套逻辑）、可测试、Notebook 重启也不会因计算逻辑漂移而结果不一致。

---

## 📑 目录

| 章节（对应 notebook 标题） | Cell | 内容 |
|---|---|---|
| # 用户价值与购买潜力分层 | 00 | 项目简介（markdown） |
| # 环境导入 | 01–02 | 依赖导入、中文字体设置 |
| # 数据加载与预处理 | 03–04 | 读取 7 个月面板（10 月建模 / 11 月验证）、构建用户特征 |
| # EDA 探索性数据分析 | 05–15 | 数据总览、单变量分布可视化、偏度量化与解读、相关分析 |
| ## 多变量相关分析 | 13–15 | 相关热力图 + 解读 |
| # 定义指标 | 16–17 | E_Score / Friction / 价值指数定义 |
| # 用户价值分层建模 | 18–19 | `segment_users` 两阶段分层 + 阈值 |
| # 分层结果概览 | 20–21 | 各人群规模与平均消费图 |
| # 分层结构透视（组内浓度占比） | 22–23 | 100% 堆叠图函数 |
| # 模型验证 | 24–26 | 滚动时间外验证（正式验证） |
| ## 汇总报告：各人群 11 月转化率 | 27–28 | 四组转化率对比（事前基期口径） |
| ## 未购人群基准：逻辑回归 vs 规则 | 29–30 | LR 5 折 OOF 基准 |
| ## 概率分 Top-k 选人（排序层） | 31–32 | 首购分 / 复购分 + 预览 |
| ## 结果持久化 | 33–34 | 导出运营名单 CSV |
| ## 标签迁移分析 | 35–38 | 队列迁移（冻结阈值）+ 解读 + 构成图 |

---

# 用户价值与购买潜力分层

**Cell 00（markdown）** —— 项目一句话简介：

> 基于 7 个月电商行为日志用户面板（按 user_id 哈希抽样 5%，约 78 万用户 / 2060 万条事件）的用户价值与购买潜力分层分析，10 月数据建模、11 月数据验证：
>
> - **未购用户 → 首购潜力识别**：探索度 E_Score（浏览页数 / 时长 / 会话综合的活跃分）+ 加购未买（Friction，加购了却没买）+ 首购概率分（模型预测的下月首次购买概率）；
> - **已购用户 → 价值分层与复购**：价值指数（RFM 连续化，消费 / 频次 / 近度的综合价值分）+ 复购概率分（再次购买概率）；
> - **验证与选人**：滚动时间外验证（正式，用未来月份数据检验标签判别力）、逻辑回归基准（AUC=排序准确度 / 校准误差(Brier) / Top-k=取分数最高前 k 人）、概率分 Top-k 选人、队列迁移分析（追踪同一批用户的标签跨月变化）。

**业务背景**：传统 RFM（最近购买 / 频次 / 金额）对"没买过的人"完全失效（频次、金额都是 0）。本项目把用户先按"买没买"分流，再分别处理：没买的看首购潜力（行为信号），买过的看价值（RFM 连续化）与复购/流失。这是整个方法的骨架，后面所有代码都在实现这张图。

---

# 环境导入

**Cell 01（markdown）**：`# 环境导入`（章节标题，无其他内容）。

---

**Cell 02（代码）** —— 依赖导入与环境设置：

```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from IPython.display import display, HTML

# 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
for font_name in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name]
        break
    except:
        continue
%matplotlib inline
```

**逐行/逐对象解释**：

- `warnings.filterwarnings('ignore')`：全局屏蔽 Python 警告（如 pandas 的 FutureWarning）。**为什么**：数据分析 notebook 追求输出干净可读；代价是隐藏了潜在 API 变更提示，生产代码里不建议这样全局屏蔽，但分析场景可接受。
- `pandas as pd`：表格数据（DataFrame）处理，项目主数据类型。
- `numpy as np`：数值计算（数组、log、percentile 等）。
- `matplotlib.pyplot as plt`：绘图底层 API（`plt.subplots` / `plt.barh` / `plt.show` 等）。
- `matplotlib.font_manager as fm`：字体管理，用来**检测**系统中是否存在某个中文字体。
- `seaborn as sns`：基于 matplotlib 的高层统计绘图库，这里用于热力图（`sns.heatmap`）。
- `from IPython.display import display, HTML`：Notebook 富文本输出。`HTML(...)` 构造 HTML 字符串，`display(...)` 在 cell 下方渲染——本项目用它输出**格式化的 HTML 表格 + 灰色小字"怎么读"说明**。
- `plt.rcParams['axes.unicode_minus'] = False`：让坐标轴负号显示为普通 ASCII 连字符 `-`。**为什么**：matplotlib 默认用 Unicode 减号，在中文字体下常渲染成方块。
- 中文字体循环：matplotlib 默认字体（DejaVu Sans）**不含中文字形**，不设置的话图表里所有中文都是豆腐块。这里依次尝试 `SimHei`（黑体）→ `Microsoft YaHei`（微软雅黑）→ `Arial Unicode MS`（Mac 字体），`fm.findfont(font_name, fallback_to_default=False)` 严格检查字体是否真实存在（`fallback_to_default=False` 表示找不到就抛异常而非静默回退），找到第一个存在的就 `plt.rcParams['font.sans-serif'] = [font_name]` 设为全图默认字体并 `break` 跳出。
- `%matplotlib inline`：Jupyter magic 命令，让 `plt.show()` 的图直接内嵌渲染在 cell 输出里（而不是弹出独立窗口）。

---

# 数据加载与预处理

**Cell 03（markdown）**：

> 数据源为 7 个月用户面板（`data/panel_7months.parquet`，user_id 哈希抽样 5%，约 78 万用户）。取 **10 月事件**用于建模/EDA，**11 月事件**用于验证。

---

**Cell 04（代码）** —— 加载面板并构建用户特征：

```python
# 数据加载：7 个月用户面板（5% 抽样，行过滤只读 10/11 月）→ 10 月建模 / 11 月验证
from config import PANEL_FILE
from analysis import load_panel, build_features

panel = load_panel(PANEL_FILE, months=['2019-10', '2019-11'])  # 只读建模/验证月，避免读全量
oct_events = panel[panel['month'].eq('2019-10')]
df_nov = panel[panel['month'].eq('2019-11')]

print(f"10月事件数: {len(oct_events):,} 条")
print(f"11月事件数: {len(df_nov):,} 条")

# 与 analysis.py 共用同一套特征构建（单一事实来源）
df = build_features(oct_events, oct_events['event_time'].max())

print(f"\n用户级特征表: {df.shape[0]:,} 用户 × {df.shape[1]} 特征")
print(f"10月用于 EDA + 建模, 11月({df_nov['user_id'].nunique():,} 活跃用户)用于验证")
df.head()
```

**逐行/逐对象解释**：

- `from config import PANEL_FILE`：面板文件路径统一放在 `config.py` 集中管理（`ROOT / 'data' / 'panel_7months.parquet'`）。**为什么**：路径不散落在各脚本里，改数据源只动一处。
- `from analysis import load_panel, build_features`：从函数库导入。`load_panel` 负责读 parquet（或 CSV）并保证 `month` 列存在；`build_features` 把"事件日志"聚合为"用户级特征表"。
- `load_panel(PANEL_FILE, months=['2019-10', '2019-11'])`：
  - `months` 参数利用 parquet 的 **predicate pushdown（谓词下推）**——在文件读取层面（pyarrow 引擎）只读取满足 `month IN (...)` 的行，而不是把 2060 万行全读进来再过滤。只读两个月约 550 万行，加载时间从 ~2.5 分钟降到 ~40 秒。
  - `load_panel` 内部还会把 `event_time` 解析为 `datetime` 类型（后续 `diff()` 计算时间差需要）。
- `oct_events = panel[panel['month'].eq('2019-10')]`：布尔索引取出 10 月事件（建模/EDA 用）。`df_nov` 同理取 11 月（验证用）。**为什么必须拆开**：这是"时间外验证（out-of-time validation）"的核心——所有模型参数只用 10 月拟合，11 月完全当"考卷"。如果 11 月参与了建模（比如用"11 月买了"去倒推 10 月该给谁高分），分数就毫无意义了（信息泄露）。
- `df = build_features(oct_events, oct_events['event_time'].max())`：
  - 入参：10 月全部事件 + 观察期结束时间（`event_time` 的最大值，用于计算"距最近购买的天数"）；
  - 出参：**每个用户一行**的特征表（约 15.1 万用户 × 12+ 列），列包括 `user_id, Purchase_Frequency, Total_Spending, Pages_Viewed, Estimated_Time, Recency_Days, Session_Count, Cart_Products, Purchased_Products, E_Score, Friction, Log_Friction`；
- `df.head()`：预览前 5 行（Notebook 会自动渲染成表格）。

---

# EDA探索性数据分析

**Cell 05（markdown）**：`# EDA探索性数据分析`（章节标题）。

## 数据总览

**Cell 06（markdown）**：`## 数据总览`（小节标题）。

---

**Cell 07（代码）** —— 描述性统计：

```python
target_numeric = ['Recency_Days', 'Purchase_Frequency', 'Total_Spending',
                  'Pages_Viewed', 'Estimated_Time', 'Cart_Products']

print(f"数据规模: {df.shape[0]} 行 × {df.shape[1]} 列\n")

miss = df.isnull().sum().sum()
print(f"缺失值: {'✅ 无缺失' if miss == 0 else f'⚠️ {miss} 个'}\n")

summary = df[target_numeric].describe().T.round(2)
summary['skew'] = df[target_numeric].skew().round(2)
display(summary)
```

**逐行/逐对象解释**：

- `target_numeric`：EDA 关注的数值列清单。注意这里没放 `Session_Count`（它也重要，但热力图想看的核心是这几列；后面相关分析会补）。
- `df.shape[0]` / `df.shape[1]`：行数（用户数）/ 列数（特征数）。`shape` 是 `(rows, cols)` 元组。
- `df.isnull().sum().sum()`：**双重 sum**——第一个 `sum()` 按列求和得"每列缺失数"，第二个 `sum()` 再加总得"全表缺失总数"。这里用于检查特征构建是否有缺口（理论上是 0，因为 `build_features` 对没购买的用户回填了 0）。
- `df[target_numeric].describe().T`：`describe()` 输出 count / mean / std / min / 四分位 / max；`.T` 转置让**行=变量、列=统计量**（默认是行=统计量，转置后更易读）。`.round(2)` 保留两位小数。
- `summary['skew'] = df[target_numeric].skew().round(2)`：追加一列"偏度"。**偏度的意义**：`skew > 0` 表示右偏（长尾在右侧，少数极大值拉高均值）。这里会发现 `Total_Spending` 偏度 ≈ 50、`Purchase_Frequency` ≈ 60——极端长尾，直接决定后面必须用 `log1p` 压缩和 GMM 动态阈值。
- `display(summary)`：渲染 HTML 表格。

**EDA 结论 → 方法**（详见 Cell 08 markdown）：未购占 88.4% → 先分流再分层；消费/频次长尾 → log1p + GMM；有加购的未购用户仅 4.4% → log(Friction) 零膨胀、阈值退化为"有加购即高摩擦"。

---

**Cell 08（markdown）**：`### 数据总览解读：三个直接影响后续方法的发现`（解读文字，含"未购 88.4% / 长尾 skew≈50~60 / 加购未买稀疏 4.4%"三点，及其对应的处理方案）。

### 单变量分布可视化：长尾与零膨胀的直观印证

**Cell 09（markdown）** —— 新插入的小节说明：

> Cell 07 的描述统计已给出偏度（skew≈50~60），这里用直方图直接看分布形状：
>
> - **原始值**：几乎所有变量都**堆积在 0 附近**（消费 / 频次 / 加购的 0 值占比 88%~96%），右侧是极长的尾巴——绝大多数细节被挤压成一根竖线；
> - **log1p 变换后**：长尾被压缩，才能看清真实形态——消费/频次是"大量 0 + 平滑正尾"，页数/时长近似单峰，加购是极端零膨胀；
> - **Recency_Days**：未购用户全部落在兜底值（观察期长度），已购用户散布在左侧 → 印证"近度主要区分买没买"。
>
> 结论：这就是为什么后面所有指标用 **log1p 压缩**、阈值用 **GMM 动态拟合**（而不是固定分箱）——固定分箱会被长尾和零膨胀带偏。

---

**Cell 10（代码）** —— 单变量分布图（原始值 vs log1p 变换）：

```python
# ═══════════════════════════════════════════════════
# 单变量分布：原始值 vs log1p 变换（印证长尾 / 零膨胀）
# ═══════════════════════════════════════════════════
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plot_cols = target_numeric + ['Session_Count']   # 与 Cell 07 的数值列一致 + 会话数（相关分析用）
n = len(plot_cols)                               # 7 个变量

# ── 图 1：原始值分布（全范围）──
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
axes = axes.ravel()
for ax, col in zip(axes, plot_cols):
    v = df[col].clip(lower=0)
    zero_pct = (v == 0).mean()
    sns.histplot(v, bins=60, ax=ax, color='#4CB391', kde=True)
    ax.set_title(f'{col}（0值占比 {zero_pct:.0%}）', fontsize=10)
    ax.set_xlabel('')
for ax in axes[n:]:
    ax.axis('off')
plt.suptitle('单变量分布（原始值）：全部严重右偏 / 零膨胀，右侧长尾细节被挤压不可见',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── 图 2：log1p 变换后的分布（长尾压缩后形态可见）──
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
axes = axes.ravel()
for ax, col in zip(axes, plot_cols):
    sns.histplot(np.log1p(df[col].clip(lower=0)), bins=60, ax=ax, color='#66b3ff', kde=True)
    ax.set_title(f'log1p({col})', fontsize=10)
    ax.set_xlabel('')
for ax in axes[n:]:
    ax.axis('off')
plt.suptitle('log1p 变换后：长尾压缩，分布形态（双峰 / 单峰 / 零膨胀）清晰可见',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**逐行/逐对象解释**：

- `plot_cols = target_numeric + ['Session_Count']`：在 Cell 07 的 6 个数值列基础上补上 `Session_Count`（会话数）——相关分析里也要用到它，分布一起看；共 7 个变量。
- `n = len(plot_cols)`：7；2×4 网格有 8 格，循环画完 7 个变量后用 `axes[n:].axis('off')` 把多余的第 8 格关掉（避免出现空白格子干扰视觉）。
- **图 1（原始值，全范围）**：
  - `v = df[col].clip(lower=0)`：防御性把负值截到 0（理论计数非负，防止极端脏数据破坏直方图）；
  - `zero_pct = (v == 0).mean()`：计算"该变量为 0 的用户占比"，写进子图标题——**一眼看出零膨胀程度**（实测：消费 / 频次 / 加购都是 88%~96%，页数 / 会话数为 0%）；`{zero_pct:.0%}` 格式化成百分比；
  - `sns.histplot(v, bins=60, ax=ax, color='#4CB391', kde=True)`：seaborn 直方图 + 核密度曲线（KDE）。`bins=60` 分 60 个桶；KDE 是平滑的概率密度估计，帮助看出峰的位置；
  - 由于最大值极大（消费 84,782、时长 14 万秒），非零部分被挤压到 0 附近的几个桶里、右侧只剩一条细线——**这正是想展示的"长尾"视觉冲击**。
- **图 2（log1p 变换后）**：`np.log1p(df[col].clip(lower=0))` 即 ln(1+x)，把长尾压缩回正常尺度后再画直方图——此时才能看清真实形态：消费 / 频次是"0 值大峰 + 平滑正尾"、页数近似单峰、`Cart_Products` 是极端零膨胀（几乎只有 0 和 1 两个值）、`Recency_Days` 在兜底值处有个尖峰（未购用户）。
- **这两张图对比的意义**：把 Cell 08 解读文字里的"长尾、零膨胀"变成**眼见为实**，并让"为什么后面所有指标用 log1p、阈值用 GMM 动态拟合"有一个直观的分布依据。

---


**Cell 11（代码）** —— 偏度量化对比：原始值 vs log1p 变换后：

```python
# ═══════════════════════════════════════════════════
# 偏度量化对比：原始值 vs log1p 变换后（判断长尾是否还需处理）
# ═══════════════════════════════════════════════════
# plot_cols 沿用上个 cell（7 个数值特征）
skew_table = pd.DataFrame({
    '原始偏度': df[plot_cols].skew().round(2),
    'log1p后偏度': np.log1p(df[plot_cols].clip(lower=0)).skew().round(2),
    '0值占比': (df[plot_cols].clip(lower=0) == 0).mean().round(3),
})
skew_table['偏度等级(log1p后)'] = skew_table['log1p后偏度'].abs().map(
    lambda s: '近似对称(<1)' if s < 1 else ('中等偏(1~2)' if s <= 2 else '严重偏(>2)'))
display(HTML(skew_table.to_html()))

print('解读：')
print('· log1p 对页数/时长/会话非常有效（偏度 12.8/12.7/6.8 → 0.8/-0.4/1.5，接近对称）；')
print('· 仍严重偏(>2)的是零膨胀类（频次/消费/加购，0 值占 88%+）：偏度来自 0 堆积，'
      '任何单调变换都无法消除——这是"大多数人不买"这一信息本身，不是尺度问题；')
print('· Recency_Days 负偏反而加大（-3.2 → -4.7）：log1p 把近度小值压缩得更密、未购兜底值在右端孤立，'
      '属 log1p 对窄值域变量的副作用；但不影响下游——价值指数只用分位数切分，对偏度稳健。')
print('· 结论：本项目下游（GMM 找切分点 / LR 排序 / 分位数）均不假设正态性，'
      'log1p 已足够，无需再做强变换（Box-Cox/Yeo-Johnson/分位数变换会破坏可解释性，'
      '且对 0 堆积类无效）。若未来用对分布敏感的模型（KNN/朴素贝叶斯/直接回归），'
      '再考虑极端值 clip（如 log1p 后 99 分位截断）。')
```

**逐行/逐对象解释**：

- `df[plot_cols].skew()`：pandas 的 `Series.skew()` 计算偏度（Fisher-Pearson 三阶标准化矩）：`>0` 右偏、`<0` 左偏、绝对值越大越偏。
- `np.log1p(df[plot_cols].clip(lower=0)).skew()`：对 log1p 后的分布再算一次偏度，**量化"长尾压缩效果"**——两张分布图是"看"，这里补上"量"。
- `'0值占比': (df[plot_cols].clip(lower=0) == 0).mean()`：0 值比例，必须和偏度**联读**：0 堆积越重（88%+），偏度越难靠单调变换降下来。
- `'偏度等级(log1p后)'` 映射列：按 `|偏度| <1 近似对称 / 1~2 中等偏 / >2 严重偏` 分档，一眼看出哪些变量"仍需关注"。
- **实测结果**（10 月面板）：
  - ✅ log1p 非常有效：页数 12.81→**0.83**、时长 12.71→**-0.35**（几乎对称）、会话 6.84→**1.47**；
  - ⚠️ 仍严重偏（>2）：频次 60.16→3.87、消费 50.47→2.69、加购 7.73→3.32——它们 0 值占比 88%+，偏度来自"**0 堆积**"（大多数用户没买/没加购），这是**信息本身**而非尺度问题，**任何单调变换都无法消除**；
  - ⚠️ Recency_Days 负偏反而加大（-3.21→-4.66）：log1p 把 0~30 天的小值压缩得更密、未购兜底值（31 天）在右端孤立——log1p 对窄值域变量的副作用；但不影响下游（价值指数只用上四分位切分，分位数对偏度稳健）。
- **为什么本项目不需要进一步处理**（代码 print 的结论）：下游环节（GMM 找切分点 / LR 排序 / 分位数）**都不假设正态性**；再做强变换（Box-Cox / Yeo-Johnson / 分位数变换 RankGauss）会破坏 log1p 的可解释性（"加 1 取对数"有直观业务含义，Yeo-Johnson 没有），且对 0 堆积类变量无效。**唯一的例外场景**：若未来改用对分布敏感的模型（KNN / 朴素贝叶斯 / 直接回归预测数值），再考虑极端值 clip（如 log1p 后按 99 分位截断）或分箱化。

---

## 多变量相关分析

**Cell 12（markdown）** —— 偏度详细解读（面向第一次接触的读者，与 Cell 11 的输出表配套）：

> ### 偏度解读：log1p 后偏度仍大，为什么不需要进一步处理？
>
> **先理解"偏度"是什么**：偏度衡量分布**左右不对称**的程度。右偏（正偏）＝大多数数值小、少数数值特别大，长尾拖在右边（本项目的消费 / 频次就是这样）；偏度为 0 表示对称。判断参考：|偏度|<1 近似对称，1~2 中等偏，>2 严重偏。
>
> **log1p 做了什么**：`ln(1+x)` 把"数值的量级"压缩（100→4.6，10000→9.2），能解决"量级型长尾"。但它**改变不了 0 值的位置**——x=0 变换后还是 0。
>
> **三个观察**：
>
> 1. **页数 / 时长 / 会话：log1p 非常有效**（12.8 / 12.7 / 6.8 → 0.8 / −0.35 / 1.5）。这类变量是"量级型长尾"（少数重度用户刷很多页、待很久），log1p 正是为它们准备的；
> 2. **频次 / 消费 / 加购：偏度来自"0 堆积"而非"尺度"**（0 值占 88%+）。绝大多数用户没买 / 没加购，0 永远堆在左端——**任何单调变换都无法消除这一堆 0**。这其实是业务信息本身（"大多数人不买"），不是数据缺陷；
> 3. **Recency_Days：负偏反而加大**（−3.2 → −4.7）。log1p 把 0~30 天的小值压缩得更密，未购用户的兜底值（31 天）在右端显得孤立。这是 log1p 对"窄值域变量"的副作用，但不影响下游——近度只参与价值指数，而价值指数用**上四分位**切分，分位数只看排序、对偏度不敏感。
>
> **核心判断：偏度要不要处理，取决于下游模型是否假设"正态"**。本项目四个下游环节都不假设：
>
> | 下游环节 | 为什么不怕偏度 |
> |---|---|
> | GMM 阈值（未购人群 E_Score / Log_Friction） | 它本来就是"拟合两个高斯分布找切分点"，不要求数据正态；零膨胀时自动退化为"有加购即高摩擦"（业务规则） |
> | 逻辑回归（LR 基准 / 概率分） | 本项目只用它做**排序**（谁更可能买），评估用 AUC（只看相对顺序）。偏度只影响系数大小、不影响排序；且 sklearn 流水线内置 StandardScaler |
> | 价值指数上四分位（VIP 划分） | 分位数只看排序位置，对长尾、偏度、极端值完全稳健 |
> | E_Score（页数/时长/会话的 z 均值） | z-score 只是"相对活跃度"的排序工具，不要求数据正态 |
>
> **为什么不再做强变换**：Box-Cox 要求 x>0（0 值还得加常数，0 堆积依旧）；Yeo-Johnson 虽能处理 0 / 负值，但变换没有业务含义（参数 λ 无法跟运营解释，"加 1 取对数"才讲得清）；分位数变换（RankGauss）能把分布拉成正态，但把 0 值映射到任意位置、数值语义断裂，上线后新用户的分位映射还依赖训练集分布。**一句话：这些变换解决的是"模型需要正态输入"的问题，而本项目没有这个需求——硬做只会损失可解释性、不带来收益。**
>
> **什么时候才需要进一步处理**：未来如果改用对分布敏感的模型——KNN（距离度量受尺度影响）、朴素贝叶斯（高斯假设）、直接回归预测数值（残差正态假设）、或需要解释 LR 系数大小——那时再考虑：极端值 clip（如 log1p 后按 99 分位截断，最保守、保留可解释性）或分箱化。
>
> **最后的检验标准**：偏度大 ≠ 模型差。本项目的时间外验证已经证明这套方法有效（实验一购买率差 +7.88pp、LR AUC 0.67），指标服务的是业务决策，而不是"让数据长得像正态分布"。

**这篇解读的导读**：

- 先给"偏度"一个直觉定义（分布不对称）与判断参考线（<1 / 1~2 / >2），让没接触过统计的读者也能定位；
- 说清 **log1p 的能力边界**：压缩"量级型长尾"有效、对"0 堆积"无效——这是理解"为什么还偏"的关键；
- 把 7 个变量分成三类逐一解释：量级型（页数/时长/会话，log1p 已解决）、零膨胀型（频次/消费/加购，偏度是业务信息本身）、窄值域副作用（Recency，不影响下游）；
- 核心论点用一张表落地：**偏度要不要处理 = 下游是否假设正态**，并列出本项目四个下游环节（GMM / LR 排序 / 分位数 / z-score）为什么都不怕偏度；
- 再对比三种强变换（Box-Cox / Yeo-Johnson / RankGauss）的代价，说明"硬做只损失可解释性、不带来收益"；
- 最后给出**未来什么场景才需要处理**（KNN / 朴素贝叶斯 / 直接回归 / 解释系数）与兜底手段（99 分位 clip），并用"偏度大 ≠ 模型差 + 时间外验证结果"收尾。

---

**Cell 13（markdown）**：`## 多变量相关分析`（小节标题）。

---

**Cell 14（代码）** —— 相关热力图 + 关键相关系数（已补全会话数）：

```python
# 3a. 相关性热力图（7 个数值特征：target_numeric + 会话数，补全浏览三指标两两相关）
corr_cols = target_numeric + ['Session_Count']   # 会话数与页数/时长同为"探索深度"，相关矩阵需包含它
plt.figure(figsize=(9, 8))
corr = df[corr_cols].corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.8, annot_kws={'fontsize': 8})
plt.title('Spearman 相关矩阵（含会话数）', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# ── 关键相关系数（均为 Spearman，与热力图格子数值一致）──
import numpy as np
print(f'浏览三指标相关（Spearman）: 页数↔时长 r={df["Pages_Viewed"].corr(df["Estimated_Time"], method="spearman"):.3f} | '
      f'页数↔会话 r={df["Pages_Viewed"].corr(df["Session_Count"], method="spearman"):.3f} | '
      f'时长↔会话 r={df["Estimated_Time"].corr(df["Session_Count"], method="spearman"):.3f}')
print(f'加购 vs 浏览（Spearman）: r={df["Cart_Products"].corr(df["Pages_Viewed"], method="spearman"):.3f} → Friction 是与浏览独立的意图维度')
print(f'未购用户占比: {(df["Purchase_Frequency"]==0).mean():.1%} | 消费偏度 {df["Total_Spending"].skew():.1f} | '
      f'频次偏度 {df["Purchase_Frequency"].skew():.1f}')
```

**逐行/逐对象解释**：

- `corr_cols = target_numeric + ['Session_Count']`：在 Cell 07 的 6 个数值列上**补上会话数**。**为什么必须补**：浏览三指标（页数 / 时长 / 会话）是相关分析的核心对象（它们高度相关 → 合成 E_Score），原热力图漏了 `Session_Count`，导致"页数↔会话、时长↔会话"两对格子缺失、只能靠 print 文字看到。补上后 7×7 矩阵中三对浏览指标两两相关全部可见（Spearman：页数↔时长 0.925、页数↔会话 0.782、时长↔会话 0.643）。
- `corr(method='spearman')`：计算 Spearman **秩相关**。**为什么不用默认的 Pearson**：Pearson 对线性关系敏感且易被极端值主导（消费偏度 50，几个大额用户就能把相关系数拉向 1）；Spearman 把数值转成排序再算相关，对长尾/单调非线性稳健。这里的长尾数据用 Spearman 更可靠。
- `mask = np.triu(np.ones_like(corr, dtype=bool))`：构造上三角全 True 的掩码矩阵。`sns.heatmap(mask=mask, ...)` 隐藏上三角。**为什么**：相关矩阵是对称的（r(i,j)=r(j,i)），只显示下三角避免信息重复、图更清爽。
- `sns.heatmap(...)` 参数：`annot=True` 在格子里写数值；`fmt='.2f'` 两位小数；`cmap='RdBu_r'` 蓝-红反向色带（蓝=负相关、红=正相关）；`center=0` 以 0 为色带中性点（否则默认色带会被数据范围带偏）；`square=True` 格子正方形；`linewidths=0.8` 格子间距线；`annot_kws={'fontsize': 8}` 数字字号（7×7 格子更密，字号调小）。
- `import numpy as np`：`np.triu` 构造掩码需要它；重复 import 无害（幂等），保证单独运行该 cell 也能工作。
- 打印的相关系数**统一为 Spearman 单一口径**（= 热力图格子里的数值）：
  - 浏览三指标：页数↔时长 **0.925**、页数↔会话 **0.782**、时长↔会话 **0.643**——**高度相关**（度量同一件事"探索深度"）→ 合成一个 E_Score；
  - 加购 vs 浏览 **0.303**——**弱相关** → Friction 是独立的"购买意图受阻"维度；
  - **为什么只用 Spearman 一套就够了**：① Spearman 把数值换成排名，对极端值免疫（消费/频次偏度 50~60）；② 对单调变换（如 log1p）**完全不变**（排名不变）——所以"原始值的 Spearman"天然等于"log1p 后的 Spearman"，不存在"两套数字"，也就不需要 log1p-Pearson 那套了（上一版同时打印两套反而造成困惑）。

---

**Cell 15（markdown）**：`### 相关分析解读：指标构建的依据`（解读 + "EDA → 后续分析的因果逻辑"对照表，把每个 EDA 发现对应到后续处理方法——复习时重点看这张表，它是整个方法论的"为什么"。开头新增 **Spearman 秩相关简介**：把数值换成排名再算相关，对极端值免疫、对单调变换（如 log1p）不变；正文相关系数统一为 Spearman 口径：浏览三指标 0.93/0.78/0.64、加购 vs 浏览 0.30、已购内部消费↔频次 0.53、频次↔近度 −0.22——与热力图格子一致）。

# 定义指标

**Cell 16（markdown）**：

> 1. **探索度 E_Score** — 浏览页数、有效停留时长（会话内相邻事件间隔 <30 分钟累计的活跃秒数）、会话数经 log1p 标准化后的均值（衡量用户探索深度）
> 2. **摩擦 Friction** — 加购但未购买的去重商品数（Cart_Products − Purchased_Products，log1p），度量"加购了却没买"的购买意图受阻（未购人群首购潜力识别用）
> 3. **已购人群"高摩擦"= 跨月完全沉默** — 基期高价值买家在观察月无任何行为（view / cart / purchase 都没有，即完全没来），由 `flag_buyer_silence` 标记，用于流失判定
>
> 由 analysis.compute_engagement_metrics 提供

**要点**：前两个指标在 `analysis.py` 的 `compute_engagement_metrics` 里计算；第三个（沉默）是"跨月行为"判定，不在同一个月内计算，由 `flag_buyer_silence` 结合观察月数据标记。注意"高摩擦"这个词在**未购人群**（加购未买）和**已购人群**（跨月沉默）里含义完全不同。

---

**Cell 17（代码）** —— 验证特征列存在：

```python
# 特征构建与分层指标统一由 analysis.py 提供（单一事实来源）
df[['user_id', 'E_Score', 'Friction']].head()
```

**解释**：只预览 `df` 的三列。前面 Cell 04 的 `build_features` 已经算好了 `E_Score` / `Friction`，这里只是确认列存在、看一眼量级。**为什么专门放一个 cell**：向读者明确"指标定义不在这里、在 analysis.py"——设计上的单一事实来源原则。

---

# 用户价值分层建模

**Cell 18（markdown）**：

> 以 analysis.py 的 `segment_users` 与 `flag_buyer_silence` 为准：
>
> - **高价值**：仅在已购用户内部，按价值指数 `log1p(消费) + log1p(频次) − log1p(近度)`（近度=距最近一次购买的天数，取负号=越久没买价值越低）的上四分位（价值从高到低排序的前 25% 分界）确定
> - **未购用户**：用行为指标（E_Score / Log_Friction）的双成分 GMM 后验概率交点阈值（高斯混合模型把人群分成两类、取两类概率相等的分界点）识别"高潜力首购用户"。Friction = 加购未买，未购用户中 92% 无加购，阈值自然落在"有加购即高摩擦"
> - **高价值用户**：按 E_Score 细分深度互动 / 直购；其中在观察月**完全沉默**者由 `flag_buyer_silence` 标记为"高价值高摩擦用户"（= 跨月沉默 / 流失）
> - 所有阈值只在 **基期月**拟合，验证月只读取基期得到的标签

**核心思想（为什么是两阶段分层）**：85%+ 的用户没买过，他们频次/金额全是 0。若把所有用户直接做 K-Means 聚类，会得到一堆"零购买簇"，标签失去业务含义（这是项目早期踩过的坑，后来砍掉了"8 簇补全到 RFM 八象限"的做法）。改成"**先按是否购买分流、再在各自人群内用可解释指标分层**"。

---

**Cell 19（代码）** —— 执行分层：

```python
from analysis import segment_users, segment_summary

# 分层逻辑（analysis.py）：
# - 高价值：仅在已购用户内部，按价值指数上四分位确定
# - 未购用户：用行为 GMM 阈值识别高潜力首购用户
# - 所有阈值只在 10 月建模期拟合，11 月验证期只读取标签
final_df, thresholds = segment_users(df)

print('建模期拟合阈值:', {k: round(v, 4) for k, v in thresholds.items()})
print()
segment_summary(final_df)
```

**逐行/逐对象解释**：

- `from analysis import segment_users, segment_summary`：导入分层主函数和人群汇总函数。
- `final_df, thresholds = segment_users(df)`：
  - **入参**：10 月用户特征表 `df`；
  - **返回**：① `final_df`——原特征表 + `Base_Segment`（已购/未购）+ `Value_Index`（已购用户的价值指数，未购为 NaN）+ `User_Segment`（最终标签）；② `thresholds`——本次拟合的 4 个阈值字典；
  - **`User_Segment` 取值**（互斥、穷尽）：`普通浏览用户`（未购且未达高潜力）、`高潜力首购用户`（未购 + E_Score 与 Friction 均过阈值）、`常规已购用户`（已购但价值指数未达前 25%）、`高价值直购用户`（已购 + 前 25% 但 E_Score 低）、`高价值深度互动用户`（已购 + 前 25% + E_Score 高）。**"高价值高摩擦"不在这里出现**——那是 `flag_buyer_silence` 结合观察月数据后追加的标记。
  - **内部关键逻辑**（`analysis.py`）：已购人群算 `Value_Index = log1p(消费)+log1p(频次)−log1p(近度)`，`value_cut = value_index.quantile(0.75)` 取上四分位；未购人群对 `E_Score` 和 `Log_Friction` 各拟合**两成分 GMM**（`gmm_intersection_threshold`：拟合 2 个高斯分布，在 2000 个网格点上找两成分后验概率相等处作为阈值，拟合不稳定时退回中位数；零膨胀时自动退化为"最小正值"=有加购即高摩擦）。
- `print('建模期拟合阈值:', ...)`：把 `thresholds` 四舍五入打印。实际值约为：价值指数 5.211、未购 E_Score −0.066、未购 log(Friction) ≈ 0.006、VIP E_Score 1.318。
- `segment_summary(final_df)`：按 `User_Segment` 分组聚合 `用户数 / 平均消费 / 平均购买频次 / 平均探索度 / 平均摩擦力 / 用户占比`，按人数降序。这是简历/汇报可直接引用的概览表。

---

# 分层结果概览

**Cell 20（markdown）**：`# 分层结果概览`（章节标题）。

---

**Cell 21（代码）** —— 人群规模与平均消费图：

```python
# 各人群规模与平均消费概览
seg_counts = final_df['User_Segment'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

axes[0].barh(seg_counts.index[::-1], seg_counts.values[::-1], color='#4CB391')
axes[0].set_title('各人群用户规模', fontsize=14, fontweight='bold')
for i, v in enumerate(seg_counts.values[::-1]):
    axes[0].text(v + 30, i, f'{v:,}', va='center', fontsize=10)

avg_spend = final_df.groupby('User_Segment')['Total_Spending'].mean().sort_values()
axes[1].barh(avg_spend.index, avg_spend.values, color='#66b3ff')
axes[1].set_title('各人群平均消费', fontsize=14, fontweight='bold')
for i, v in enumerate(avg_spend.values):
    axes[1].text(v + 1, i, f'{v:.1f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()
```

**逐行/逐对象解释**：

- `final_df['User_Segment'].value_counts()`：返回 `{标签: 人数}` 的 Series，按人数**降序**（普通浏览 > 常规已购 > 高潜力首购 > 高价值深度互动 > 高价值直购）。
- `plt.subplots(1, 2, figsize=(15, 5.5))`：一行两列子图，总画布 15×5.5 英寸。
- `axes[0].barh(seg_counts.index[::-1], ...)`：**横向**条形图。`[::-1]` 反转顺序——`value_counts` 降序后最大的在最上；但 barh 是从下往上画，反转后**最大的条形出现在最顶部**（视觉习惯：最大值在顶）。颜色 `#4CB391`（绿色系）。
- `axes[0].text(v + 30, i, f'{v:,}', ...)`：在每个条形右端写人数（`{v:,}` 千分位）。`v + 30` 是文字横坐标（条形右端偏右一点避免重叠），`i` 是纵坐标（第几行）。
- `final_df.groupby('User_Segment')['Total_Spending'].mean().sort_values()`：各人群平均消费，**按消费升序排序**（条形图从下到上金额递增，形成阶梯感）。未购人群消费为 0，会排在最下面。
- `axes[1].text(v + 1, i, f'{v:.1f}', ...)`：条形右端写平均消费，一位小数。
- `plt.tight_layout()`：自动调整子图间距防重叠；`plt.show()` 渲染。

**从图里能读到的信息**：普通浏览 12.8 万（占绝对大头）、高潜力首购 5,408 人、高价值（直购 2,363 + 深度互动 2,018）；平均消费上深度互动（2,675）> 高摩擦（1,697）> 直购（1,300）——深度互动是最肥的客群。

---

# 分层结构透视（组内浓度占比）

**Cell 22（markdown）**：`# 分层结构透视（组内浓度占比）`（章节标题）。

---

**Cell 23（代码）** —— 100% 组内归一化堆叠图函数：

```python
def plot_strategic_segments_matrix_100pct(df):
    """
    绘制 100% 组内归一化横向堆叠条形图，透视用户精细化结构。

    横轴统一为"组内浓度占比"，聚焦展示各战略人群在已购/未购两大基础
    分层内部的真实分布占比，消除底层庞大常规用户群绝对数量造成的
    "视觉坍塌"。

    Parameters:
    -----------
    df : pandas.DataFrame
        包含最终分层结果 'Base_Segment' 和 'User_Segment' 的数据框。

    Returns:
    --------
    None
        直接在前端输出图表。
    """

    # 1. 标签归纳映射
    def categorize_for_plot(label):
        if '高摩擦' in label:
            return '高价值高摩擦'
        elif '深度互动' in label:
            return '高价值深度互动'
        elif '直购' in label:
            return '高价值直购'
        elif '常规' in label:
            return '常规已购'
        elif '高潜力' in label:
            return '高潜力首购'
        else:
            return '普通浏览'

    df['Plot_Category'] = df['User_Segment'].apply(categorize_for_plot)

    # 2. 交叉表：基础分层 × 精细化人群
    plot_data = pd.crosstab(df['Base_Segment'], df['Plot_Category'])

    # 预定顺序
    plot_cols = ['普通浏览', '高潜力首购', '常规已购', '高价值直购', '高价值深度互动', '高价值高摩擦']
    for col in plot_cols:
        if col not in plot_data.columns:
            plot_data[col] = 0
    plot_data = plot_data[plot_cols]

    # 基础分层固定顺序
    base_order = ['未购用户', '已购用户']
    plot_data = plot_data.reindex(base_order).fillna(0)
    total_users = len(df)

    # 3. 提前算出每一层（每一行）的总人数
    row_totals = plot_data.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        '普通浏览': '#e0e0e0',
        '高潜力首购': '#66b3ff',
        '常规已购': '#c2e699',
        '高价值直购': '#78c679',
        '高价值深度互动': '#fdae6b',
        '高价值高摩擦': '#ff9999'
    }

    # 左侧起始位置（按百分比累加，从 0.0 开始）
    left_pos = pd.Series([0.0] * len(plot_data), index=plot_data.index)

    for category in plot_cols:
        values = plot_data[category]  # 绝对人数
        pct_values = (values / row_totals) * 100  # 组内百分比 (0-100)

        ax.barh(plot_data.index, pct_values, left=left_pos,
                label=category, color=colors[category], edgecolor='white', height=0.6)

        # 内部文字标签
        if category != '普通浏览':
            for i, val in enumerate(values):
                if val > 0:
                    pct = pct_values.iloc[i]
                    x_pos = left_pos.iloc[i] + pct / 2
                    if pct > 1.5:
                        ax.text(x_pos, i, f'{pct:.1f}%\n({int(val)}人)',
                                ha='center', va='center', color='#333333', fontsize=9, fontweight='bold')

        left_pos += pct_values

    # 4. 最右侧：该层总人数与占大盘比例（固定在 x=102）
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        total_bar = row.sum()
        pct_total = (total_bar / total_users) * 100
        ax.text(102, i, f' {int(total_bar)}人 (占大盘{pct_total:.1f}%)',
                ha='left', va='center', color='#333333', fontsize=11, fontweight='bold')

    # 5. 装饰
    ax.set_xlim(0, 115)
    plt.title('用户精细化结构透视（组内浓度占比）', fontsize=18, pad=25, fontweight='bold')
    plt.xlabel('组内浓度占比 (%)', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.legend(title='用户标签', bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=6, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# 调用 100% 堆叠图函数
plot_strategic_segments_matrix_100pct(final_df)
```

**逐行/逐对象解释**：

- **这个图回答什么问题**：6 个精细化标签在"未购 / 已购"两大基础层内部各占多少比例。**为什么要组内 100% 归一化**：普通浏览 12.8 万人 vs 高价值几百~几千人，若画绝对数量条形图，高价值人群会被压成一条看不见的细线（"视觉坍塌"）；按行归一化成百分比后，未购层 / 已购层各自内部的结构就清楚了。
- `categorize_for_plot(label)`：把 6 个 `User_Segment` 字符串映射为 6 个短类别名（用于图例）。用 `if '高摩擦' in label` 这类**子串匹配**而不是等值匹配——防御未来标签措辞微调（比如加后缀"用户"）。
- `df['Plot_Category'] = df['User_Segment'].apply(categorize_for_plot)`：新增一列。**副作用**：修改了传入的 `final_df`（Notebook 环境无碍，但函数设计上这不是纯函数）。
- `pd.crosstab(df['Base_Segment'], df['Plot_Category'])`：交叉表——行 = `Base_Segment`（未购用户 / 已购用户），列 = 6 个精细化类别，值 = 人数。
- 补齐缺失列：`plot_cols` 是预定的 6 列顺序；若某列不存在（比如当前数据没有"高价值高摩擦"，因为它要 `flag_buyer_silence` 后才出现）就补 0 列，然后 `plot_data = plot_data[plot_cols]` 强制按预定顺序排布。**为什么**：保证颜色/图例顺序在每次运行都一致。
- `plot_data.reindex(base_order).fillna(0)`：行也固定顺序（未购在上、已购在下）。
- `row_totals = plot_data.sum(axis=1)`：每层总人数（未购层 13.4 万、已购层 1.75 万）。
- 堆叠逻辑：`left_pos` 初始为全 0，每画一个类别 `ax.barh(..., left=left_pos)` 后 `left_pos += pct_values`——经典的"手动堆叠"，后一类别从前一类别的右边界开始。
- `pct_values = (values / row_totals) * 100`：**行内**归一化成 0-100。
- 内部文字标签：`if pct > 1.5` 才标文字（太小看不清，避免密密麻麻）；文字位置 `x_pos = left_pos + pct/2`（块中心），内容 `百分比% + (人数)`。
- 最右侧标注：`ax.text(102, i, ...)` 固定在 x=102 处写"该层总人数（占大盘 X%）"，`ax.set_xlim(0, 115)` 留出右侧空间。
- 装饰：`spines['top'/'right'].set_visible(False)` 去掉上/右边框（更现代简洁的图表风格）；图例放图下方居中（`bbox_to_anchor=(0.5, -0.08)`）。
- `plot_strategic_segments_matrix_100pct(final_df)`：用分层结果调用。

---

# 模型验证

**Cell 24（markdown）**：`# 模型验证`（章节标题）。

## 时间外验证（滚动，正式验证）：7 个月面板逐对检验

**Cell 25（markdown）**：

> 在 7 个月面板上逐对运行时间外验证（基期月 t 建模 → 未来月验证），作为**正式验证**——10 月基期约 15 万用户，样本量与跨月重复（6 组实验一 + 5 组实验二）都优于单一月份对：
>
> - **①a 实验一（未购人群）**：基期月 t → 观察月 t+1 购买（2 月对，6 组）；
> - **①b 实验二（已购人群·跨月沉默）**：基期月 t 高价值买家 → 观察月 t+1 完全沉默（= 高价值高摩擦）→ 验证月 t+2 是否购买（3 月组，5 组；用 t+2 做结果，避免"沉默月=结果月"循环）；
> - **②a/②b 两套 LR 基准**（逻辑回归作为对照模型，量化"规则分层 vs 直接建模"的差距）：未购人群预测 t+1 首购；已购人群以"沉默"为规则标记预测 t+2 复购。
>
> 解读要点：实验一购买率差（目标组购买率 − 对照组购买率）应跨月稳定为正（标签稳健性）；实验二（沉默）若显著为负且方向一致，说明"跨月完全沉默"是可靠的流失信号；LR AUC 跨月稳定优于规则 → 概率分选人优先。

**为什么实验二必须用 3 月组**：沉默的定义是"观察月 t+1 无任何行为"。若用 t+1 的购买作为结果，那"沉默组"在定义上就注定买不了（无行为），结论是循环论证（"沉默月=结果月"）。所以把结果推到 t+2——先看 t+1 是否沉默，再看 t+2 是否购买。

---

**Cell 26（代码）** —— 滚动验证结果加载 / 重算 + 展示：

```python
# ═══════════════════════════════════════════════════
# 滚动时间外验证（结果默认读取 CSV；如需重算约 15-25 分钟）
# ═══════════════════════════════════════════════════
import pandas as pd
import numpy as np
from pathlib import Path
from config import OUTPUT_ROLLING, OUTPUT_ROLLING_BASELINE, PANEL_FILE

# 分析只读这 7 列（category_code / brand 等字符串列不读，省 IO 与内存）
PANEL_COLUMNS = ['event_time', 'event_type', 'price', 'product_id', 'user_id', 'user_session', 'month']

RECOMPUTE = False      # 改为 True 则重新计算（读 7 个月面板并逐对验证，耗时约 15-25 分钟）

if RECOMPUTE:
    from analysis import load_panel, rolling_validation
    panel = load_panel(PANEL_FILE, columns=PANEL_COLUMNS)
    res = rolling_validation(panel)
    rates, aucs = res['验证表'], res['基准表']
    rates.to_csv(OUTPUT_ROLLING, index=False, encoding='utf-8-sig')
    aucs.to_csv(OUTPUT_ROLLING_BASELINE, index=False, encoding='utf-8-sig')
else:
    if not (Path(OUTPUT_ROLLING).exists() and Path(OUTPUT_ROLLING_BASELINE).exists()):
        raise SystemExit(
            '未找到滚动验证结果 CSV（rolling_validation_results.csv / rolling_baseline_results.csv）。\n'
            '请先运行 `python run_rolling.py`（需 data/panel_7months.parquet，约 15-25 分钟），'
            '或将上方 RECOMPUTE 改为 True 在本 Notebook 内重算。')
    rates = pd.read_csv(OUTPUT_ROLLING)
    aucs = pd.read_csv(OUTPUT_ROLLING_BASELINE)

# ── 表格格式化：百分比 / 小数点，便于直接阅读 ──
def fmt_pct(x):
    return '—' if pd.isna(x) else f'{x:.1%}'

def fmt_p(x):
    return '—' if pd.isna(x) else f'{x:.1e}'

def fmt_rates(df, keep_cols):
    out = df.copy()
    out['目标购买率'] = out['目标购买率'].map(fmt_pct)
    out['对照购买率'] = out['对照购买率'].map(fmt_pct)
    out['p值'] = out['p值'].map(fmt_p)
    return out[keep_cols]

rates1 = rates[rates['实验'].str.contains('高潜力首购')].copy()
rates2 = rates[rates['实验'].str.contains('沉默')].copy()

display(HTML('<h3>①a 实验一（未购人群）：高潜力首购 vs 普通浏览 → 次月购买率</h3>'))
display(HTML('<p style="color:#555">怎么读：每一行是一次"用基期月标签预测下月首购"的时间外验证。比较<b>目标组 vs 对照组购买率</b>：'
             '目标组明显更高（且 p&lt;0.05）⇒ "高潜力首购"标签能预测首购（本表 6 行全部显著）。</p>'))
display(HTML(fmt_rates(rates1, ['训练月', '验证月', '目标人数', '目标购买率',
                                '对照人数', '对照购买率', 'p值']).to_html(index=False)))

display(HTML('<h3>①b 实验二（已购人群·沉默）：沉默高价值 vs 活跃高价值 → 验证月购买率</h3>'))
display(HTML('<p style="color:#555">怎么读：每一行是一次 3 月组验证（基期 t 高价值买家 → 观察月 t+1 完全沉默 → 验证月 t+2 是否购买，'
             '避免"沉默月=结果月"循环）。比较<b>沉默组 vs 对照组购买率</b>：沉默组明显更低（且 p&lt;0.05）'
             '⇒ "跨月完全沉默"预示流失（本表 5 行全部显著）。</p>'))
display(HTML(fmt_rates(rates2, ['训练月', '沉默月', '验证月', '目标人数', '目标购买率',
                                '对照人数', '对照购买率', 'p值']).to_html(index=False)))

# ── 滚动基准表（两套 LR）格式化 + 按臂拆分 ──
def fmt_aucs(df, keep_cols):
    out = df.copy()
    out['结果率'] = out['结果率'].map(fmt_pct)
    out['规则 AUC'] = out['规则 AUC'].map(lambda x: f'{x:.3f}')
    out['LR AUC (OOF)'] = out['LR AUC (OOF)'].map(lambda x: f'{x:.3f}')
    out['LR Top-k 率'] = out['LR Top-k 率'].map(fmt_pct)
    return out[keep_cols]

aucs1 = aucs[aucs['人群'].str.contains('未购人群')].copy()
aucs2 = aucs[aucs['人群'].str.contains('已购人群')].copy()

display(HTML('<h3>②a 基准（未购人群·首购）：规则分层 vs 逻辑回归</h3>'))
display(HTML('<p style="color:#555">怎么读：AUC 衡量"按分数排序选人"的准确度（0.5=随机，越接近 1 越好）。'
             '以 10 月未购用户为样本预测次月首购，<b>LR AUC (OOF) 稳定高于规则 AUC</b> ⇒ '
             '概率分比规则标签更擅长按分数排序选人（规则标签留作解释层）。</p>'))
display(HTML(fmt_aucs(aucs1, ['训练月', '验证月', '样本', '结果率', '规则人群规模',
                              '规则 AUC', 'LR AUC (OOF)', 'LR Top-k 率']).to_html(index=False)))

display(HTML('<h3>②b 基准（已购人群·复购）：规则分层 vs 逻辑回归</h3>'))
display(HTML('<p style="color:#555">怎么读：以基期已购用户为样本预测验证月复购（规则标记 = 跨月沉默）。'
             '<b>LR AUC (OOF) 稳定高于规则 AUC</b> ⇒ 复购概率分比沉默规则更擅长按分数排序选人。'
             '注意：沉默规则是"流失识别器"，其复购 AUC 低于 0.5 属正常（负向指标）。</p>'))
display(HTML(fmt_aucs(aucs2, ['训练月', '沉默月', '验证月', '样本', '结果率', '规则人群规模',
                              '规则 AUC', 'LR AUC (OOF)', 'LR Top-k 率']).to_html(index=False)))
# 拆分展示：实验一（未购人群）与实验二（已购人群·沉默）分开看，避免两类实验混排

# ── 可视化一：购买率对比（实验一 / 实验二 分面柱状图）──
exp1 = rates[rates['实验'].str.contains('高潜力首购')]
exp2 = rates[rates['实验'].str.contains('沉默')]
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, g, arm, c_a, c_b in [
    (axes[0], exp1, '实验一：高潜力首购 vs 普通浏览（未购人群）', '#4CB391', '#9e9e9e'),
    (axes[1], exp2, '实验二：沉默 vs 活跃高价值（已购人群）', '#ff9999', '#9e9e9e')]:
    x = np.arange(len(g)); w = 0.36
    ax.bar(x - w/2, g['目标购买率'] * 100, w, label='目标组', color=c_a)
    ax.bar(x + w/2, g['对照购买率'] * 100, w, label='对照组', color=c_b)
    for xi, (t, c) in enumerate(zip(g['目标购买率'] * 100, g['对照购买率'] * 100)):
        ax.text(xi - w/2, t + 0.4, f'{t:.1f}', ha='center', fontsize=8)
        ax.text(xi + w/2, c + 0.4, f'{c:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(g['验证月'], rotation=45)
    ax.set_ylabel('购买率 (%)')
    ax.set_title(arm, fontsize=12, fontweight='bold')
    ax.legend()
plt.suptitle('滚动验证：目标组与对照组购买率逐月对比（柱高差越大，标签区分力越强）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── 可视化二：购买率差跨月趋势（稳定性）──
piv = rates.pivot(index='训练月', columns='实验', values='购买率差').sort_index()
ax = piv.plot(marker='o', figsize=(10, 4.5))
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_ylabel('购买率差 (pp)')
ax.set_title('目标组−对照组购买率之差跨月趋势：实验一为正（标签有效）；实验二为负（沉默=流失）', fontsize=13)
plt.tight_layout()
plt.show()

# ── 可视化三：LR 基准 AUC 对比 ──
x = np.arange(len(aucs)); w = 0.36
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x - w/2, aucs['规则 AUC'], w, label='规则分层', color='#c2c2c2')
ax.bar(x + w/2, aucs['LR AUC (OOF)'], w, label='逻辑回归 (OOF)', color='#66b3ff')
ax.axhline(0.5, color='gray', ls='--', lw=1, label='随机水平 = 0.5')
labels = [f"{r['训练月']}→{r['验证月']}" + (f"\n(沉默月 {r['沉默月']})" if pd.notna(r['沉默月']) else '')
          for _, r in aucs.iterrows()]
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('AUC')
ax.set_title('LR 基准：逻辑回归 AUC 稳定高于规则分层（均优于随机 0.5）', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()
```

**逐行/逐对象解释**：

- **`RECOMPUTE = False` 开关设计**：滚动验证要读全量 7 个月面板（2060 万行）并逐对跑 11 次特征构建+分层+LR，耗时 15-25 分钟。因此结果**默认读取已入库的 CSV**（`rolling_validation_results.csv` / `rolling_baseline_results.csv`，已随仓库提交）；需要重算才把开关改 True。**为什么 CSV 入库**：保证 clone 仓库后 Notebook 不用等 20 分钟就能出结果；CSV 与代码口径一致（由 `run_rolling.py` 生成）。
- `PANEL_COLUMNS`：只读 7 个分析必需列。原始 parquet 还有 `category_id / category_code / brand` 等字符串列，本项目分析用不到——**列裁剪**把 IO 和内存占用降约 40%。
- `if RECOMPUTE:` 分支：`rolling_validation(panel)` 返回 `{'验证表': ..., '基准表': ...}`——`验证表` 是实验一/二逐月统计（目标/对照人数、购买率、购买率差、p 值），`基准表` 是两套 LR 指标（样本、结果率、规则人群规模、规则 AUC、LR AUC、LR Top-k 率）。写回 CSV 用 `encoding='utf-8-sig'`（带 BOM，Excel 打开中文不乱码）。
- `else:` 分支的文件存在性检查：**这是后加的兜底**——若 CSV 缺失（如旧 clone），直接 `raise SystemExit` 并给出明确指引，而不是 `pd.read_csv` 抛一个让人摸不着头脑的 FileNotFoundError。
- `fmt_pct(x)`：`—` 表示缺失（NaN），否则 `{:.1%}` 百分比一位小数。`fmt_p(x)`：p 值用科学计数法 `{:.1e}`（p 都是 1e-68 ~ 1e-262 量级，小数显示不下）。
- `fmt_rates(df, keep_cols)`：把"购买率/对照购买率"转百分比、"p 值"转科学计数，然后只保留 `keep_cols` 列（去掉 `实验`/`目标组` 等冗余列，表格更紧凑）。
- `rates1 = rates[rates['实验'].str.contains('高潜力首购')]`：按实验名拆出实验一（6 行）和实验二（5 行）。`.str.contains('沉默')` 匹配"高价值高摩擦(沉默)"。
- `display(HTML('<h3>...'))`：输出 HTML 小标题；`<p style="color:#555">` 灰色小字解释"怎么读"（每张表配一句方法论说明，这是面向评审/面试的贴心设计）。注意 HTML 里 `<` 要写成 `&lt;` 转义（`p&lt;0.05`）。
- `fmt_aucs`：AUC 列格式化 3 位小数，结果率/LR Top-k 率百分比。
- `aucs1 / aucs2`：按 `人群` 列拆分未购人群（6 行）和已购人群（5 行）。
- **可视化一（分面双柱图）**：`for ax, g, arm, c_a, c_b in [(axes[0], exp1, ...), (axes[1], exp2, ...)]` 循环画两个子图。`x = np.arange(len(g))` 是每月位置，`w = 0.36` 柱宽；目标组柱在 `x - w/2`、对照组柱在 `x + w/2`（并排双柱）。`ax.bar(x - w/2, g['目标购买率'] * 100, ...)`：购买率是小数（0.133），乘 100 变百分比。柱顶文字标数值。`ax.set_xticklabels(g['验证月'], rotation=45)` 横轴是验证月。**这张图回答"标签有没有区分力"**：柱高差越大越好（实验一目标高、实验二沉默组低）。
- **可视化二（购买率差趋势）**：`rates.pivot(index='训练月', columns='实验', values='购买率差')` 透视成"行=训练月、列=实验、值=购买率差"，`axhline(0)` 画 0 参考线。**回答"结论跨月稳不稳"**：实验一折线恒在 0 上方（+4.5~+9.3pp）、实验二恒在下方（−19.9~−31.8pp）→ 标签稳健。
- **可视化三（LR vs 规则 AUC）**：x 轴是每个滚动组合（标签含"训练月→验证月"，已购人群还标"沉默月"）；`axhline(0.5)` 随机水平线。**回答"直接建模 vs 手工规则谁排序更强"**：LR（蓝）稳定高于规则（灰）。
- **注意（后加说明）**：已购人群"规则 AUC"≈0.47 低于 0.5 不是 bug——规则是**沉默标记**（预测"不买"），对"复购=1"的目标天然是负向指标；评估流失识别力时应对 `1−y` 看（AUC ≈ 0.53）。Notebook 已加灰色注释说明，避免被误读为"规则比随机差"。

---

## 汇总报告：各人群 11 月转化率对比（面板口径）

**Cell 27（markdown）**：`## 汇总报告：各人群 11 月转化率对比（面板口径）`（小节标题）。

---

**Cell 28（代码）** —— 四组 11 月转化率对比（含存活偏差修复）：

```python
# ═══════════════════════════════════════════════════
# 汇总报告：各人群 11 月转化率对比
# ═══════════════════════════════════════════════════
from analysis import flag_buyer_silence

# 已购人群"高摩擦" = 跨月完全沉默：10月高价值买家在 11 月无任何行为
# ⚠️ 口径说明：先保存 10 月基期标签（Base_User_Segment），再做沉默分流。
# 四组转化率一律按【事前基期口径】计算（分母 = 10 月全量该标签用户）。
# 若改用 flag 后的 User_Segment 选 VIP，只会剩"11 月仍活跃"的用户，
# 转化率变成已知未来的条件概率（存活偏差/未来信息泄漏）。
final_df = final_df.copy()
final_df['Base_User_Segment'] = final_df['User_Segment']
final_df = flag_buyer_silence(final_df, df_nov)

target_potential_ids = final_df[final_df['Base_User_Segment'] == '高潜力首购用户']['user_id'].to_numpy()
control_low_value_ids = final_df[final_df['Base_User_Segment'] == '普通浏览用户']['user_id'].to_numpy()
target_immersive_vip_ids = final_df[final_df['Base_User_Segment'] == '高价值深度互动用户']['user_id'].to_numpy()
target_efficient_vip_ids = final_df[final_df['Base_User_Segment'] == '高价值直购用户']['user_id'].to_numpy()

def evaluate_nov_performance(user_list, group_name):
    if len(user_list) == 0:
        return None
    group_nov = df_nov[df_nov['user_id'].isin(user_list)]
    nov_purchases = group_nov[group_nov['event_type'] == 'purchase']
    buyers = nov_purchases['user_id'].nunique()
    cvr = buyers / len(user_list)
    rev = nov_purchases['price'].sum()
    arppu = rev / buyers if buyers > 0 else 0
    return {
        '人群分组': group_name,
        '人数': len(user_list),
        '11月下单': buyers,
        '转化率': f"{cvr:.2%}",
        '11月营收': round(rev, 2),
        '人均消费': round(arppu, 2)
    }

groups = [
    (target_potential_ids, "实验组-高潜力首购（10月基期口径）"),
    (control_low_value_ids, "对照组-普通浏览（10月基期口径）"),
    (target_immersive_vip_ids, "高价值深度互动（10月基期全量口径）"),
    (target_efficient_vip_ids, "高价值直购（10月基期全量口径）"),
]

report = pd.DataFrame([g for g in [evaluate_nov_performance(*g) for g in groups] if g is not None])
display(HTML(report.to_html(index=False)))
```

**逐行/逐对象解释**：

- `flag_buyer_silence(final_df, df_nov)`：把"10 月被分为高价值（直购/深度互动）、但 11 月完全无任何事件（view/cart/purchase 都没有）"的用户改标为 `高价值高摩擦用户`。**语义**：跨月完全沉默 = 流失高危判定。
- **`Base_User_Segment` 口径修复（重点，后加）**：
  - 原始写法是直接 `final_df = flag_buyer_silence(...)`，然后用 `User_Segment == '高价值深度互动用户'` 选人——但此时该标签里**只剩 11 月仍活跃的 VIP**（沉默的被移走了）；
  - 这导致 VIP 组的"11 月转化率"变成了**已知 11 月活跃的前提下的条件概率**——分组信息本身用了 11 月数据（未来信息泄漏 / 存活偏差）。实测：深度互动旧口径 60.4%（1,534 人）vs 事前全量口径 45.9%（2,018 人），差 14.5pp；
  - 修复：`final_df['Base_User_Segment'] = final_df['User_Segment']` 先把 10 月基期标签存档，`flag_buyer_silence` 只改 `User_Segment`，选人一律用 `Base_User_Segment`。这样四组（高潜力/普通浏览/两个 VIP）**都是"10 月定义、与 11 月行为无关"的事前分组**，转化率口径一致、可比。
  - `final_df = final_df.copy()`：避免在后续被链式赋值改到共享对象（防御 SettingWithCopyWarning）。
- `target_potential_ids / control_low_value_ids`：未购人群的实验组（高潜力首购）和对照组（普通浏览），用 `.to_numpy()` 转成 ndarray（后续 `isin` 更快）。
- `evaluate_nov_performance(user_list, group_name)`：给定人群（10 月定义的 user_id 集合），统计 11 月表现：
  - `df_nov[df_nov['user_id'].isin(user_list)]`：过滤出该人群的 11 月事件；
  - `nov_purchases = group_nov[group_nov['event_type'] == 'purchase']`：只留购买事件；
  - `buyers = nov_purchases['user_id'].nunique()`：**按用户去重**的购买人数（一人买多次只算 1 次转化）——转化率的正确口径；
  - `cvr = buyers / len(user_list)`：转化率（分母 = 事前基期人数）；
  - `rev = nov_purchases['price'].sum()`：11 月该人群贡献的营收（购买事件价格之和）；
  - `arppu = rev / buyers if buyers > 0 else 0`：**ARPPU**（每付费用户平均收入）；若无人购买则 0（避免除零）。
- `groups`：四组的 (user_id 数组, 显示名) 列表。组名后缀"（10月基期口径）"明示口径。
- `report = pd.DataFrame([...])`：对每组调用 `evaluate_nov_performance(*g)`（`*g` 解包成两个参数），过滤掉空组（`if g is not None`），拼成 DataFrame；`display(HTML(report.to_html(index=False)))` 渲染成表格。

**结果**（面板口径）：高潜力首购 13.30% vs 普通浏览 5.41%（+7.88pp）；深度互动 45.9%、直购 36.6%（事前全量口径）。

---

## 未购人群基准：逻辑回归 vs 规则分层

**Cell 29（markdown）**：

> 用逻辑回归（5 折 OOF：5 折交叉验证中，每折用"没训练过该折数据"的模型做预测，避免高估）检验规则分层（手工阈值打标签）作为个体排序器的判别力，输出 AUC / 校准误差(Brier) / Top-k 购买率等关键指标。

**OOF 为什么关键**：如果模型在全体数据上训练再评估自己，预测是"见过的样本"，指标会乐观偏置。5 折 OOF 把数据切 5 份，每折用其余 4 折训练、对本折预测——每个样本的预测都来自"没见过它"的模型，评估才可信。

---

**Cell 30（代码）** —— 未购人群 LR 基准：

```python
# ═══════════════════════════════════════════════════
# 未购用户逻辑回归基准（首购增量检验）
# 以 10 月未购用户为样本，用逻辑回归预测 11 月是否购买，
# 与规则分层对比 AUC / 校准误差(Brier) / Top-k 购买率（5 折 OOF）。
# ═══════════════════════════════════════════════════
from analysis import nonbuyer_baseline

bl = nonbuyer_baseline(final_df, df_nov)
m = bl['metrics']
print(pd.Series({k: v for k, v in m.items() if k != 'coefficients'}).to_string())
print('逻辑回归系数(全量拟合):')
print(pd.Series(m['coefficients']).round(4).to_string())
```

**逐行/逐对象解释**：

- `nonbuyer_baseline(final_df, df_nov)`（实现在 analysis.py）：
  - 样本 = 10 月未购用户（`Purchase_Frequency == 0`，约 13.4 万）；
  - 标签 y = 11 月是否发生购买（1/0）；
  - 特征 = `log1p(页数/时长/会话数/加购数)` 4 维（`_nonbuyer_lr_features`）；
  - 5 折 StratifiedKFold（分层抽样保证每折正负样本比例一致）+ `cross_val_predict(..., method='predict_proba')` 得 OOF 概率；
  - 规则基准 = `User_Segment == '高潜力首购用户'` 二值标签；
  - 返回 `{'metrics': {...}, 'preds': DataFrame(user_id, y, p_lr, rule)}`。
- `bl['metrics']` 里的关键指标：
  - `样本(未购用户)` / `11月购买率`（y 均值，约 5.7%）；
  - `规则人群规模`（k=5408）/ `规则 Top-k 购买率`（13.30%）/ `LR Top-k 购买率`（16.31%）——**同预算对比**：都取前 5408 人，LR 圈出的首购率比规则高 3pp；
  - `规则 AUC`（0.528）/ `LR AUC (5折OOF)`（0.673）——排序准确度，0.5=随机；
  - `LR AUC 仅浏览特征`（x_vol，只用前 3 维浏览特征）/ `LR AUC +规则标记`（把规则二值并入特征再训）；
  - `LR Brier (OOF)` / `规则 Brier`——**校准误差**（均方误差式的概率校准指标，越低越好）；
  - `LR−规则 AUC 差 (bootstrap 95% CI)`——200 次有放回重采样算 AUC 差的置信区间（判差异是否显著）。
- `print(pd.Series({k: v for k, v in m.items() if k != 'coefficients'}).to_string())`：把 metrics 打印成纵向文本（排除系数 dict）；`pd.Series(...)` 转换后 `.to_string()` 对齐显示。
- `m['coefficients']`：全量数据重新拟合 LR 后的**系数**（`pipe.fit(x_full, y)` 后的 `coef_[0]`），说明各特征的方向与相对重要性（如加购商品数系数最大——首购意图最强信号）。注意：系数是全量拟合（解释用），评估指标一律用 OOF（评估用）——**两套口径分开**，避免用"见过样本"的系数自我表扬。

---

## 概率分 Top-k 选人（排序层）

**Cell 31（markdown）**：

> 未购人群与已购人群的基准均显示：规则分层作为个体排序器弱于逻辑回归。因此：
>
> - **排序层（选谁触达 / 谁要预警）**：
>   - 未购用户 → 首购概率分 `First_Purchase_Prob`（Top-k 触达）；
>   - 已购用户 → 复购概率分 `Repurchase_Prob`（Top-k 复购运营，Bottom-k〔分数最低的后 k 人，流失风险最高〕流失预警）；
> - **解释层（怎么触达）**：保留 E_Score / Friction 分群与策略话术（购物车挽回 / 首购券 / 体验排障）；
> - 名单同时输出概率分与分群标签；`TopK_Flag` 标记"同预算（= 规则人群规模）下未购用户应触达的前 k 人"。

**两层架构的动机**：规则标签解释性强（能讲人话：加购未买 + 高探索 → 购物车挽回话术），但个体排序弱（AUC 0.53）；LR 概率分排序强（AUC 0.67）但不好解释。所以**选人用概率分（排序层），话术用标签（解释层）**——这是本项目"规则 + 模型"双层的核心设计。

---

**Cell 32（代码）** —— 打概率分 + 预览：

```python
# 概率分 Top-k 选人（排序层）：未购 → 首购分，已购 → 复购分
from analysis import score_buyers, score_nonbuyers

final_df = score_nonbuyers(final_df, df_nov)
final_df = score_buyers(final_df, df_nov)
k_rule = int((final_df['User_Segment'] == '高潜力首购用户').sum())
m = bl['metrics']
print(f"同预算 k={k_rule}：规则 Top-k 购买率 {m['规则 Top-k 购买率']:.2%} → LR Top-k {m['LR Top-k 购买率']:.2%}（OOF 评估）")
print('名单新增列：未购用户 First_Purchase_Prob / First_Purchase_Rank / TopK_Flag；已购用户 Repurchase_Prob / Repurchase_Rank\n')

# 未购人群按首购分排序的前 10 名预览
top_preview = (final_df.loc[final_df['User_Segment'] == '高潜力首购用户',
                            ['user_id', 'E_Score', 'Friction', 'First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag']]
               .sort_values('First_Purchase_Rank').head(10))
display(HTML(top_preview.to_html(index=False)))

# 已购（高价值高摩擦 = 11月完全沉默）人群按复购分排序的前 10 名预览
vip_preview = (final_df.loc[final_df['User_Segment'] == '高价值高摩擦用户',
                            ['user_id', 'Value_Index', 'Repurchase_Prob', 'Repurchase_Rank']]
               .sort_values('Repurchase_Rank').head(10))
display(HTML(vip_preview.to_html(index=False)))

# Top-k 名单规模核对
print('TopK_Flag=1 的未购用户数:', int(final_df['TopK_Flag'].sum()))
```

**逐行/逐对象解释**：

- `score_nonbuyers(final_df, df_nov)`（analysis.py）：对未购用户全量拟合 LR（特征同上）→ 加 3 列：`First_Purchase_Prob`（首购概率）、`First_Purchase_Rank`（概率降序排名，`rank(method='min')` 并列同 rank）、`TopK_Flag`（rank ≤ k 置 1，k = 规则人群规模 5408）。已购用户这三列保持 NaN。
- `score_buyers(final_df, df_nov)`：对已购用户（`Purchase_Frequency > 0`）拟合复购 LR（7 维 RFM+行为特征）→ 加 `Repurchase_Prob` / `Repurchase_Rank`。未购用户保持 NaN。
- `k_rule = int((final_df['User_Segment'] == '高潜力首购用户').sum())`：规则人群规模（5408）作为"同预算"基准——**触达预算按规则圈出的人数算，再让 LR 用同样的钱选人**，比较才公平。
- `m = bl['metrics']`：复用 Cell 30 的 OOF 指标。打印"规则 Top-k 13.30% → LR Top-k 16.31%"——**这就是"概率分比规则选人更强"的一行证据**。
- `top_preview`：高潜力首购人群内按首购分排名取前 10（列：user_id / E_Score / Friction / 概率分 / 排名 / TopK_Flag）。注意**此时 `User_Segment` 仍是 flag 后的标签**（高潜力首购不受 flag 影响）。
- `vip_preview`：高价值高摩擦（= 11 月完全沉默的 VIP）人群按**复购分**排序取前 10。**为什么看复购分**：复购概率越低流失风险越高，`Repurchase_Rank` 最小的其实是复购分最高的人——这里取 `sort_values('Repurchase_Rank').head(10)` 展示的是"复购分最高"的沉默 VIP（最值得优先召回挽回的）。运营上也可取 rank 最大的（Bottom-k，流失最严重）。
- `int(final_df['TopK_Flag'].sum())`：核对 TopK_Flag=1 的人数应等于 5408。

---

## 结果持久化

**Cell 33（markdown）**：`## 结果持久化`（小节标题）。

---

**Cell 34（代码）** —— 导出运营名单：

```python
from analysis import export_tracking

tracking = export_tracking(final_df, 'tracked_users_list_Nov.csv')
print(f"已导出 {len(tracking):,} 名候选用户（高潜力首购 + 高价值高摩擦）至 tracked_users_list_Nov.csv")
```

**逐行/逐对象解释**：

- `export_tracking(final_df, 'tracked_users_list_Nov.csv')`（analysis.py）：
  - 筛选 `User_Segment ∈ {高潜力首购用户, 高价值高摩擦用户}`（= 未购人群要触达 + 已购人群要召回的两类人）；
  - 保留 10 列：`user_id, User_Segment, E_Score, Friction, Value_Index, First_Purchase_Prob, First_Purchase_Rank, TopK_Flag, Repurchase_Prob, Repurchase_Rank`——**排序层（概率分/排名） + 解释层（标签/指标）双齐全**；
  - `to_csv(..., encoding='utf-8-sig')` 带 BOM，Excel 直接打开不乱码。
- 名单规模 6,792 = 5,408（高潜力首购）+ 1,384（沉默高价值）。
- **用途**：这份名单是后续随机 A/B 触达实验的**抽样框**——运营按预算取 `rank ≤ 预算` 即可选人。⚠️ 注意名单的 `First_Purchase_Prob` 用了 11 月结果拟合（全量拟合，非 OOF），若用于 11 月当月触达存在泄漏；评估预期效果应以 Cell 30/22 的 OOF 指标为准，上线需滚动窗口重训重校准。

---

## 标签迁移分析

**Cell 35（markdown）**：

> 固定基期月（2019-10）分层后，逐月追踪**同一批用户**（队列：固定 10 月那批用户，看他们后续月的标签变化；7 个月用户面板，5% 抽样约 78 万用户），用**冻结的 10 月阈值**（固定沿用 10 月拟合的切分线，不让每月重算）重算后续月标签——高价值直购 / 深度互动 / 常规已购 是保持、降级、升级还是沉默？
>
> - **① 标签保持 / 转化**：各 10 月标签在后续月的标签构成（"无任何活动" = 当月完全无任何行为）；
> - **② 标签保持率**：仍保持原标签的比例随时间变化（越高 = 标签越稳定，下降越快 = 越容易转化）。
>
> > **为什么必须冻结阈值**：若不冻结，每月重新拟合的 GMM / 分位阈值会漂移（如 E_Score 阈值从 −0.07 漂到 −1.01），标签变化会被"尺子变了"污染，无法区分是用户变了还是阈值变了。冻结阈值后跨月标签才可比。

**补充（后加）**：冻结的不只是阈值数值，**E_Score 的 Z 标准化参数也冻结**（基期拟合一次 scaler，后续月 transform）——否则"尺子"（均值/标准差）每月漂移，跨月 E_Score 依旧不可比。

---

**Cell 36（代码）** —— 读取队列迁移结果：

```python
# ═══════════════════════════════════════════════════
# 队列迁移分析（结果默认读取 CSV；如需重算运行 run_cohort.py，约 10-20 分钟）
# ═══════════════════════════════════════════════════
import importlib
import config as _cfg
importlib.reload(_cfg)          # 防御：kernel 早于本版本启动时，刷新配置缓存
import analysis as _ana
importlib.reload(_ana)
import os
from pathlib import Path
import pandas as pd
import numpy as np
from config import OUTPUT_COHORT_LABELS

if not Path(OUTPUT_COHORT_LABELS).exists():
    raise SystemExit(
        '未找到 cohort_frozen_labels.csv。请先运行 `python run_cohort.py`'
        '（需 7 个月面板，约 10-20 分钟）。')
fl = pd.read_csv(OUTPUT_COHORT_LABELS)
LABELS_ORDER = ['普通浏览用户', '高潜力首购用户', '常规已购用户', '高价值直购用户', '高价值深度互动用户', '高价值高摩擦用户']
```

**逐行/逐对象解释**：

- `importlib.reload(_cfg)` / `importlib.reload(_ana)`：**强制重载模块**。**为什么**：如果 kernel 是在本项目旧版本启动的，`config` / `analysis` 的缓存还是旧的（比如旧的阈值逻辑）；reload 保证读到磁盘上的最新代码。这是 Notebook 开发的经典防御（Notebook 只 import 一次模块）。
- `OUTPUT_COHORT_LABELS`：`cohort_frozen_labels.csv` 路径（config.py 集中管理）。
- 文件存在性检查：缺失时 `SystemExit` 提示运行 `run_cohort.py`（结果默认入库，clone 可直接读）。
- `fl`：冻结标签表，结构 = `基期标签 × month × 各冻结标签占比列`（如"常规已购用户 / 2019-11 / 常规已购 16.2% / 无任何活动 40.5% / ..."）。由 `run_cohort.py` → `analysis.cohort_migration` 生成：固定 10 月基期分层 → 逐月用冻结阈值重算同一批用户标签 → 行内归一占比。
- `LABELS_ORDER`：6 个标签的展示顺序常量（供 Cell 38 图表排序）。

---

**Cell 37（markdown）** —— 解读（数字已按最新冻结标准化产物同步）：

> **① 标签保持率**（10 月各标签队列在后续月仍保持原标签的比例，越高 = 越稳定）：
>
> | 10 月标签 | 11 月 | 2020-01 | 2020-04 |
> |---|---|---:|---:|
> | 普通浏览 | 32.3% | 24.6% | 17.1% |
> | 高潜力首购 | 16.7% | 7.6% | 5.1% |
> | 常规已购 | 16.2% | 8.5% | 6.1% |
> | 高价值直购 | 10.9% | 5.4% | 1.8% |
> | 高价值深度互动 | 18.8% | 7.8% | 2.3% |
>
> 所有标签的次月保持率都只有 **10.9%~32.3%**，6 个月后大多降至 **1.8%~17.1%** —— 标签是"月度快照"，不是持久属性。
>
> 原因：标签按"当月是否购买 + 价值指数"定义，而用户购买天然是间歇性的，某月不买就会掉进"未购 / 普通浏览"或"无任何活动（完全无行为）"桶。唯一例外是**普通浏览**最"稳"（32.3%）——但它不是粘性强，而是"未购 + 低探索"的默认落点，用户不买就落回这里，是"稳定容器"而非"稳定客户"。
>
> **② 转化去向**（详见下方各标签构成图，以 11 月为例）：
> - **普通浏览**：56.5% 无任何活动、32.3% 保持、5.8% 变高潜力首购、5.5% 升级已购；
> - **高潜力首购**：34.0% 无任何活动、36.0% 变普通浏览、16.7% 保持、10.2% 变常规已购、3.1% 升级高价值；
> - **常规已购**：40.5% 无任何活动、28.5% 变普通浏览、16.2% 保持、9.2% 变高潜力首购、5.5% 升级高价值；
> - **高价值直购**：38.1% 无任何活动、19.4% 变普通浏览、18.7% 降级常规已购、10.9% 保持、7.0% 升级深度互动、5.9% 变高潜力首购；
> - **高价值深度互动**：24.0% 无任何活动、20.2% 变普通浏览、21.6% 降级常规已购、18.8% 保持、5.6% 变直购、9.9% 变高潜力首购。
>
> **③ 商业发现**：
> 1. **普通浏览是"稳定容器"**：保持率最高（32.3%）但这是"不买就落回"的默认桶，不代表用户粘性；真正要运营的是它内部浮现的高潜力信号（5.8% 变高潜力首购）；
> 2. **高价值深度互动粘性最强（购买标签中）**：无任何活动占比最低（11 月 24%）、次月购买占比最高（46.0% = 常规 21.6% + 直购 5.6% + 深度互动 18.8%）→ 适合长期会员 / 新品内测运营；
> 3. **常规已购流失风险最高**：40.5% 次月无任何活动，6 个月后约 70% 无任何活动 → 需要及时唤醒触达；
> 4. **无任何活动 ≠ 永久流失**：标签基于当月行为，用户重新活跃后会回到相应标签（如高价值直购 2020-02 有 7.9% 回到常规已购）→ 沉默高价值用户值得定向召回；
> 5. **运营名单必须按月刷新**：标签用于"当月选人 + 差异化话术"，不能一劳永逸。

**怎么读这张表**：每行是一个 10 月标签的队列，列是后续月的"保持率"（仍叫原标签的比例）。所有行都快速衰减 → 标签是月度快照。**"无任何活动"是主去向**（6 个月后 69%~78%）——但注意它 ≠ 永久流失，用户可能只是那个月没来。

---

**Cell 38（代码）** —— 各标签逐月构成堆叠图：

```python
# ═══════════════════════════════════════════════════
# 各标签逐月构成（全部 5 个 10 月标签）
# 每个标签一张堆叠图：x = 时间（月份），堆叠 = 该月用冻结阈值重算的标签占比
# 与基期标签同色的块 = 保持原标签；浅灰 = 无任何活动；深灰/蓝 = 转化为其他标签
# ═══════════════════════════════════════════════════
import importlib
import config as _cfg
importlib.reload(_cfg)          # 防御：kernel 早于本版本启动时，刷新配置缓存
import analysis as _ana
importlib.reload(_ana)
import pandas as pd
import numpy as np
from pathlib import Path
from config import OUTPUT_COHORT_LABELS

if not Path(OUTPUT_COHORT_LABELS).exists():
    raise SystemExit(
        '未找到 cohort_frozen_labels.csv。请先运行 `python run_cohort.py`'
        '（需 7 个月面板，约 10-20 分钟）。')
fl = pd.read_csv(OUTPUT_COHORT_LABELS)

ALL_LABELS = ['普通浏览用户', '高潜力首购用户', '常规已购用户', '高价值直购用户', '高价值深度互动用户']
COLOR_MAP = {
    '普通浏览用户': '#bdbdbd', '高潜力首购用户': '#9ecae1',
    '常规已购用户': '#a6bddb', '高价值直购用户': '#74c476', '高价值深度互动用户': '#fdae6b',
    '无任何活动': '#e0e0e0',
}

lab_cols = [c for c in fl.columns if c not in ('基期标签', 'month')]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
for i, lab in enumerate(ALL_LABELS):
    ax = axes[i]
    sub = fl[fl['基期标签'] == lab].set_index('month')[lab_cols].sort_index()
    bottom = np.zeros(len(sub))
    for col in lab_cols:
        vals = sub[col].to_numpy() * 100
        ax.bar(sub.index, vals, bottom=bottom, label=col,
               color=COLOR_MAP.get(col, '#999999'), width=0.6)
        for xi, v in enumerate(vals):
            if v >= 6:
                ax.text(xi, bottom[xi] + v / 2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8, color='#333333')
        bottom += vals
    ax.set_title(f'10月「{lab.replace("用户", "")}」→ 后续每月构成', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_ylabel('占比 (%)')
    ax.tick_params(axis='x', rotation=45)
axes[5].axis('off')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8, frameon=False)
plt.suptitle('全部 10 月标签在后续月份的构成变化（横轴 = 时间；与基期标签同色 = 保持；浅灰 = 无任何活动）',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.show()
```

**逐行/逐对象解释**：

- `ALL_LABELS`：5 个**基期标签**（10 月可能出现的）。**为什么没有"高价值高摩擦"**：那是 `flag_buyer_silence` 的**事后标记**（需要观察月数据），不是 10 月分层直接产出的标签，不会作为"基期标签"出现。
- `COLOR_MAP`：每个目标标签一种颜色；**"无任何活动"用浅灰 `#e0e0e0`**——视觉上"灰 = 沉默"直觉化。
- `lab_cols = [c for c in fl.columns if c not in ('基期标签', 'month')]`：其余列即各冻结标签的占比列（含"无任何活动"）。
- `plt.subplots(2, 3, figsize=(18, 9))`：2 行 3 列共 6 个格子；`.flatten()` 转成一维数组方便循环。
- 对每个基期标签 lab：
  - `sub = fl[fl['基期标签'] == lab].set_index('month')[lab_cols].sort_index()`：取该标签队列的逐月占比，索引改为月份并排序；
  - `bottom = np.zeros(len(sub))`：堆叠起点（每月的累积高度）；
  - 内层循环画每个目标标签列：`ax.bar(sub.index, vals, bottom=bottom, ...)`——`bottom` 参数让每个新块从前一块顶部开始；`width=0.6` 柱宽；颜色 `COLOR_MAP.get(col, '#999999')`（未知列兜底灰色）；
  - 文字标注：`v >= 6` 的块才写字（太小看不清、避免重叠），文字放块中心 `bottom[xi] + v/2`；
  - `bottom += vals`：累加高度 → 实现堆叠；
  - `ax.set_ylim(0, 100)`：占比满 100；
  - `ax.set_title(f'10月「{lab.replace("用户", "")}」→ 后续每月构成')`：去掉"用户"后缀让标题更短。
- `axes[5].axis('off')`：第 6 个格子留空（只有 5 个标签）。
- `fig.legend(*axes[0].get_legend_handles_labels(), ...)`：取第一个子图的图例句柄合并成整图图例，放图下方（`loc='lower center'`，`ncol=6` 排一行）。
- `plt.tight_layout(rect=[0, 0.06, 1, 0.96])`：`rect` 留出底部 6% 空间给图例。
- **怎么读图**：每个子图横轴是月份（2019-11 → 2020-04），堆叠块 = 当月冻结标签占比。与基期标签同色的块 = 保持；浅灰块 = 无任何活动（随时间长高）；其他颜色 = 转化去向。

---

## 附：整个流水线一张图

```
7个月面板(78万用户) → 10月建模(15.1万用户) → build_features 用户特征表
   → EDA（数据总览 + 相关分析）→ 确定"先分流再分层 + log1p + GMM"方法
   → segment_users 两阶段分层（未购: E_Score×Friction GMM / 已购: 价值指数上四分位 × E_Score）
   → flag_buyer_silence 跨月沉默标记（流失判定）
   → 滚动时间外验证（实验一 +7.88pp / 实验二 −19.9~−31.8pp，跨月稳定）
   → LR 基准（OOF：AUC 0.528→0.673）+ 概率分 Top-k 选人（13.30%→16.31%）
   → tracked_users_list_Nov.csv（6,792 人 = 5,408 高潜力首购 + 1,384 沉默高价值）
   → 队列迁移分析（冻结阈值 + 冻结 E_Score 标准化：保持率 10.9%~18.8% → 1.8%~6.1%，按月刷新名单）
```

---

*本文档由 main.ipynb（提交 6534786 之后 + EDA 新增单变量分布可视化、偏度量化与解读）逐 cell 生成；如 notebook 更新，请同步维护本文件。*
