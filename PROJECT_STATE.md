# 项目状态速览 · UserValue&Potential

> 本文档是"上下文接力"用的压缩摘要。新开会话时，让助手先读本文件即可无缝继续。
> 目录：`D:\Code\RFM_Optimize\old_project\RFM-behavior-segmentation`
> 状态：**"跨月沉默" + 队列迁移分析（冻结阈值）均已落地，数据源统一为 7 个月面板**；分析主线见 Notebook（33 cells）。

## 一、项目一句话

基于 7 个月电商行为日志的用户价值与购买潜力分层：**未购用户 → 首购潜力识别；已购用户 → 价值分层与"跨月完全沉默"（观察月无任何行为）流失判定**；滚动时间外验证 + 两臂逻辑回归基准证明标签稳健、概率分排序优于规则；**队列迁移分析回答"10 月标签在后续月份是保持还是转化"**。

## 二、版本定位（重要）

本版本 = git 提交 `fdef422`（"重构：用户价值与购买潜力分层（UserValue&Potential）"）对应的状态 **+ 从旧工作区 `project4.0` 移植的队列迁移功能**：

- ✅ 已包含：跨月沉默流失定义（`flag_buyer_silence`）、滚动时间外验证（实验二为 3 月组）、两臂 LR 基准、概率分 Top-k 圈人、**队列迁移分析（`cohort_migration` + `segment_users` 冻结阈值参数）**、Notebook"滚动时间外验证"与"队列迁移分析"章节
- ❌ 未包含：马尔可夫状态机、纵向特征进 LR（均未做，用户倾向简化）
- 旧版本"浏览成本摩擦"（Friction_1）已彻底删除，未购臂 `Friction` 不受影响

## 三、数据

- **唯一数据源 = 7 个月用户面板**：`data/panel_7months.parquet`（515.9MB，5% 用户抽样 ≈ 78 万用户 / 2060 万行）✅ 在目录中（从 `project4.0/data/` 拷贝；也可用 `python sample_user_cohort.py` 从按月 csv.gz 重建）。Notebook 前半部分用面板 **10 月事件**（约 212 万条 / 15.1 万用户）建模，**11 月事件**（约 339 万条）验证
- **双月样本 `my_cohort_data_Oct_Nov.csv` 已删除**（2026-08 数据源统一：旧双月 1 万用户样本不再使用，git 历史可恢复）
- 队列分析产物（`run_cohort.py` 生成）：`cohort_frozen_labels.csv`（冻结阈值重算的标签占比）✅ 在目录中

## 四、文件结构（本目录）

| 文件 | 作用 |
|---|---|
| `main.ipynb` | 分析入口（33 cells）：EDA → 特征 → 分层 → 结构透视 → **滚动验证（正式验证，模型验证章节第一节）** → 汇总 → 未购臂基准 → 概率分圈人 → 名单 → 队列迁移分析（含解读 markdown + 逐月转化去向图） |
| `analysis.py` | 函数库：指标/分层（支持冻结阈值）/跨月沉默/检验/LR 基准/打分/滚动验证/`cohort_migration` |
| `config.py` | 集中配置（路径、MONTHS、BASE_MONTH、PANEL_FILE、随机种子） |
| `run_rolling.py` | 滚动验证入口 → 两张 CSV |
| `run_cohort.py` | 队列迁移入口 → 1 张冻结标签 CSV |
| `sample_user_cohort.py` | 按用户抽样生成面板 |
| `rolling_validation_results.csv` / `rolling_baseline_results.csv` | 滚动验证结果（已生成，面板口径） |
| `cohort_frozen_labels.csv` | 冻结阈值重算的标签占比（已生成） |
| `tracked_users_list_Nov.csv` | 6,792 名候选名单（10 列，含首购/复购概率分） |
| `tests/` | 单元测试（12 个测试类 / 21 个用例）：`python -m unittest discover -s tests` |
| 报告：`数据分析报告.md` / `README.md` / `项目介绍.md` | 文档（全部已收口为当前面板口径） |

> 精简记录（2026-08）：已删除 `kaggle_user_panel_sampling.ipynb`（与 `sample_user_cohort.py` 重复）；Notebook 删除 7 个失效/冗余 cell（偏度叙述、单变量分布图、双月实验二说明、敏感性分析；结构透视 cell 保留）；LR 基准 cell 删除校准曲线/Top-k 图，只保留指标输出；`analysis.py` 删除 `Log_Friction` 冗余别名列。随后从 `project4.0` 移植队列迁移（`cohort_migration` + 冻结阈值），`data/panel_7months.parquet` 已就位；**再按用户要求删除状态迁移表整套内容**（migration/cumulative/revival 及其 CSV 与 Notebook ①②③ 图），队列迁移只保留冻结标签（标签保持/转化）主线；**统一数据源**：删除双月样本 `my_cohort_data_Oct_Nov.csv`，Notebook 全部改用 7 个月面板（10 月建模 / 11 月验证），滚动时间外验证升级为正式验证（模型验证章节第一节），原独立"实验一"cell 删除（其 flag_buyer_silence 逻辑移入汇总报告 cell）；`load_panel` 支持按月过滤（只读 10/11 月，加载从 ~2.5 分钟降至 ~40 秒）；`requirements.txt` 补充 pyarrow。

## 五、核心方法（重要约定）

- **E_Score** = 页数/时长/会话的 log1p+Z 等权均值（**语义固定为"描述性探索度"**）
- **Friction（未购臂）** = 加购未买去重商品数（log1p）；GMM 阈值，零膨胀自动退化为"有加购即高摩擦"
- **已购臂"高摩擦" = 跨月完全沉默**（`flag_buyer_silence`：基期高价值买家在观察月无任何行为 = 流失判定）
- **Value_Index** = log1p(消费)+log1p(频次)−log1p(近度)，已购内上四分位 = 高价值；VIP 再按 E_Score 分深度互动/直购
- **滚动验证**：实验一（未购臂）2 月对（基期→观察月）；实验二（已购臂）3 月组（基期→沉默→验证月 t+2，避免"沉默月=结果月"循环）
- **队列迁移（`cohort_migration`）**：固定基期（BASE_MONTH='2019-10'）分层 → 逐月用**冻结基期阈值**重算同一批用户的标签（`segment_users(thresholds=...)`；**E_Score 的 Z 标准化参数也在基期拟合后冻结复用**，防止"尺子"每月漂移），返回"基期标签 × 月份 × 冻结标签占比"（'无任何活动' = 当月完全无行为）；只做标签保持/转化，不含状态迁移表
- `rate_test`：卡方（Yates），期望频数为 0 时回退 Fisher；空组返回 NaN
- 分层/阈值只在**基期月**拟合，验证月不参与（防泄漏）

## 六、关键结果（面板口径，见 rolling_validation_results.csv / cohort_frozen_labels.csv）

- **滚动时间外验证（正式验证）**：实验一（未购臂）2019-10→11 组：高潜力首购 5,408 vs 普通浏览 12.85 万，11 月购买率 **13.30% vs 5.41%**（购买率差 +7.88pp，p=2.2e-131）；6/6 个月对显著，购买率差 **+4.5~+9.3pp**，跨月稳定
- **实验二**（沉默=流失）：5/5 个月组极显著，购买率差 **−19.9~−31.8pp**（p<1e-68），方向一致——"跨月完全沉默"是可靠流失信号
- **LR 基准**：未购臂 AUC 0.528→0.673、Top-k 首购率 13.30%→16.31%；滚动口径未购臂 AUC 0.673~0.722、已购臂 0.606~0.646，均稳定优于规则
- **名单**：6,792 人 = 5,408 高潜力首购 + 1,384 沉默高价值（10 列）
- **队列迁移（10 月基期，关键数字）**：
  - 已购三标签月度保持率低：次月 10.9%~18.8%（直购 10.9% / 常规 16.2% / 深度互动 18.8%），6 个月后降至 1.8%~6.1%（标签是"月度快照"，需按月刷新）
  - "无任何活动"（完全无行为）是主去向：所有标签占比 6 个月后升至 69%~78%
  - 高价值深度互动粘性最强：无任何活动占比最低（11 月 24%）、次月购买占比最高（46.0% = 常规 21.6% + 直购 5.6% + 深度互动 18.8%）
  - 常规已购流失风险最高：40.5% 次月无任何活动

## 七、Git 状态

- 本地 git 仓库，与 GitHub `DX39069043/RFM-behavior-segmentation` 的 main 同步（2026-08 提交"精简+队列迁移+数据源统一"版本）
- git 位于 `D:\Git`（命令需 `$env:PATH="D:\Git\cmd;$env:PATH"` 或用全路径）

## 八、已知问题 / 下一步（供新会话决定方向）

1. 可选方向（如需继续迭代，**用户倾向简化，做之前先确认**）：冻结标签的马尔可夫转移矩阵、纵向特征进 LR、队列迁移的敏感性分析（不同基期月）
2. **队列迁移重算耗时**：`run_cohort.py` 全量约 10-20 分钟（逐月 build_features + GMM）；Notebook 默认读现成 CSV

## 九、环境备忘

- Python 3.12（Miniconda），pyarrow 已装；requirements.txt 已锁版本（含 pyarrow）
- pwsh 命令需完整访问权限；`$env:PYTHONPATH='D:\Code\RFM_Optimize\old_project\RFM-behavior-segmentation'` 供脚本 import analysis
- 运行顺序：`python run_rolling.py`（需面板）→ `python run_cohort.py`（需面板）→ Notebook 主流程（面板 10/11 月，行过滤加载约 40 秒）
