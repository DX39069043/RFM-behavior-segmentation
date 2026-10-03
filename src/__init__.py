"""电商用户行为分层与差异化运营分析：核心分析包。

模块
----
- ``config``            ：集中配置（路径、月份、随机种子、会话阈值）
- ``analysis``          ：特征构建、GMM 阈值、分层、检验、滚动验证、队列迁移等全部计算逻辑
- ``run_rolling``       ：滚动时间外验证入口（``python -m src.run_rolling``）
- ``run_cohort``        ：队列迁移分析入口（``python -m src.run_cohort``）
- ``sample_user_cohort``：按 user_id 哈希抽样生成 7 个月用户面板

Notebook / 测试从项目根目录导入：``from src.analysis import build_features``。
"""
