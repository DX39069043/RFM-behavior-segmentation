"""队列迁移分析（标签视角）：固定基期分层 → 用冻结的基期阈值逐月重算标签。

产出 1 张 CSV：
- cohort_frozen_labels.csv  冻结阈值重算的标签占比（基期标签 × 月份 × 冻结标签，回答"标签是保持还是转化"）
"""
from config import BASE_MONTH, MONTHS, OUTPUT_COHORT_LABELS, PANEL_FILE
from analysis import cohort_migration, load_panel

# 分析只需要这 7 列；category_code / brand 等字符串列不读，省 IO 与内存
PANEL_COLUMNS = ['event_time', 'event_type', 'price', 'product_id', 'user_id', 'user_session', 'month']


def main() -> None:
    panel = load_panel(PANEL_FILE, months=MONTHS, columns=PANEL_COLUMNS)
    print(f'面板: {len(panel):,} 行, {panel["user_id"].nunique():,} 用户；基期 {BASE_MONTH}')
    frozen = cohort_migration(panel, BASE_MONTH)
    frozen.to_csv(OUTPUT_COHORT_LABELS, index=False, encoding='utf-8-sig')

    print('\n===== 冻结阈值标签（基期标签 → 后续月标签占比，首月示例） =====')
    fl = frozen.copy()
    fl = fl[fl['month'].eq(sorted(fl['month'].unique())[0])]
    print(fl.to_string(index=False))
    print(f'\n已保存: {OUTPUT_COHORT_LABELS.name}')


if __name__ == '__main__':
    main()
