"""队列迁移分析（标签视角）：固定基期分层 → 用冻结的基期阈值逐月重算标签。

产出 1 张 CSV：
- cohort_frozen_labels.csv  冻结阈值重算的标签占比（基期标签 × 月份 × 冻结标签，回答"标签是保持还是转化"）
"""
from config import BASE_MONTH, OUTPUT_COHORT_LABELS, PANEL_FILE
from analysis import cohort_migration, load_panel


def main() -> None:
    panel = load_panel(PANEL_FILE)
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
