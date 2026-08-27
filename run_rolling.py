"""滚动时间外验证：读取 7 个月用户面板，逐对执行 训练月 t → 验证月 t+1。

产出 rolling_validation_results.csv（两项对照实验）与 rolling_baseline_results.csv（两臂 LR 基准 AUC）。
"""
from config import MONTHS, OUTPUT_ROLLING, OUTPUT_ROLLING_BASELINE, PANEL_FILE
from analysis import load_panel, rolling_validation

# 分析只需要这 7 列；category_code / brand 等字符串列不读，省 IO 与内存
PANEL_COLUMNS = ['event_time', 'event_type', 'price', 'product_id', 'user_id', 'user_session', 'month']


def main() -> None:
    panel = load_panel(PANEL_FILE, months=MONTHS, columns=PANEL_COLUMNS)
    print(f'面板: {len(panel):,} 行, {panel["user_id"].nunique():,} 用户')
    res = rolling_validation(panel)
    res['验证表'].to_csv(OUTPUT_ROLLING, index=False, encoding='utf-8-sig')
    res['基准表'].to_csv(OUTPUT_ROLLING_BASELINE, index=False, encoding='utf-8-sig')

    print('\n===== 滚动验证表（实验一/二跨月） =====')
    print(res['验证表'].to_string(index=False))
    print('\n===== 滚动基准表（两臂 LR） =====')
    print(res['基准表'].to_string(index=False))
    print(f'\n已保存: {OUTPUT_ROLLING.name} / {OUTPUT_ROLLING_BASELINE.name}')


if __name__ == '__main__':
    main()
