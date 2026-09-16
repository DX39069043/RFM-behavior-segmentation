"""滚动时间外验证：读取 7 个月用户面板，逐对执行 训练月 t → 验证月 t+1。

产出 rolling_validation_results.csv（实验一/二的逐组率对比）。
注：LR 基准（nonbuyer_baseline / buyer_baseline）已移出滚动验证主流程，
需要时可单独调用；历史基准结果保留在 rolling_baseline_results.csv。
"""
from config import MONTHS, OUTPUT_ROLLING, PANEL_FILE
from analysis import build_month_tables, load_panel, rolling_validation

# 分析只需要这 7 列；category_code / brand 等字符串列不读，省 IO 与内存
PANEL_COLUMNS = ['event_time', 'event_type', 'price', 'product_id',
                 'user_id', 'user_session', 'month']


def main() -> None:
    # 读入 7 个月面板（按月份行过滤 + 列裁剪，避免读全量）
    panel = load_panel(PANEL_FILE, months=MONTHS, columns=PANEL_COLUMNS)
    print(f'面板: {len(panel):,} 行, {panel["user_id"].nunique():,} 用户')

    # 面板 → 月份表（特征 + 当月独立分层；不落盘，本次运行内存里用完即弃）
    tables = build_month_tables(panel, MONTHS)

    # 逐月对执行时间外验证（实验一：未购人群；实验二：已购沉默人群）
    res = rolling_validation(tables, MONTHS)
    rates = res['验证表']

    # 结果落盘（utf-8-sig 带 BOM，Excel 打开中文不乱码）
    rates.to_csv(OUTPUT_ROLLING, index=False, encoding='utf-8-sig')

    print('\n===== 滚动验证表（实验一/二跨月） =====')
    print(rates.to_string(index=False))
    print(f'\n已保存: {OUTPUT_ROLLING.name}')


if __name__ == '__main__':
    main()
