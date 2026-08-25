"""本地版用户级抽样：对已下载的按月 CSV 抽取“用户面板”。

与 kaggle_user_panel_sampling.ipynb 完全相同的抽样逻辑（user_id 确定性哈希），
供你已把全量按月文件下载到本地时使用。

用法:
    python sample_user_cohort.py --input-dir ./raw_data \\
        --out panel_7months.parquet --keep-per-mille 50

说明:
    - 每个用户要么全部月份都在面板里，要么完全不在（跨文件一致），可跨月追踪；
    - 默认保留 5% 用户；若每月行数不足 10 万可调大 --keep-per-mille（如 100 = 10%）；
    - 若未安装 pyarrow，将回退输出 CSV（体积较大）。
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

COLS = ['event_time', 'event_type', 'product_id', 'category_id',
        'category_code', 'brand', 'price', 'user_id', 'user_session']


def keep_user(uid: pd.Series, keep_per_mille: int) -> np.ndarray:
    """只依赖 user_id 数值的确定性抽样：同一用户在所有月份得到相同结论。

    全量原始文件里 user_id 可能被 pandas 读成 float64（存在缺失值时），直接
    to_numpy(dtype=uint64) 会把缺失值转成垃圾大数、破坏哈希，因此先统一转数值。
    """
    numeric = pd.to_numeric(uid, errors='coerce')
    valid = numeric.notna().to_numpy()
    u = numeric.to_numpy(dtype=np.int64).astype(np.uint64)
    h = (u * np.uint64(2654435761)) % np.uint64(1000)
    return valid & (h < np.uint64(keep_per_mille))


def main() -> None:
    ap = argparse.ArgumentParser(description='按用户抽取跨月用户面板')
    ap.add_argument('--input-dir', required=True, help='包含按月 CSV 的目录')
    ap.add_argument('--out', default='panel_7months.parquet', help='输出文件路径')
    ap.add_argument('--keep-per-mille', type=int, default=50,
                    help='每千个用户保留数（50 = 5%）')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.csv*')))
    if not files:
        raise SystemExit(f'--input-dir 下没有找到 CSV/CSV.GZ 文件: {args.input_dir}')
    print('输入文件:')
    for f in files:
        print(f'  {os.path.basename(f)}  {os.path.getsize(f) / 1e9:.2f} GB')

    kept = []
    for f in files:
        total_rows = keep_rows = bad_id = 0
        # pandas 会按扩展名自动识别 gzip（compression='infer'）
        for chunk in pd.read_csv(f, usecols=COLS, chunksize=2_000_000):
            sel = keep_user(chunk['user_id'], args.keep_per_mille)
            total_rows += len(chunk)
            keep_rows += int(sel.sum())
            bad_id += int(((~sel) & chunk['user_id'].isna()).sum())
            if sel.any():
                kept.append(chunk.loc[sel])
        print(f'{os.path.basename(f)}: 全量 {total_rows:,} 行 → 保留 {keep_rows:,} 行 '
              f'(user_id 缺失 {bad_id:,} 行)')

    panel = pd.concat(kept, ignore_index=True)
    panel['month'] = panel['event_time'].str[:7]

    print(f'\n面板总行数: {len(panel):,}')
    print(f'面板用户数: {panel["user_id"].nunique():,}')
    print('\n每月事件数:')
    print(panel.groupby('month').size().to_string())

    try:
        import pyarrow  # noqa: F401
        panel.to_parquet(args.out, index=False)
        print(f'\n已保存(parquet): {args.out}')
    except ImportError:
        panel.to_csv(args.out, index=False, encoding='utf-8')
        print(f'\n未安装 pyarrow，已回退保存(CSV): {args.out}')


if __name__ == '__main__':
    main()
