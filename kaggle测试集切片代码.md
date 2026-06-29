```python
import pandas as pd
import random

print("第一步：从10月份随便抓取 10,000 个真实的用户ID...")
# 先读一小部分数据，拿到真实存在的 user_id 名单
temp_df = pd.read_csv('/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv', nrows=2000000)
all_users = temp_df['user_id'].dropna().unique().tolist()

# 随机抽样 10,000 个用户作为我们的“实验小白鼠”
sampled_users = set(random.sample(all_users, 10000))
print(f"成功锁定 10,000 个目标用户！开始在全量数据中追踪他们...\n")

# 第二步：分块读取大文件，只捞出这 10,000 个人的数据
file_paths = [
    '/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv',
    '/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Nov.csv' # 假设你想跨越这两个月验证
]

# 用于存放我们提取好的干净数据
final_data = []

for file in file_paths:
    print(f"正在扫描文件: {file} ...")
    # chunksize=1000000 表示每次只读 100万行，绝不撑爆内存
    for chunk in pd.read_csv(file, chunksize=1000000):
        # 核心逻辑：只保留 user_id 在我们名单里的行！
        filtered_chunk = chunk[chunk['user_id'].isin(sampled_users)]
        final_data.append(filtered_chunk)

print("\n第三步：拼合跨月数据并保存...")
# 把碎片拼成一个完整的 DataFrame
my_perfect_dataset = pd.concat(final_data, ignore_index=True)

# 按照时间排个序，强迫症福音
my_perfect_dataset = my_perfect_dataset.sort_values(by=['user_id', 'event_time'])

# 保存到本地输出，大概只有几十到一百多兆，完美！
my_perfect_dataset.to_csv('my_cohort_data_Oct_Nov.csv', index=False)
print(f"大功告成！提纯后的数据集行数：{len(my_perfect_dataset)}，现在可以下载了！")
```

