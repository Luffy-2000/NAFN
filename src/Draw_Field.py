import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 读取数据
df = pd.read_parquet('/home/zhaozijian/fscil-nids-main/data/iot_nid/iot-nidd_100pkts_6f_clean.parquet')

# 选择需要绘制的字段
# fields = [col for col in df.columns if col.startswith('SCALED_')]
fields = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL', 'LOAD']

# 画出每个字段的频率分布图
for field in fields:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[field].explode(), bins=50, kde=True)
    plt.title(f'Distribution of {field}')
    plt.xlabel(field)
    plt.ylabel('Frequency')
    plt.savefig(f'./PDF/{field}.pdf', format='pdf', bbox_inches='tight')
    # plt.close()
