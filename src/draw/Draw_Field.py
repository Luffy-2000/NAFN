import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read data
# df = pd.read_parquet('/home/zhaozijian/fscil-nids-main/data/iot_nid/iot-nidd_100pkts_6f_clean.parquet')
# df = pd.read_parquet('/home/zhaozijian/fscil-nids-main-git/data/edge_iiot/edge-iiot_100pkts_6f_1p-mt100k_benign_class_clean.parquet')

df = pd.read_parquet('/home/zhaozijian/fscil-nids-main-git/data/cic2018/cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet')


print("Reading completed")
exit()
# Select fields to plot
# fields = [col for col in df.columns if col.startswith('SCALED_')]
fields = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL', 'LOAD']

# Draw frequency distribution for each field
for field in fields:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[field].explode(), bins=50, kde=True)
    plt.title(f'Distribution of {field}')
    plt.xlabel(field)
    plt.ylabel('Frequency')
    plt.savefig(f'./PDF/{field}.pdf', format='pdf', bbox_inches='tight')
    # plt.close()
