import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def plot_ecdf(data):
    
    for key, values in data.items():
        x, y = ecdf(values)
        plt.plot(x, y, label=key)

    plt.xlabel('')
    plt.ylabel('ECDF')
    
    plt.grid(linestyle = '--', color="gray", axis='y')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

def print_min_max(df, columns):
    for col in columns:
        max_val = max(df[col].apply(lambda x: max(x)))
        min_series = df[col].apply(lambda x: min(x[x != 0]) if len(x[x != 0]) > 0 else None)
        min_series = min_series.dropna()
        min_val = min(min_series) if not min_series.empty else None

        print(f'MAX {col}-> {max_val}')
        print(f'MIN {col}-> {min_val}')


df = pd.read_parquet('../data/cic2018/cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet')


# Cut off biflows longer than MAX_BFL_DUR
MAX_BFL_DUR_ts = 10
MAX_BFL_DUR = MAX_BFL_DUR_ts * 1_000_000 # to ms

df1 = df.copy()

counter_bfl = 0

def check_bfl_threshold(row):
    global counter_bfl
    
    if sum(row['IAT']) >= MAX_BFL_DUR:
        
        counter_bfl += 1 
        curr_dur = 0
        
        for i, iat in enumerate(row['IAT']):
            curr_dur += iat
            if curr_dur >= MAX_BFL_DUR:
                                
                assert sum(row['IAT'][:i]) <= curr_dur and sum(row['IAT'][:i]) <= MAX_BFL_DUR
                row['IAT'] = row['IAT'][:i]
                
                for col in ['PL', 'DIR', 'WIN', 'FLG', 'TTL']:
                    row[col] = row[col][:i]
                break
    return row

df1 = df1.apply(check_bfl_threshold, axis=1)
print(counter_bfl)


data = dict()

def get_data(x, key):
    iat = np.atleast_1d(x['IAT'])
    data.setdefault(key, []).append(np.sum(iat[:20]) / 1_000_000)

df1.apply(lambda row: get_data(row, key='cic'), axis=1)

plot_ecdf(data)


# 确保列表列统一为 1D 列表，且长度至少为 NUM_PKTS（与 data loader 的 num_pkts 一致）
NUM_PKTS = 20
PAD_VAL = 0.5  # DIR 用 0.5，其他用 -1（与 utility.py 的 pad_value 一致）

def pad_to_len(x, col, target_len=NUM_PKTS):
    arr = np.atleast_1d(x).tolist()
    n = len(arr)
    if n >= target_len:
        return arr[:target_len]
    pad_val = 0.5 if col == 'DIR' else -1
    return arr + [pad_val] * (target_len - n)

for col in ['IAT', 'PL', 'DIR', 'WIN', 'FLG', 'TTL']:
    df1[col] = df1[col].apply(lambda x, c=col: pad_to_len(x, c))

df1.to_parquet(f'../data/cic2018_{MAX_BFL_DUR_ts}ts/cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_{MAX_BFL_DUR_ts}ts.parquet')