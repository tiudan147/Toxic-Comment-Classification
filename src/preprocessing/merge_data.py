import os

import pandas as pd

"""
Chỉ merge, ko có gì ngoài merge
"""

raw_data_dir = r"..\..\dataset\raw_data" 

clean_files = ["df_dev_clean.csv", "df_test_clean.csv", "df_train_clean.csv"]
vihsd_files = ["vihsd_test.csv", "vihsd_train.csv", "vihsd_val.csv"]

dfs = []
for file in clean_files:
    df = pd.read_csv(os.path.join(raw_data_dir, file),usecols = ['content_clean', 'Toxicity'])
    df_sub = pd.DataFrame({
        "text": df["content_clean"],
        "label": df["Toxicity"]
    })
    dfs.append(df_sub)

for file in vihsd_files:
    df = pd.read_csv(os.path.join(raw_data_dir, file),usecols = ['cmt_col', 'labels'])
    df_sub = pd.DataFrame({
        "text": df["cmt_col"],
        "label": df["labels"]
    })
    dfs.append(df_sub)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(os.path.join(raw_data_dir, "merged_dataset.csv"))