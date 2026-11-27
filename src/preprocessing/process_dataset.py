import pandas as pd 
from process_text import process_text
from sklearn.model_selection import train_test_split

def process_dataset(df):
    # Xử lý text(gọi hàm từ process_text.py) và xử lý label
    processed_list = []

    for row in df.itertuples():
        text = process_text(row.text)
        label = 1 if row.label == 2 else row.label 
        processed_list.append([str(text), int(label)])

    processed_df = pd.DataFrame(processed_list,columns=["text", "label"])

    return processed_df

# Chia dữ liệu ra tập train, test và lưu lại trong dataset
# df = pd.read_csv(r"..\..\dataset\raw_data\test.csv")
df = pd.read_csv(r"..\..\dataset\raw_data\merged_dataset.csv")

processed_df = process_dataset(df)

processed_df = processed_df.dropna(subset=['text', 'label']) # xóa dòng bị trống

processed_df = processed_df.drop_duplicates(subset=['text'], keep='first') # xóa dòng bị lặp, giữ cái đầu tiên

# processed_df.to_csv(r"..\..\dataset\processed_data\test.csv")

X = processed_df["text"]
y = processed_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

valid = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

train.to_csv(r"..\..\dataset\processed_data\train.csv")
valid.to_csv(r"..\..\dataset\processed_data\valid.csv")