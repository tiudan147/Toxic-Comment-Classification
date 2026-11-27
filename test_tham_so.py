import pandas as pd
from src.evaluation import evaluate_model
from src.features import Word2VecVectorizer, get_tfidf_vectorizer
from src.models import get_naive_bayes_model, get_svm_model

df_train = pd.read_csv("dataset/processed_data/train.csv").dropna(subset=["text"])
df_val = pd.read_csv("dataset/processed_data/valid.csv").dropna(subset=["text"])
X_train_text, y_train = df_train["text"], df_train["label"]
X_val_text, y_val = df_val["text"], df_val["label"]

# 1. Thử nghiệm max_features cho TF-IDF
for mf in [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, None]:
    tfidf = get_tfidf_vectorizer(max_features=mf)
    X_tr = tfidf.fit_transform(X_train_text)
    X_va = tfidf.transform(X_val_text)
    svm = get_svm_model(C=1.0, class_weight="balanced").fit(X_tr, y_train)
    nb = get_naive_bayes_model(alpha=0.5).fit(X_tr, y_train)
    evaluate_model(y_val, svm.predict(X_va), model_name=f"SVM (max_features={mf})")
    evaluate_model(y_val, nb.predict(X_va), model_name=f"NB (max_features={mf})")

# Chuẩn bị TF-IDF cố định (5000) cho thí nghiệm 2 và 3
tfidf = get_tfidf_vectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf = tfidf.transform(X_val_text)

# 2. Thử nghiệm alpha cho Naive Bayes
for alpha in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]:
    nb = get_naive_bayes_model(alpha=alpha).fit(X_train_tfidf, y_train)
    evaluate_model(y_val, nb.predict(X_val_tfidf), model_name=f"NB (alpha={alpha})")

# 3. Thử nghiệm C cho Linear SVM
for c in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
    svm = get_svm_model(C=c, class_weight="balanced").fit(X_train_tfidf, y_train)
    evaluate_model(y_val, svm.predict(X_val_tfidf), model_name=f"SVM (C={c})")

# 4. Thử nghiệm vector_size cho Word2Vec
for vs in [50, 100, 150, 200, 300]:
    w2v = Word2VecVectorizer(vector_size=vs, window=5, min_count=2, epochs=10).fit(X_train_text)
    X_tr_w2v = w2v.transform(X_train_text)
    X_va_w2v = w2v.transform(X_val_text)
    svm = get_svm_model(C=1.0, class_weight="balanced").fit(X_tr_w2v, y_train)
    evaluate_model(y_val, svm.predict(X_va_w2v), model_name=f"SVM W2V (vector_size={vs})")
