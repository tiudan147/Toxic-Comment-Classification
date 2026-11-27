"""
Thực nghiệm chứng minh lựa chọn 4 tham số trong train.py:
    alpha=0.5 (Naive Bayes), C=1.0 (Linear SVM), max_features=5000 (TF-IDF), vector_size=150 (Word2Vec).

Chạy:
    python test_tham_so.py
"""

import argparse
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.features import Word2VecVectorizer, get_tfidf_vectorizer
from src.models import get_naive_bayes_model, get_svm_model

CHOSEN = {"max_features": 5000, "alpha": 0.5, "C": 1.0, "vector_size": 150}


def eval_model(model, X_tr, y_tr, X_va, y_va):
    """Huấn luyện model và trả về các chỉ số đánh giá trên tập Validation."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    return {
        "accuracy": round(accuracy_score(y_va, pred), 4),
        "macro_f1": round(f1_score(y_va, pred, average="macro", zero_division=0), 4),
        "toxic_f1": round(f1_score(y_va, pred, pos_label=1, zero_division=0), 4),
        "precision": round(precision_score(y_va, pred, pos_label=1, zero_division=0), 4),
        "recall": round(recall_score(y_va, pred, pos_label=1, zero_division=0), 4),
        "time_s": round(time.perf_counter() - t0, 3),
    }


def run_sweep(title, param_name, grid, step_fn):
    """Chạy khảo sát 1 tham số qua danh sách giá trị và trả về DataFrame."""
    print(f"\n=== {title} ===")
    rows = []
    for val in grid:
        metrics = step_fn(val)
        label = "All" if val is None else val
        chosen = val == CHOSEN[param_name]
        rows.append({param_name: label, "chosen": chosen, **metrics})
        mark = "  <-- CHỌN" if chosen else ""
        print(f"  {param_name}={str(label):<6} | MacroF1={metrics['macro_f1']:.4f} | ToxicF1={metrics['toxic_f1']:.4f} | Time={metrics['time_s']}s{mark}")
    return pd.DataFrame(rows)


def plot_experiments(experiments, save_path):
    """Vẽ biểu đồ 2x2 trực quan hóa 4 thí nghiệm."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Chứng minh lựa chọn tham số trong train.py (Validation)", fontsize=14, fontweight="bold")

    for ax, (title, df, col, use_log) in zip(axes.flat, experiments):
        x = list(range(len(df))) if col == "max_features" else df[col]
        ax.plot(x, df["macro_f1"], marker="o", linewidth=2, label="Macro F1")
        ax.plot(x, df["toxic_f1"], marker="^", linestyle="--", label="Toxic F1")
        if "nb_macro_f1" in df:
            ax.plot(x, df["nb_macro_f1"], marker="s", linestyle=":", label="NB Macro F1")

        chosen_row = df[df["chosen"]].iloc[0]
        cx = df.index[df["chosen"]].tolist()[0] if col == "max_features" else chosen_row[col]
        ax.axvline(cx, color="red", linestyle=":", linewidth=1.6, label=f"Chọn {col}={chosen_row[col]}")
        ax.scatter([cx], [chosen_row["macro_f1"]], color="red", s=90, zorder=5)

        if col == "max_features":
            ax.set_xticks(x)
            ax.set_xticklabels(df[col].tolist(), rotation=25)
        elif use_log:
            ax.set_xscale("log")
        else:
            ax.set_xticks(df[col].tolist())

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("F1 Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="dataset/processed_data/train.csv")
    parser.add_argument("--val-path", default="dataset/processed_data/valid.csv")
    parser.add_argument("--out-dir", default="results/param_proof")
    args = parser.parse_args()

    val_path = args.val_path
    if not os.path.exists(val_path) and os.path.exists(val_path.replace("valid.csv", "val.csv")):
        val_path = val_path.replace("valid.csv", "val.csv")

    df_tr = pd.read_csv(args.train_path).dropna(subset=["text"])
    df_va = pd.read_csv(val_path).dropna(subset=["text"])
    X_tr_txt, y_tr = df_tr["text"], df_tr["label"]
    X_va_txt, y_va = df_va["text"], df_va["label"]
    print(f"Train: {len(df_tr)} | Val: {len(df_va)}")

    # 1. Khảo sát max_features (TF-IDF) với alpha=0.5 và C=1.0
    def step_mf(mf):
        vec = get_tfidf_vectorizer(max_features=mf)
        X_tr, X_va = vec.fit_transform(X_tr_txt), vec.transform(X_va_txt)
        svm_m = eval_model(get_svm_model(C=CHOSEN["C"], class_weight="balanced"), X_tr, y_tr, X_va, y_va)
        nb_m = eval_model(get_naive_bayes_model(alpha=CHOSEN["alpha"]), X_tr, y_tr, X_va, y_va)
        return {**svm_m, "nb_macro_f1": nb_m["macro_f1"], "dim": X_tr.shape[1]}

    df_mf = run_sweep("1. TF-IDF max_features (SVM C=1.0, NB alpha=0.5)", "max_features",
                      [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, None], step_mf)

    # Vector TF-IDF chuẩn (5000 chiều) dùng chung cho Thí nghiệm 2 & 3
    tfidf = get_tfidf_vectorizer(max_features=CHOSEN["max_features"])
    X_tr_tfidf, X_va_tfidf = tfidf.fit_transform(X_tr_txt), tfidf.transform(X_va_txt)

    # 2. Khảo sát alpha (Naive Bayes)
    df_alpha = run_sweep("2. Naive Bayes alpha (TF-IDF 5000)", "alpha",
                         [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
                         lambda a: eval_model(get_naive_bayes_model(alpha=a), X_tr_tfidf, y_tr, X_va_tfidf, y_va))

    # 3. Khảo sát C (Linear SVM TF-IDF)
    df_c = run_sweep("3. Linear SVM C (TF-IDF 5000, balanced)", "C",
                     [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
                     lambda c: eval_model(get_svm_model(C=c, class_weight="balanced"), X_tr_tfidf, y_tr, X_va_tfidf, y_va))

    # 4. Khảo sát vector_size (Word2Vec + Linear SVM C=1.0)
    def step_w2v(vs):
        t0 = time.perf_counter()
        w2v = Word2VecVectorizer(vector_size=vs, window=5, min_count=2, epochs=10).fit(X_tr_txt)
        X_tr_w, X_va_w = w2v.transform(X_tr_txt), w2v.transform(X_va_txt)
        m = eval_model(get_svm_model(C=CHOSEN["C"], class_weight="balanced"), X_tr_w, y_tr, X_va_w, y_va)
        m["time_s"] = round(time.perf_counter() - t0, 3)
        return m

    df_w2v = run_sweep("4. Word2Vec vector_size (SVM C=1.0)", "vector_size", [50, 100, 150, 200, 300], step_w2v)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, df in [("exp1_max_features", df_mf), ("exp2_nb_alpha", df_alpha),
                     ("exp3_svm_C", df_c), ("exp4_w2v_vector_size", df_w2v)]:
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)

    plot_experiments([
        ("1. TF-IDF max_features", df_mf, "max_features", False),
        ("2. Naive Bayes alpha", df_alpha, "alpha", True),
        ("3. Linear SVM C (TF-IDF)", df_c, "C", True),
        ("4. Word2Vec vector_size", df_w2v, "vector_size", False),
    ], os.path.join(args.out_dir, "param_justification_charts.png"))

    print(f"\n[XONG] Đã lưu CSV và biểu đồ tại: {args.out_dir}/")


if __name__ == "__main__":
    main()
