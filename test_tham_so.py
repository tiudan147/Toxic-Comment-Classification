"""
================================================================================
THỰC NGHIỆM CHỨNG MINH LỰA CHỌN THAM SỐ TRONG train.py
File: test_tham_so.py
================================================================================
MỤC ĐÍCH:
    Chạy thực nghiệm khảo sát (Ablation / Sensitivity Analysis) trên tập
    Validation để chứng minh cơ sở thực nghiệm cho 4 tham số được chọn trong
    `train.py`:
        1. `max_features = 5000`  (TF-IDF Vectorizer)
        2. `alpha = 0.5`          (Multinomial Naive Bayes trên TF-IDF)
        3. `C = 1.0`              (Linear SVM trên TF-IDF, class_weight="balanced")
        4. `vector_size = 150`    (Word2Vec Vectorizer + Linear SVM)

    Nguyên tắc thực nghiệm:
        Khi khảo sát một tham số, giữ nguyên toàn bộ các tham số khác đúng
        như cấu hình trong `train.py` để cô lập ảnh hưởng của từng tham số.

KẾT QUẢ XUẤT RA (trong thư mục `results/param_proof/`):
    - `exp1_max_features.csv`           : Kết quả khảo sát `max_features` (NB & SVM)
    - `exp2_nb_alpha.csv`               : Kết quả khảo sát `alpha` (Naive Bayes)
    - `exp3_svm_C.csv`                  : Kết quả khảo sát `C` (Linear SVM TF-IDF)
    - `exp4_w2v_vector_size.csv`        : Kết quả khảo sát `vector_size` (Word2Vec + SVM)
    - `param_justification_charts.png`  : Biểu đồ 2x2 trực quan hóa 4 thí nghiệm
    - `summary_report.md`               : Báo cáo tổng hợp số liệu & luận giải

CÁCH CHẠY:
    python test_tham_so.py
    python test_tham_so.py --train-path dataset/processed_data/train.csv --val-path dataset/processed_data/valid.csv
================================================================================
"""

import argparse
import os
import sys
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Đảm bảo terminal in đúng tiếng Việt UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.features import get_tfidf_vectorizer, Word2VecVectorizer
from src.models import get_naive_bayes_model, get_svm_model


# Cấu hình chuẩn trong train.py (Baseline / Chosen params)
CHOSEN_PARAMS = {
    "max_features": 5000,
    "alpha": 0.5,
    "C": 1.0,
    "vector_size": 150,
}

# Lưới giá trị khảo sát cho từng tham số
MAX_FEATURES_GRID = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, None]
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]
C_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
VECTOR_SIZE_GRID = [50, 100, 150, 200, 300]


def compute_metrics(y_true, y_pred):
    """Tính các chỉ số đánh giá giống hệt src/evaluation/metrics.py."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "toxic_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "toxic_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "toxic_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def resolve_data_paths(train_path, val_path):
    """Kiểm tra đường dẫn dữ liệu; hỗ trợ fallback valid.csv <-> val.csv nếu cần."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Không tìm thấy tập Train tại '{train_path}'. "
            "Hãy chạy tiền xử lý dữ liệu hoặc truyền --train-path."
        )
    if not os.path.exists(val_path):
        alt_val = val_path.replace("valid.csv", "val.csv")
        if alt_val != val_path and os.path.exists(alt_val):
            val_path = alt_val
        else:
            raise FileNotFoundError(
                f"Không tìm thấy tập Validation tại '{val_path}'. "
                "Hãy chạy tiền xử lý dữ liệu hoặc truyền --val-path."
            )
    return train_path, val_path


def run_exp1_max_features(X_train_text, y_train, X_val_text, y_val):
    """Thí nghiệm 1: Khảo sát max_features của TF-IDF (giữ alpha=0.5, C=1.0)."""
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 1] Khảo sát TF-IDF `max_features` (alpha=0.5, C=1.0)")
    print("=" * 70)

    rows = []
    for mf in MAX_FEATURES_GRID:
        t0 = time.perf_counter()
        vec = get_tfidf_vectorizer(max_features=mf)
        X_tr = vec.fit_transform(X_train_text)
        X_va = vec.transform(X_val_text)
        vec_time = time.perf_counter() - t0
        actual_dim = X_tr.shape[1]

        # 1a. Naive Bayes với alpha=0.5
        t1 = time.perf_counter()
        nb = get_naive_bayes_model(alpha=CHOSEN_PARAMS["alpha"])
        nb.fit(X_tr, y_train)
        nb_pred = nb.predict(X_va)
        nb_time = time.perf_counter() - t1
        nb_m = compute_metrics(y_val, nb_pred)

        # 1b. Linear SVM với C=1.0
        t2 = time.perf_counter()
        svm = get_svm_model(C=CHOSEN_PARAMS["C"], class_weight="balanced")
        svm.fit(X_tr, y_train)
        svm_pred = svm.predict(X_va)
        svm_time = time.perf_counter() - t2
        svm_m = compute_metrics(y_val, svm_pred)

        label_mf = "All" if mf is None else str(mf)
        is_chosen = mf == CHOSEN_PARAMS["max_features"]
        mark = " <-- CHỌN" if is_chosen else ""
        print(
            f"  max_features={label_mf:<6} (dim={actual_dim:<6}) | "
            f"NB MacroF1={nb_m['macro_f1']:.4f} | "
            f"SVM MacroF1={svm_m['macro_f1']:.4f} | "
            f"Time={vec_time + svm_time:.2f}s{mark}"
        )

        rows.append({
            "max_features_param": label_mf,
            "actual_features": actual_dim,
            "is_chosen": is_chosen,
            "tfidf_time_s": round(vec_time, 3),
            "nb_macro_f1": round(nb_m["macro_f1"], 4),
            "nb_toxic_f1": round(nb_m["toxic_f1"], 4),
            "nb_accuracy": round(nb_m["accuracy"], 4),
            "nb_time_s": round(nb_time, 3),
            "svm_macro_f1": round(svm_m["macro_f1"], 4),
            "svm_toxic_f1": round(svm_m["toxic_f1"], 4),
            "svm_toxic_precision": round(svm_m["toxic_precision"], 4),
            "svm_toxic_recall": round(svm_m["toxic_recall"], 4),
            "svm_accuracy": round(svm_m["accuracy"], 4),
            "svm_time_s": round(svm_time, 3),
        })

    return pd.DataFrame(rows)


def run_exp2_nb_alpha(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """Thí nghiệm 2: Khảo sát alpha của Multinomial Naive Bayes (max_features=5000)."""
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 2] Khảo sát Naive Bayes `alpha` (max_features=5000)")
    print("=" * 70)

    rows = []
    for alpha in ALPHA_GRID:
        t0 = time.perf_counter()
        nb = get_naive_bayes_model(alpha=alpha)
        nb.fit(X_train_tfidf, y_train)
        pred = nb.predict(X_val_tfidf)
        elapsed = time.perf_counter() - t0
        m = compute_metrics(y_val, pred)

        is_chosen = alpha == CHOSEN_PARAMS["alpha"]
        mark = " <-- CHỌN" if is_chosen else ""
        print(
            f"  alpha={alpha:<6} | MacroF1={m['macro_f1']:.4f} | "
            f"ToxicF1={m['toxic_f1']:.4f} (P={m['toxic_precision']:.4f}, R={m['toxic_recall']:.4f}) | "
            f"Acc={m['accuracy']*100:.2f}%{mark}"
        )

        rows.append({
            "alpha": alpha,
            "is_chosen": is_chosen,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "toxic_f1": round(m["toxic_f1"], 4),
            "toxic_precision": round(m["toxic_precision"], 4),
            "toxic_recall": round(m["toxic_recall"], 4),
            "train_time_s": round(elapsed, 3),
        })

    return pd.DataFrame(rows)


def run_exp3_svm_c(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """Thí nghiệm 3: Khảo sát C của Linear SVM trên TF-IDF (max_features=5000)."""
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 3] Khảo sát Linear SVM `C` (TF-IDF max_features=5000, balanced)")
    print("=" * 70)

    rows = []
    for c_val in C_GRID:
        t0 = time.perf_counter()
        svm = get_svm_model(C=c_val, class_weight="balanced")
        svm.fit(X_train_tfidf, y_train)
        pred = svm.predict(X_val_tfidf)
        elapsed = time.perf_counter() - t0
        m = compute_metrics(y_val, pred)

        is_chosen = c_val == CHOSEN_PARAMS["C"]
        mark = " <-- CHỌN" if is_chosen else ""
        print(
            f"  C={c_val:<6} | MacroF1={m['macro_f1']:.4f} | "
            f"ToxicF1={m['toxic_f1']:.4f} (P={m['toxic_precision']:.4f}, R={m['toxic_recall']:.4f}) | "
            f"Acc={m['accuracy']*100:.2f}% | Time={elapsed:.2f}s{mark}"
        )

        rows.append({
            "C": c_val,
            "is_chosen": is_chosen,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "toxic_f1": round(m["toxic_f1"], 4),
            "toxic_precision": round(m["toxic_precision"], 4),
            "toxic_recall": round(m["toxic_recall"], 4),
            "train_time_s": round(elapsed, 3),
        })

    return pd.DataFrame(rows)


def run_exp4_w2v_vector_size(X_train_text, y_train, X_val_text, y_val):
    """Thí nghiệm 4: Khảo sát vector_size của Word2Vec (window=5, min_count=2, epochs=10, C=1.0)."""
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 4] Khảo sát Word2Vec `vector_size` (epochs=10, C=1.0)")
    print("=" * 70)

    rows = []
    for vs in VECTOR_SIZE_GRID:
        t0 = time.perf_counter()
        w2v = Word2VecVectorizer(vector_size=vs, window=5, min_count=2, epochs=10)
        w2v.fit(X_train_text)
        X_tr_w2v = w2v.transform(X_train_text)
        X_va_w2v = w2v.transform(X_val_text)
        w2v_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        svm = get_svm_model(C=CHOSEN_PARAMS["C"], class_weight="balanced")
        svm.fit(X_tr_w2v, y_train)
        pred = svm.predict(X_va_w2v)
        svm_time = time.perf_counter() - t1
        m = compute_metrics(y_val, pred)

        is_chosen = vs == CHOSEN_PARAMS["vector_size"]
        mark = " <-- CHỌN" if is_chosen else ""
        print(
            f"  vector_size={vs:<4} | MacroF1={m['macro_f1']:.4f} | "
            f"ToxicF1={m['toxic_f1']:.4f} | Acc={m['accuracy']*100:.2f}% | "
            f"W2V_Time={w2v_time:.2f}s{mark}"
        )

        rows.append({
            "vector_size": vs,
            "is_chosen": is_chosen,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "toxic_f1": round(m["toxic_f1"], 4),
            "toxic_precision": round(m["toxic_precision"], 4),
            "toxic_recall": round(m["toxic_recall"], 4),
            "w2v_time_s": round(w2v_time, 3),
            "svm_time_s": round(svm_time, 3),
            "total_time_s": round(w2v_time + svm_time, 3),
        })

    return pd.DataFrame(rows)


def plot_results(df_mf, df_alpha, df_c, df_w2v, save_path):
    """Vẽ biểu đồ 2x2 trực quan hóa lý do chọn 4 tham số trong train.py."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Chứng minh lựa chọn tham số trong train.py (Tập Validation)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. max_features (TF-IDF)
    ax = axes[0, 0]
    x_labels = df_mf["max_features_param"].tolist()
    x_idx = list(range(len(x_labels)))
    ax.plot(x_idx, df_mf["svm_macro_f1"], marker="o", linewidth=2, label="Linear SVM (Macro F1)")
    ax.plot(x_idx, df_mf["nb_macro_f1"], marker="s", linewidth=2, linestyle="--", label="Naive Bayes (Macro F1)")
    chosen_idx = df_mf.index[df_mf["is_chosen"]].tolist()[0]
    ax.axvline(chosen_idx, color="red", linestyle=":", linewidth=1.8, label="Chọn max_features=5000")
    ax.scatter(
        [chosen_idx],
        [df_mf.loc[chosen_idx, "svm_macro_f1"]],
        color="red",
        s=110,
        zorder=5,
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_title("1. Ảnh hưởng của TF-IDF max_features", fontweight="bold")
    ax.set_xlabel("max_features")
    ax.set_ylabel("Macro F1")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. alpha (Naive Bayes)
    ax = axes[0, 1]
    ax.plot(df_alpha["alpha"], df_alpha["macro_f1"], marker="o", linewidth=2, label="Macro F1")
    ax.plot(df_alpha["alpha"], df_alpha["toxic_f1"], marker="^", linewidth=1.8, linestyle="--", label="Toxic F1")
    ax.set_xscale("log")
    ax.axvline(CHOSEN_PARAMS["alpha"], color="red", linestyle=":", linewidth=1.8, label="Chọn alpha=0.5")
    chosen_row = df_alpha[df_alpha["is_chosen"]].iloc[0]
    ax.scatter([chosen_row["alpha"]], [chosen_row["macro_f1"]], color="red", s=110, zorder=5)
    ax.set_title("2. Ảnh hưởng của Naive Bayes alpha", fontweight="bold")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("F1 Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. C (Linear SVM TF-IDF)
    ax = axes[1, 0]
    ax.plot(df_c["C"], df_c["macro_f1"], marker="o", linewidth=2, label="Macro F1")
    ax.plot(df_c["C"], df_c["toxic_f1"], marker="^", linewidth=1.8, linestyle="--", label="Toxic F1")
    ax.set_xscale("log")
    ax.axvline(CHOSEN_PARAMS["C"], color="red", linestyle=":", linewidth=1.8, label="Chọn C=1.0")
    chosen_row = df_c[df_c["is_chosen"]].iloc[0]
    ax.scatter([chosen_row["C"]], [chosen_row["macro_f1"]], color="red", s=110, zorder=5)
    ax.set_title("3. Ảnh hưởng của Linear SVM C (TF-IDF)", fontweight="bold")
    ax.set_xlabel("C (thang log)")
    ax.set_ylabel("F1 Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. vector_size (Word2Vec + SVM)
    ax = axes[1, 1]
    ax.plot(df_w2v["vector_size"], df_w2v["macro_f1"], marker="o", linewidth=2, label="Macro F1")
    ax.plot(df_w2v["vector_size"], df_w2v["toxic_f1"], marker="^", linewidth=1.8, linestyle="--", label="Toxic F1")
    ax.axvline(CHOSEN_PARAMS["vector_size"], color="red", linestyle=":", linewidth=1.8, label="Chọn vector_size=150")
    chosen_row = df_w2v[df_w2v["is_chosen"]].iloc[0]
    ax.scatter([chosen_row["vector_size"]], [chosen_row["macro_f1"]], color="red", s=110, zorder=5)
    ax.set_title("4. Ảnh hưởng của Word2Vec vector_size", fontweight="bold")
    ax.set_xlabel("vector_size (số chiều)")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(df_w2v["vector_size"].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def _format_table(df):
    """Định dạng DataFrame sang Markdown (fallback về plain text nếu thiếu tabulate)."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_summary_report(df_mf, df_alpha, df_c, df_w2v, save_path):
    """Tạo báo cáo Markdown tóm tắt kết quả thực nghiệm để đưa vào báo cáo."""
    best_mf_svm = df_mf.loc[df_mf["svm_macro_f1"].idxmax()]
    chosen_mf = df_mf[df_mf["is_chosen"]].iloc[0]

    best_alpha = df_alpha.loc[df_alpha["macro_f1"].idxmax()]
    chosen_alpha = df_alpha[df_alpha["is_chosen"]].iloc[0]

    best_c = df_c.loc[df_c["macro_f1"].idxmax()]
    chosen_c = df_c[df_c["is_chosen"]].iloc[0]

    best_w2v = df_w2v.loc[df_w2v["macro_f1"].idxmax()]
    chosen_w2v = df_w2v[df_w2v["is_chosen"]].iloc[0]

    lines = [
        "# Báo cáo Thực nghiệm Chứng minh Tham số (`train.py`)",
        "",
        "## 1. TF-IDF `max_features = 5000`",
        f"- Giá trị chọn (`5000`): SVM Macro F1 = **{chosen_mf['svm_macro_f1']:.4f}**, NB Macro F1 = **{chosen_mf['nb_macro_f1']:.4f}**, thời gian TF-IDF = {chosen_mf['tfidf_time_s']}s.",
        f"- Đỉnh SVM Macro F1 quan sát được: `max_features={best_mf_svm['max_features_param']}` đạt **{best_mf_svm['svm_macro_f1']:.4f}** (chênh lệch `{best_mf_svm['svm_macro_f1'] - chosen_mf['svm_macro_f1']:+.4f}`).",
        "",
        _format_table(df_mf[["max_features_param", "actual_features", "nb_macro_f1", "svm_macro_f1", "svm_toxic_f1", "tfidf_time_s"]]),
        "",
        "## 2. Multinomial Naive Bayes `alpha = 0.5`",
        f"- Giá trị chọn (`alpha=0.5`): Macro F1 = **{chosen_alpha['macro_f1']:.4f}**, Toxic F1 = **{chosen_alpha['toxic_f1']:.4f}**, Precision = {chosen_alpha['toxic_precision']:.4f}, Recall = {chosen_alpha['toxic_recall']:.4f}.",
        f"- Đỉnh Macro F1 quan sát được: `alpha={best_alpha['alpha']}` đạt **{best_alpha['macro_f1']:.4f}** (chênh lệch `{best_alpha['macro_f1'] - chosen_alpha['macro_f1']:+.4f}`).",
        "",
        _format_table(df_alpha[["alpha", "accuracy", "macro_f1", "toxic_f1", "toxic_precision", "toxic_recall"]]),
        "",
        "## 3. Linear SVM `C = 1.0` (TF-IDF, `class_weight='balanced'`)",
        f"- Giá trị chọn (`C=1.0`): Macro F1 = **{chosen_c['macro_f1']:.4f}**, Toxic F1 = **{chosen_c['toxic_f1']:.4f}**, Precision = {chosen_c['toxic_precision']:.4f}, Recall = {chosen_c['toxic_recall']:.4f}.",
        f"- Đỉnh Macro F1 quan sát được: `C={best_c['C']}` đạt **{best_c['macro_f1']:.4f}** (chênh lệch `{best_c['macro_f1'] - chosen_c['macro_f1']:+.4f}`).",
        "",
        _format_table(df_c[["C", "accuracy", "macro_f1", "toxic_f1", "toxic_precision", "toxic_recall", "train_time_s"]]),
        "",
        "## 4. Word2Vec `vector_size = 150` (Linear SVM `C=1.0`)",
        f"- Giá trị chọn (`vector_size=150`): Macro F1 = **{chosen_w2v['macro_f1']:.4f}**, Toxic F1 = **{chosen_w2v['toxic_f1']:.4f}**, tổng thời gian = {chosen_w2v['total_time_s']}s.",
        f"- Đỉnh Macro F1 quan sát được: `vector_size={int(best_w2v['vector_size'])}` đạt **{best_w2v['macro_f1']:.4f}** (chênh lệch `{best_w2v['macro_f1'] - chosen_w2v['macro_f1']:+.4f}`).",
        "",
        _format_table(df_w2v[["vector_size", "accuracy", "macro_f1", "toxic_f1", "w2v_time_s", "total_time_s"]]),
        "",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Thực nghiệm chứng minh lựa chọn tham số trong train.py")
    parser.add_argument("--train-path", default="dataset/processed_data/train.csv")
    parser.add_argument("--val-path", default="dataset/processed_data/valid.csv")
    parser.add_argument("--output-dir", default="results/param_proof")
    args = parser.parse_args()

    train_path, val_path = resolve_data_paths(args.train_path, args.val_path)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  CHỨNG MINH THAM SỐ TRAIN.PY: alpha=0.5, C=1.0, max_features=5000, vector_size=150")
    print("=" * 70)

    df_train = pd.read_csv(train_path).dropna(subset=["text"])
    df_val = pd.read_csv(val_path).dropna(subset=["text"])
    X_train_text, y_train = df_train["text"], df_train["label"]
    X_val_text, y_val = df_val["text"], df_val["label"]
    print(f"- Train: {len(df_train)} mẫu | Validation: {len(df_val)} mẫu")

    # Thí nghiệm 1: max_features
    df_mf = run_exp1_max_features(X_train_text, y_train, X_val_text, y_val)

    # Chuẩn bị ma trận TF-IDF cố định max_features=5000 cho Thí nghiệm 2 & 3
    tfidf_5000 = get_tfidf_vectorizer(max_features=CHOSEN_PARAMS["max_features"])
    X_train_tfidf = tfidf_5000.fit_transform(X_train_text)
    X_val_tfidf = tfidf_5000.transform(X_val_text)

    # Thí nghiệm 2: alpha
    df_alpha = run_exp2_nb_alpha(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # Thí nghiệm 3: C
    df_c = run_exp3_svm_c(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # Thí nghiệm 4: vector_size
    df_w2v = run_exp4_w2v_vector_size(X_train_text, y_train, X_val_text, y_val)

    # Lưu kết quả CSV, biểu đồ và báo cáo Markdown
    df_mf.to_csv(os.path.join(args.output_dir, "exp1_max_features.csv"), index=False)
    df_alpha.to_csv(os.path.join(args.output_dir, "exp2_nb_alpha.csv"), index=False)
    df_c.to_csv(os.path.join(args.output_dir, "exp3_svm_C.csv"), index=False)
    df_w2v.to_csv(os.path.join(args.output_dir, "exp4_w2v_vector_size.csv"), index=False)

    chart_path = os.path.join(args.output_dir, "param_justification_charts.png")
    plot_results(df_mf, df_alpha, df_c, df_w2v, chart_path)

    report_path = os.path.join(args.output_dir, "summary_report.md")
    write_summary_report(df_mf, df_alpha, df_c, df_w2v, report_path)

    print("\n" + "=" * 70)
    print(f"[HOÀN TẤT] Đã lưu toàn bộ kết quả tại: {args.output_dir}/")
    print(f"  - Biểu đồ tổng hợp : {chart_path}")
    print(f"  - Báo cáo Markdown : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
