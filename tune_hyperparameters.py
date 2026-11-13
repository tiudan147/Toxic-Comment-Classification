"""
================================================================================
TUNING THAM SỐ MÔ HÌNH NAIVE BAYES & LINEAR SVM (HYPERPARAMETER TUNING)
File: tune_hyperparameters.py
================================================================================
MỤC ĐÍCH:
    Thử nghiệm hàng loạt các giá trị tham số của 2 mô hình trên cùng bộ đặc
    trưng TF-IDF (giống hệt train.py) để trả lời 2 câu hỏi:
        1. Việc thay đổi tham số (alpha của NB, C & class_weight của SVM)
           có làm thay đổi kết quả nhiều hay không?
        2. Bộ tham số nào là TỐT NHẤT (theo Macro F1 trên tập Validation)?

    Sau khi chọn được tham số tốt nhất, script đánh giá lại bộ tham số đó trên
    tập Test (so sánh với bộ tham số mặc định hiện tại) và lưu toàn bộ kết quả:

        results/tuning_nb_results.csv        : Bảng thử nghiệm Naive Bayes
        results/tuning_svm_results.csv       : Bảng thử nghiệm SVM (TF-IDF)
        results/tuning_svm_w2v_results.csv   : Bảng thử nghiệm SVM (Word2Vec, tùy chọn)
        results/best_params.json             : Bộ tham số tốt nhất
        results/tuning_summary.txt           : Tóm tắt baseline vs best

CÁCH CHẠY:
    python tune_hyperparameters.py                    # Chỉ tuning NB + SVM (TF-IDF)
    python tune_hyperparameters.py --tune-w2v         # Kèm cả SVM (Word2Vec)
    python tune_hyperparameters.py --data-dir dataset/processed_data

THƯỚC ĐO CHÍNH:
    Macro F1 trên tập Validation (chỉ số phản ánh trung thực nhất trên dữ liệu
    mất cân bằng nhãn - xem chi tiết trong src/evaluation/metrics.py).
================================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

# Đảm bảo terminal in đúng tiếng Việt UTF-8 không bị lỗi ký tự
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm thư mục gốc dự án vào sys.path để nạp các module trong src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from src.models import get_naive_bayes_model, get_svm_model
from src.evaluation import evaluate_model

# ------------------------------------------------------------------------------
# LƯỚI THAM SỐ CẦN THỬ NGHIỆM (Có thể tự do mở rộng danh sách này)
# ------------------------------------------------------------------------------
ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
C_GRID = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
CLASS_WEIGHT_GRID = [None, "balanced"]

# Tham số mặc định ĐANG DÙNG trong train.py (làm mốc so sánh - baseline)
NB_BASELINE_ALPHA = 0.5
SVM_BASELINE_C = 1.0
SVM_BASELINE_CW = "balanced"


def compute_metrics(y_true, y_pred):
    """
    Tính nhanh 4 chỉ số chính mà KHÔNG in báo cáo dài (dùng cho lưới thử nghiệm).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "toxic_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "toxic_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def rank_and_print(df_grid, name):
    """
    Sắp xếp bảng kết quả theo Macro F1 giảm dần và in ra màn hình.
    """
    df_sorted = df_grid.sort_values(
        by=["macro_f1", "toxic_f1"], ascending=False
    ).reset_index(drop=True)
    df_sorted["macro_f1"] = df_sorted["macro_f1"].round(4)
    df_sorted["toxic_f1"] = df_sorted["toxic_f1"].round(4)
    df_sorted["toxic_recall"] = df_sorted["toxic_recall"].round(4)
    df_sorted["accuracy"] = (df_sorted["accuracy"] * 100).round(2)

    print(f"\n{'=' * 65}")
    print(f"   BẢNG XẾP HẠNG THAM SỐ {name} (theo Macro F1 trên Validation)")
    print(f"{'=' * 65}")
    print(df_sorted.to_string(index=True))
    return df_sorted


def tune_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Thử toàn bộ giá trị alpha của Multinomial Naive Bayes trên tập Validation.
    """
    rows = []
    print(f"\n[BƯỚC TUNING 1] Naive Bayes: thử {len(ALPHA_GRID)} giá trị alpha...")
    for alpha in ALPHA_GRID:
        model = get_naive_bayes_model(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        rows.append({
            "alpha": alpha,
            "is_baseline": (alpha == NB_BASELINE_ALPHA),
            **metrics,
        })
        print(f"   alpha={alpha:<6} -> Macro F1={metrics['macro_f1']:.4f} | "
              f"Toxic F1={metrics['toxic_f1']:.4f} | Acc={metrics['accuracy']*100:.2f}%")

    df_grid = pd.DataFrame(rows)
    rank_and_print(df_grid, "NAIVE BAYES (alpha)")
    return df_grid


def tune_svm(X_train, y_train, X_val, y_val, feature_name="TF-IDF"):
    """
    Thử toàn bộ tổ hợp (C, class_weight) của Linear SVM trên tập Validation.
    """
    rows = []
    total = len(C_GRID) * len(CLASS_WEIGHT_GRID)
    print(f"\n[BƯỚC TUNING] Linear SVM ({feature_name}): "
          f"thử {total} tổ hợp (C x class_weight)...")
    count = 0
    for cw in CLASS_WEIGHT_GRID:
        for c_value in C_GRID:
            count += 1
            model = get_svm_model(C=c_value, class_weight=cw)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics = compute_metrics(y_val, y_pred)
            rows.append({
                "C": c_value,
                "class_weight": cw if cw else "none",
                "is_baseline": (c_value == SVM_BASELINE_C and cw == SVM_BASELINE_CW),
                **metrics,
            })
            print(f"   [{count:>2}/{total}] C={c_value:<6} cw={cw!s:<8} -> "
                  f"Macro F1={metrics['macro_f1']:.4f} | "
                  f"Toxic F1={metrics['toxic_f1']:.4f} | Acc={metrics['accuracy']*100:.2f}%")

    df_grid = pd.DataFrame(rows)
    rank_and_print(df_grid, f"LINEAR SVM ({feature_name})")
    return df_grid


def pick_best_row(df_grid, param_cols):
    """
    Chọn dòng tốt nhất theo Macro F1 (dự phòng Toxic F1 nếu bằng nhau).
    """
    best = df_grid.sort_values(
        by=["macro_f1", "toxic_f1"], ascending=False
    ).iloc[0]
    return {col: best[col] for col in param_cols}, best


def compare_test_baseline_vs_best(model_name, make_baseline, make_best, X_test, y_test, out_lines):
    """
    Đánh giá trên tập Test: bộ tham số MẶC ĐỊNH (baseline) vs bộ TỐT NHẤT.
    """
    print(f"\n{'=' * 65}")
    print(f"   KIỂM ĐỊNH CUỐI TRÊN TẬP TEST: {model_name}")
    print(f"{'=' * 65}")

    baseline_metrics = compute_metrics(y_test, make_baseline().fit_predict_test(X_test))
    best_metrics = compute_metrics(y_test, make_best().fit_predict_test(X_test))

    rows = []
    for label, m in [("Baseline (đang dùng)", baseline_metrics), ("Best (tuning)", best_metrics)]:
        rows.append({
            "Mô hình": f"{model_name} - {label}",
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro F1": round(m["macro_f1"], 4),
            "Toxic F1": round(m["toxic_f1"], 4),
            "Toxic Recall": round(m["toxic_recall"], 4),
        })
    df_cmp = pd.DataFrame(rows)
    print(df_cmp.to_string(index=False))
    delta_macro = (best_metrics["macro_f1"] - baseline_metrics["macro_f1"]) * 100
    print(f"--> Chênh lệch Macro F1 trên Test: {delta_macro:+.2f}% "
          f"({'CÓ cải thiện' if delta_macro > 0.05 else 'KHÔNG cải thiện đáng kể' if abs(delta_macro) <= 0.05 else 'BỊ suy giảm!'})")
    out_lines.extend(df_cmp.to_string(index=False).splitlines())
    out_lines.append(f"Chênh lệch Macro F1 trên Test ({model_name}): {delta_macro:+.2f}%")
    return best_metrics


class FitPredict:
    """
    Bọc nhỏ giúp fit lại mô hình với tham số cho trước và dự đoán trên Test.
    """
    def __init__(self, factory):
        self.factory = factory

    def fit_predict_test(self, X_test):
        model = self.factory()
        model.fit(self.X_train, self.y_train)
        return model.predict(X_test)

    def bind(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self


def main():
    parser = argparse.ArgumentParser(description="Tuning tham số Naive Bayes & Linear SVM")
    parser.add_argument("--data-dir", default="dataset/processed_data",
                        help="Thư mục chứa train.csv, val.csv, test.csv đã làm sạch")
    parser.add_argument("--max-features", type=int, default=5000,
                        help="Số chiều TF-IDF (giống train.py: 5000)")
    parser.add_argument("--ngram", default="1,2",
                        help="ngram_range của TF-IDF, ví dụ '1,2' (giống train.py)")
    parser.add_argument("--skip-w2v", action="store_true",
                        help="Bỏ qua phần tuning SVM trên Word2Vec")
    args = parser.parse_args()

    print("=" * 65)
    print("   TUNING THAM SỐ MÔ HÌNH: NAIVE BAYES & LINEAR SVM")
    print("=" * 65)

    # --------------------------------------------------------------------------
    # BƯỚC 1: NẠP DỮ LIỆU ĐÃ XỬ LÝ
    # --------------------------------------------------------------------------
    print(f"\n[BƯỚC 1] Nạp dữ liệu từ {args.data_dir}/ ...")
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"[LỖI] Không tìm thấy file: {path}")
            print("Hãy chạy pipeline tiền xử lý trước (merge -> split -> process_data) "
                  "hoặc chỉ định đúng --data-dir.")
            sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    print(f"- Train: {len(df_train)} dòng | Val: {len(df_val)} dòng | Test: {len(df_test)} dòng")

    X_train_text, y_train = df_train["text"].fillna(""), df_train["label"]
    X_val_text, y_val = df_val["text"].fillna(""), df_val["label"]
    X_test_text, y_test = df_test["text"].fillna(""), df_test["label"]

    # --------------------------------------------------------------------------
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TF-IDF (giống hệt cấu hình trong train.py)
    # --------------------------------------------------------------------------
    ngram_range = tuple(int(x) for x in args.ngram.split(","))
    print(f"\n[BƯỚC 2] TF-IDF (max_features={args.max_features}, "
          f"ngram_range={ngram_range}, sublinear_tf=True)...")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    print(f"- Ma trận Train: {X_train_tfidf.shape} | Val: {X_val_tfidf.shape} | Test: {X_test_tfidf.shape}")

    os.makedirs("results", exist_ok=True)
    summary_lines = []
    best_params = {"tfidf": {"max_features": args.max_features,
                             "ngram_range": list(ngram_range)}}

    # --------------------------------------------------------------------------
    # BƯỚC 3: TUNING NAIVE BAYES (TF-IDF)
    # --------------------------------------------------------------------------
    start = time.time()
    df_nb = tune_naive_bayes(X_train_tfidf, y_train, X_val_tfidf, y_val)
    df_nb.to_csv("results/tuning_nb_results.csv", index=False, encoding="utf-8-sig")
    nb_best, nb_best_row = pick_best_row(df_nb, ["alpha"])
    print(f"\n>>> Tham số NB tốt nhất: alpha={nb_best['alpha']} "
          f"(Val Macro F1={nb_best_row['macro_f1']:.4f})")
    print(f"    (Baseline hiện tại: alpha={NB_BASELINE_ALPHA}, "
          f"Macro F1={df_nb[df_nb['is_baseline']]['macro_f1'].iloc[0]:.4f})")
    summary_lines.append(f"Naive Bayes best: alpha={nb_best['alpha']} "
                         f"(Val Macro F1={nb_best_row['macro_f1']:.4f})")
    best_params["naive_bayes"] = {
        "alpha": float(nb_best["alpha"]),
        "val_macro_f1": round(float(nb_best_row["macro_f1"]), 4),
    }
    print(f"    Thời gian: {time.time() - start:.1f}s")

    # --------------------------------------------------------------------------
    # BƯỚC 4: TUNING LINEAR SVM (TF-IDF)
    # --------------------------------------------------------------------------
    start = time.time()
    df_svm = tune_svm(X_train_tfidf, y_train, X_val_tfidf, y_val, feature_name="TF-IDF")
    df_svm.to_csv("results/tuning_svm_results.csv", index=False, encoding="utf-8-sig")
    svm_best, svm_best_row = pick_best_row(df_svm, ["C", "class_weight"])
    cw_display = svm_best["class_weight"]
    print(f"\n>>> Tham số SVM (TF-IDF) tốt nhất: C={svm_best['C']}, "
          f"class_weight={cw_display} (Val Macro F1={svm_best_row['macro_f1']:.4f})")
    print(f"    (Baseline hiện tại: C={SVM_BASELINE_C}, class_weight={SVM_BASELINE_CW}, "
          f"Macro F1={df_svm[df_svm['is_baseline']]['macro_f1'].iloc[0]:.4f})")
    summary_lines.append(f"SVM TF-IDF best: C={svm_best['C']}, class_weight={cw_display} "
                         f"(Val Macro F1={svm_best_row['macro_f1']:.4f})")
    best_params["svm_tfidf"] = {
        "C": float(svm_best["C"]),
        "class_weight": None if cw_display == "none" else cw_display,
        "val_macro_f1": round(float(svm_best_row["macro_f1"]), 4),
    }
    print(f"    Thời gian: {time.time() - start:.1f}s")

    # --------------------------------------------------------------------------
    # BƯỚC 5 (TÙY CHỌN): TUNING LINEAR SVM (WORD2VEC)
    # --------------------------------------------------------------------------
    if not args.skip_w2v:
        try:
            from src.features import Word2VecVectorizer

            print("\n[BƯỚC 5] Huấn luyện Word2Vec (150 chiều) trên tập Train...")
            start = time.time()
            w2v = Word2VecVectorizer(vector_size=150, window=5, min_count=2, sg=1, epochs=10)
            w2v.fit(X_train_text)
            X_train_w2v = w2v.transform(X_train_text)
            X_val_w2v = w2v.transform(X_val_text)
            X_test_w2v = w2v.transform(X_test_text)
            print(f"- Hoàn tất Word2Vec sau {time.time() - start:.1f}s "
                  f"(Train: {X_train_w2v.shape})")

            df_svm_w2v = tune_svm(X_train_w2v, y_train, X_val_w2v, y_val,
                                  feature_name="Word2Vec")
            df_svm_w2v.to_csv("results/tuning_svm_w2v_results.csv", index=False,
                              encoding="utf-8-sig")
            w2v_best, w2v_best_row = pick_best_row(df_svm_w2v, ["C", "class_weight"])
            cw_w2v = w2v_best["class_weight"]
            print(f"\n>>> Tham số SVM (Word2Vec) tốt nhất: C={w2v_best['C']}, "
                  f"class_weight={cw_w2v} (Val Macro F1={w2v_best_row['macro_f1']:.4f})")
            summary_lines.append(f"SVM Word2Vec best: C={w2v_best['C']}, "
                                 f"class_weight={cw_w2v} "
                                 f"(Val Macro F1={w2v_best_row['macro_f1']:.4f})")
            best_params["svm_w2v"] = {
                "C": float(w2v_best["C"]),
                "class_weight": None if cw_w2v == "none" else cw_w2v,
                "val_macro_f1": round(float(w2v_best_row["macro_f1"]), 4),
            }
        except ImportError:
            print("\n[BƯỚC 5 - BỎ QUA] Chưa cài gensim nên bỏ qua tuning SVM (Word2Vec). "
                  "Chạy lại trên Colab hoặc cài: pip install gensim")

    # --------------------------------------------------------------------------
    # BƯỚC 6: KIỂM ĐỊNH TRÊN TẬP TEST - BASELINE vs BEST
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 6] Kiểm định bộ tham số tốt nhất trên tập Test...")

    nb_baseline = FitPredict(lambda: get_naive_bayes_model(alpha=NB_BASELINE_ALPHA)
                             ).bind(X_train_tfidf, y_train)
    nb_best_model = FitPredict(lambda: get_naive_bayes_model(alpha=nb_best["alpha"])
                               ).bind(X_train_tfidf, y_train)
    compare_test_baseline_vs_best(
        "Naive Bayes (TF-IDF)",
        lambda: nb_baseline, lambda: nb_best_model, X_test_tfidf, y_test, summary_lines
    )

    svm_baseline = FitPredict(lambda: get_svm_model(C=SVM_BASELINE_C,
                                                    class_weight=SVM_BASELINE_CW)
                              ).bind(X_train_tfidf, y_train)
    svm_best_model = FitPredict(
        lambda: get_svm_model(C=svm_best["C"],
                              class_weight=None if svm_best["class_weight"] == "none"
                              else svm_best["class_weight"])
    ).bind(X_train_tfidf, y_train)
    compare_test_baseline_vs_best(
        "Linear SVM (TF-IDF)",
        lambda: svm_baseline, lambda: svm_best_model, X_test_tfidf, y_test, summary_lines
    )

    # --------------------------------------------------------------------------
    # BƯỚC 7: LƯU KẾT QUẢ TỔNG HỢP
    # --------------------------------------------------------------------------
    with open("results/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    with open("results/tuning_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 65)
    print("   [HOÀN TẤT] ĐÃ CHỌN ĐƯỢC THAM SỐ TỐT NHẤT (theo Val Macro F1):")
    print("=" * 65)
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    print("\n- Bảng kết quả đầy đủ : results/tuning_nb_results.csv, "
          "results/tuning_svm_results.csv")
    print("- Bộ tham số tốt nhất : results/best_params.json")
    print("- So sánh trên Test   : results/tuning_summary.txt")
    print("\n=> Dùng bộ tham số trên để cập nhật trong train.py "
          "(notebook Colab sẽ tự cập nhật giúp bạn).")


if __name__ == "__main__":
    main()
