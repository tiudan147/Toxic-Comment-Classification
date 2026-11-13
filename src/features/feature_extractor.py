from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def get_tfidf_vectorizer(max_features=5000):
    return TfidfVectorizer(max_features=max_features)

class Word2VecVectorizer:
    def __init__(self, vector_size=150, window=5, min_count=2, epochs=100):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, texts, y=None):
        sentences = [doc.split() for doc in texts if isinstance(doc, str)]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=42
        )
        return self
    
    def transform(self, texts):
        wv = self.model.wv  # Kho từ vựng và vector đã học
        vectors = [] # mỗi vector ứng với một câu
        for doc in texts:
            words = doc.split()
            valid_word_vectors = [wv[w] for w in words if w in wv]

            if valid_word_vectors:
                # Tính trung bình cộng vector của các từ trong câu
                doc_vector = np.mean(valid_word_vectors, axis=0)
                vectors.append(doc_vector)
            
            else:
                vectors.append(np.zeros(self.vector_size, dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

    def fit_transform(self, texts, y=None): # dành cho vừa huấn luyện vừa transform
        self.fit(texts, y)
        return self.transform(texts)



if __name__ == "__main__":
    sample_texts = [
        "mày ngu_ngốc quá đấy",
        "hôm_nay thời_tiết rất đẹp và dễ_chịu",
        "đồ ngu lừa_đảo cút đi"
    ]

    print("==================================================")
    print("DEMO TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (TF-IDF & WORD2VEC)")
    print("==================================================")

    # 1. Thử nghiệm TF-IDF
    print("\n--- 1. KẾT QUẢ VỚI TF-IDF ---")
    tfidf = get_tfidf_vectorizer()
    X_tfidf = tfidf.fit_transform(sample_texts)
    print(f"- Số mẫu văn bản       : {X_tfidf.shape[0]}")
    print(f"- Số lượng đặc trưng   : {X_tfidf.shape[1]}")
    print(f"- Danh sách từ vựng    : {tfidf.get_feature_names_out()[:10]}...")
    print(f"- Ma trận TF-IDF (dạng mảng numpy):\n{X_tfidf.toarray()[:2]}")

    # 2. Thử nghiệm Word2Vec
    print("\n--- 2. KẾT QUẢ VỚI WORD2VEC (Mean Vector) ---")
    w2v = Word2VecVectorizer(vector_size=10, min_count=1, epochs=5)
    X_w2v = w2v.fit_transform(sample_texts)
    print(f"- Kích thước ma trận Word2Vec : {X_w2v.shape} (3 câu, mỗi câu là 1 vector 10 chiều)")
    print(f"- Vector của câu đầu tiên:\n{X_w2v[0]}")