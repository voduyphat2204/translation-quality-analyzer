from sentence_transformers import SentenceTransformer, util

print("⏳ Đang tải mô hình AI Ngôn ngữ (Lần đầu sẽ mất khoảng 1-2 phút)...")
# Tải mô hình đa ngôn ngữ cực xịn: Hỗ trợ tiếng Việt, Anh, Pháp, Tây Ban Nha...
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Tải mô hình hoàn tất!\n")

def calculate_semantic_similarity(sentence1: str, sentence2: str) -> float:
    """
    Tính độ tương đồng ngữ nghĩa (Semantic Similarity) bằng AI (Word/Sentence Embeddings).
    Hỗ trợ nhận diện từ đồng nghĩa và đa ngôn ngữ.
    """
    # 1. Rào chắn lỗi
    if not sentence1.strip() or not sentence2.strip():
        return 0.0

    # 2. Đưa câu văn vào Não bộ AI để biến thành Vector đa chiều (Tensor)
    # LƯU Ý: Không cần băm (tokenize) nữa, AI tự hiểu nguyên cả câu!
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # 3. Tính Cosine Similarity bằng hàm Toán học Tích lượng giác của thư viện
    cosine_scores = util.cos_sim(embedding1, embedding2)

    # 4. Trích xuất con số từ Tensor trả về
    return float(cosine_scores[0][0])

if __name__ == "__main__":
    # Test 1: Từ đồng nghĩa (Khắc phục điểm yếu cũ)
    machine_1 = "chiếc ô tô màu đỏ"
    human_1 = "chiếc xe hơi màu đỏ"
    score_1 = calculate_semantic_similarity(machine_1, human_1)
    
    print(f"Câu 1: '{machine_1}' | Câu 2: '{human_1}'")
    print(f"-> Độ tương đồng Ngữ nghĩa: {score_1 * 100:.2f}%\n")

    # Test 2: Khác biệt hoàn toàn
    machine_2 = "chiếc ô tô màu đỏ"
    human_2 = "con chó đang sủa gâu gâu"
    score_2 = calculate_semantic_similarity(machine_2, human_2)
    
    print(f"Câu 1: '{machine_2}' | Câu 2: '{human_2}'")
    print(f"-> Độ tương đồng Ngữ nghĩa: {score_2 * 100:.2f}%\n")