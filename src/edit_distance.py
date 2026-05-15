from __future__ import annotations


def calculate_levenshtein_distance(seq1: list[str], seq2: list[str]) -> int:
    """
    Tính khoảng cách Levenshtein (số bước biến đổi tối thiểu) giữa 2 mảng từ vựng.
    Sử dụng Quy hoạch động (Dynamic Programming) tối ưu bộ nhớ với 2 dòng thay vì
    ma trận đầy đủ, giảm từ O(m*n) xuống O(min(m,n)) bộ nhớ.

    :param seq1: Mảng từ vựng thứ nhất
    :param seq2: Mảng từ vựng thứ hai
    :return: Số nguyên thể hiện số thao tác (Thêm, Xóa, Sửa) ít nhất.
    """
    # 1. Chặn lỗi (Edge Case) — kiểm tra đúng kiểu dữ liệu
    # isinstance() kiểm tra xem biến có đúng là list không.
    # Nếu ai đó truyền vào None, số, hay chuỗi thì báo lỗi rõ ràng thay vì crash âm thầm.
    if not isinstance(seq1, list) or not isinstance(seq2, list):
        raise TypeError(
            f"Cả hai đầu vào phải là list, nhận được: {type(seq1)}, {type(seq2)}"
        )

    # 2. Đảm bảo seq1 luôn là chuỗi DÀI HƠN, seq2 là chuỗi NGẮN HƠN
    # Mục đích: dòng "prev" và "curr" có độ dài bằng len(seq2) + 1
    # --> Luôn giữ seq2 là chuỗi ngắn hơn để tiết kiệm bộ nhớ tối đa.
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)

    # 3. Khởi tạo dòng đầu tiên (Base Case cho i=0)
    # Thay vì tạo cả ma trận (m+1)x(n+1), chỉ cần 2 dòng tại một thời điểm.
    # "prev" là dòng phía trên dòng đang xét — tương đương dp[0][j] = j
    # (Nếu seq1 rỗng, cần chèn j từ để tạo ra seq2[:j])
    prev = list(range(n + 1))

    # 4. Vòng lặp điền từng dòng (Core Logic)
    for i in range(1, m + 1):
        # Khởi tạo dòng hiện tại — ô đầu tiên luôn = i
        # (Nếu seq2 rỗng, cần xóa i từ để biến seq1[:i] thành chuỗi rỗng)
        curr = [i] + [0] * n

        for j in range(1, n + 1):
            # Nếu 2 từ tại vị trí đang xét giống nhau -> Không tốn thao tác (cost = 0)
            # Nếu khác nhau -> Tốn 1 thao tác thay thế (cost = 1)
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1

            # Quyết định chọn con đường tốn ít thao tác nhất trong 3 lựa chọn:
            # - prev[j] + 1      : Xóa từ seq1[i-1]        (đến từ ô phía trên)
            # - curr[j-1] + 1    : Chèn từ seq2[j-1]        (đến từ ô bên trái)
            # - prev[j-1] + cost : Thay thế hoặc giữ nguyên  (đến từ ô chéo trên-trái)
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost
            )

        # Trượt dòng xuống: dòng hiện tại trở thành dòng "phía trên" cho vòng lặp tiếp theo
        prev = curr

    # Kết quả tối ưu nằm ở ô cuối cùng của dòng cuối cùng
    return prev[n]


def calculate_wer(machine_tokens: list[str], reference_tokens: list[str]) -> float:
    """
    Tính Tỷ lệ lỗi từ (Word Error Rate - WER).
    WER = (Số thao tác Levenshtein) / (Độ dài câu chuẩn).
    Giá trị từ 0.0 (hoàn hảo) đến > 1.0 (máy dịch thêm quá nhiều từ rác).
    """
    # 1. Chặn lỗi — kiểm tra kiểu dữ liệu đầu vào
    if not isinstance(machine_tokens, list) or not isinstance(reference_tokens, list):
        raise TypeError("Cả hai đầu vào phải là list.")

    # 2. Xử lý Edge Case: tránh lỗi chia cho 0 khi câu chuẩn rỗng
    # - Nếu cả hai đều rỗng -> Hoàn hảo, WER = 0.0
    # - Nếu chỉ câu chuẩn rỗng nhưng máy vẫn sinh ra từ -> Lỗi hoàn toàn, WER = 1.0
    if len(reference_tokens) == 0:
        return 0.0 if len(machine_tokens) == 0 else 1.0

    # 3. Tính khoảng cách chỉnh sửa rồi chia cho độ dài câu chuẩn
    edit_distance = calculate_levenshtein_distance(machine_tokens, reference_tokens)
    wer = edit_distance / len(reference_tokens)
    return wer


if __name__ == "__main__":
    # Kịch bản Test — bao phủ các trường hợp quan trọng
    test_cases = [
        (
            ['hệ', 'thống', 'hoạt', 'động', 'rất', 'tốt'],
            ['hệ', 'thống', 'hoạt', 'động', 'hoàn', 'hảo'],
            "2 thay thế cuối — kết quả mong đợi: distance=2, WER=33.3%"
        ),
        (
            ['xin', 'chào'],
            ['xin', 'chào'],
            "Giống hệt nhau — kết quả mong đợi: distance=0, WER=0.0%"
        ),
        (
            [],
            ['a', 'b', 'c'],
            "Máy dịch rỗng — kết quả mong đợi: distance=3, WER=100.0%"
        ),
        (
            ['a', 'b', 'c'],
            [],
            "Câu chuẩn rỗng — kết quả mong đợi: WER=1.0 (edge case)"
        ),
        (
            ['tôi', 'đi'],
            ['bạn', 'chạy', 'nhanh'],
            "Hoàn toàn khác — kết quả mong đợi: distance=3, WER=100.0%"
        ),
    ]

    for machine, ref, mo_ta in test_cases:
        dist = calculate_levenshtein_distance(machine, ref)
        wer  = calculate_wer(machine, ref)
        print(f"[{mo_ta}]")
        print(f"  → Edit distance: {dist}  |  WER: {wer * 100:.1f}%\n")