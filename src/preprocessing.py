from __future__ import annotations
import re

def clean_and_tokenize(text: str) -> list[str]:
    """
    Làm sạch và chia tách từ vựng từ một chuỗi văn bản thô.
    Trả về list rỗng nếu đầu vào không phải chuỗi.
    """
    # 1. Chặn lỗi — kiểm tra đúng kiểu dữ liệu
    if not isinstance(text, str):
        return []

    # 2. Đồng nhất chữ thường
    text = text.lower()

    # 3. Xóa dấu câu, giữ lại chữ cái Unicode (gồm tiếng Việt) và khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)

    # 4. Tách thành list token
    return text.split()


if __name__ == "__main__":
    test_cases = [
        ("Hello, World! Hệ thống AI dịch thuật năm 2026...", "bình thường"),
        ("",                                                  "chuỗi rỗng"),
        (None,                                               "None"),
        (12345,                                              "số nguyên"),
        ("!!! ???",                                          "chỉ dấu câu"),
    ]
    for cau, mo_ta in test_cases:
        ket_qua = clean_and_tokenize(cau)
        print(f"[{mo_ta}] → {ket_qua}")