import os
# os là thư viện chuẩn của Python để làm việc với hệ điều hành:
# đọc đường dẫn file, kiểm tra file tồn tại, v.v.
# Dùng os thay vì gõ tay đường dẫn vì Windows dùng '\' còn Mac/Linux dùng '/'
# --> os.path.join() tự xử lý sự khác biệt này cho mình.

from src.preprocessing import clean_and_tokenize
from src.edit_distance import calculate_wer, calculate_levenshtein_distance
# Import 2 hàm đã xây dựng ở các file trước vào đây để dùng.
# "from src.X import Y" nghĩa là: vào thư mục src, mở file X.py, lấy hàm Y ra.


def evaluate_translation_quality(machine_file: str, reference_file: str) -> dict | None:
    """
    Hàm chính: đọc 2 file dịch, chấm điểm từng câu, xuất báo cáo tổng quan.

    :param machine_file:   Đường dẫn tới file bản dịch của máy
    :param reference_file: Đường dẫn tới file bản dịch chuẩn của con người
    :return: Dict chứa kết quả báo cáo, hoặc None nếu có lỗi đầu vào.
             (Trả về dict thay vì chỉ in ra để sau này có thể dùng kết quả trong code khác)
    """

    # -------------------------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU TỪ FILE
    # -------------------------------------------------------------------------

    # Dùng "with open(...) as ..." để Python tự đóng file sau khi đọc xong,
    # dù có lỗi xảy ra hay không. Nếu không dùng "with", file có thể bị "treo"
    # và gây lỗi khi chương trình khác cố đọc cùng lúc.
    #
    # encoding='utf-8' bắt buộc phải có vì tiếng Việt có dấu.
    # Thiếu dòng này, Python sẽ đọc sai ký tự và in ra ký tự lạ.
    #
    # Cú pháp "with A as a, B as b:" mở 2 file cùng lúc trong 1 khối lệnh,
    # gọn hơn viết 2 khối "with" lồng nhau.
    try:
        with open(machine_file, 'r', encoding='utf-8') as mf, \
             open(reference_file, 'r', encoding='utf-8') as rf:
            machine_lines = mf.readlines()
            # .readlines() đọc toàn bộ file và trả về 1 list, mỗi phần tử là 1 dòng.
            # Ví dụ: ["Xin chào.\n", "Tôi đi học.\n"]
            # --> Dấu "\n" (xuống dòng) vẫn còn, sẽ được xử lý bằng .strip() ở bước sau.
            reference_lines = rf.readlines()
    except OSError as e:
        # OSError bắt mọi lỗi liên quan đến file: không có quyền đọc, ổ đĩa lỗi, v.v.
        # Phân biệt với "file không tồn tại" đã được kiểm tra ở __main__ bên dưới.
        print(f"Lỗi khi đọc file: {e}")
        return None

    # -------------------------------------------------------------------------
    # BƯỚC 2: KIỂM TRA TÍNH HỢP LỆ CỦA DỮ LIỆU
    # -------------------------------------------------------------------------

    # Kiểm tra số dòng khớp nhau: Máy dịch 100 câu thì chuẩn phải có đúng 100 câu.
    # Nếu lệch, không thể ghép cặp câu để chấm điểm -> từ chối tiếp tục.
    if len(machine_lines) != len(reference_lines):
        print("LỖI: Hai file dữ liệu không có cùng số dòng!")
        print(f"  File máy dịch : {len(machine_lines)} dòng")
        print(f"  File chuẩn    : {len(reference_lines)} dòng")
        return None

    total_sentences = len(machine_lines)

    # Chặn lỗi chia cho 0: Nếu cả 2 file đều trống (0 dòng),
    # phép tính total_wer / total_sentences ở cuối sẽ crash.
    if total_sentences == 0:
        print("LỖI: Cả hai file đều rỗng, không có gì để đánh giá.")
        return None

    # -------------------------------------------------------------------------
    # BƯỚC 3: VÒNG LẶP CHẤM ĐIỂM TỪNG CÂU
    # -------------------------------------------------------------------------

    total_wer = 0.0
    # Biến tích lũy: cộng dồn WER của từng câu vào đây.
    # Khai báo là 0.0 (float) thay vì 0 (int) để Python không làm tròn kết quả
    # khi thực hiện phép cộng với số thực sau này.

    results = []
    # List lưu kết quả từng câu — vừa để in ra màn hình,
    # vừa để trả về cho caller dùng nếu cần (thay vì chỉ in rồi mất).

    print(f"BẮT ĐẦU ĐÁNH GIÁ {total_sentences} CÂU...\n")
    print("-" * 60)

    for i in range(total_sentences):
        # range(total_sentences) sinh ra dãy số 0, 1, 2, ..., total_sentences - 1
        # để duyệt qua từng chỉ số dòng.

        # .strip() xóa khoảng trắng và ký tự xuống dòng "\n" ở 2 đầu chuỗi.
        # Ví dụ: "  Xin chào.\n" --> "Xin chào."
        # Không strip thì bộ tokenizer sẽ nhận được "\n" như một ký tự thật.
        m_text = machine_lines[i].strip()
        r_text = reference_lines[i].strip()

        # Đưa câu thô vào bộ tiền xử lý (preprocessing):
        # Chuyển thường, xóa dấu câu, tách thành list từ.
        # Ví dụ: "Xin chào, thế giới!" --> ['xin', 'chào', 'thế', 'giới']
        m_tokens = clean_and_tokenize(m_text)
        r_tokens = clean_and_tokenize(r_text)

        # Tính số thao tác chỉnh sửa tối thiểu (Levenshtein distance)
        distance = calculate_levenshtein_distance(m_tokens, r_tokens)

        # Tính tỷ lệ lỗi (WER): distance / số từ trong câu chuẩn
        wer = calculate_wer(m_tokens, r_tokens)

        # Cộng dồn WER vào tổng để tính trung bình ở cuối
        total_wer += wer

        # Lưu kết quả câu này vào list để có thể dùng lại sau
        results.append({
            'sentence_index': i + 1,
            'machine':  m_text,
            'reference': r_text,
            'distance': distance,
            'wer':      wer,
        })

        print(f"Câu {i + 1}:")
        print(f"  Máy dịch : {m_text}")
        print(f"  Chuẩn    : {r_text}")
        print(f"  -> Lỗi: {distance} thao tác | WER: {wer * 100:.2f}%\n")
        # :.2f là định dạng số thực, chỉ hiển thị 2 chữ số sau dấu phẩy.
        # Ví dụ: 0.33333... --> "33.33%"

    # -------------------------------------------------------------------------
    # BƯỚC 4: TỔNG HỢP VÀ IN BÁO CÁO
    # -------------------------------------------------------------------------

    # Chia tổng WER cho số câu để lấy trung bình
    average_wer = (total_wer / total_sentences) * 100

    print("=" * 60)
    print("BAO CAO TONG QUAN (EVALUATION REPORT)")
    print("=" * 60)
    print(f"Tong so cau da cham       : {total_sentences}")
    print(f"Ty le loi trung binh (WER): {average_wer:.2f}%")

    # Phân loại chất lượng mô hình dựa trên ngưỡng WER tiêu chuẩn ngành
    if average_wer < 20:
        quality = "TOT (San sang su dung)"
    elif average_wer < 50:
        quality = "TRUNG BINH (Can cai thien them)"
    else:
        quality = "TE (Can dua ve Lab huan luyen lai)"

    print(f"Danh gia mo hinh          : {quality}")
    print("=" * 60)

    # Trả về dict kết quả thay vì None — người gọi hàm có thể dùng
    # kết quả này để vẽ biểu đồ, lưu file, hoặc test tự động.
    return {
        'total_sentences': total_sentences,
        'average_wer':     average_wer,
        'quality':         quality,
        'details':         results,
    }


# =============================================================================
# ĐIỂM KHỞI CHẠY CHƯƠNG TRÌNH
# =============================================================================

if __name__ == "__main__":
    # Khối này chỉ chạy khi bạn chạy trực tiếp "python main.py".
    # Nếu file này được import vào file khác thì khối này bị bỏ qua hoàn toàn.
    # --> Đây là quy ước chuẩn của Python để phân biệt "chạy" và "import".

    # os.path.abspath(__file__) lấy đường dẫn tuyệt đối của file main.py hiện tại.
    # os.path.dirname() lấy thư mục chứa file đó.
    # Kết quả: base_dir = "C:/Users/Phat/Desktop/translation-quality-analyzer"
    # --> Dù chạy từ thư mục nào, đường dẫn luôn đúng.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # os.path.join() ghép các phần đường dẫn lại với nhau đúng theo hệ điều hành.
    # Kết quả: ".../translation-quality-analyzer/data/machine_translation.txt"
    machine_path   = os.path.join(base_dir, "data", "machine_translation.txt")
    reference_path = os.path.join(base_dir, "data", "reference.txt")

    # Kiểm tra file tồn tại trước khi truyền vào hàm — thông báo lỗi rõ ràng
    # thay vì để Python tự crash với thông báo khó hiểu.
    if not os.path.exists(machine_path) or not os.path.exists(reference_path):
        print("LOI: Khong tim thay file du lieu trong thu muc 'data/'.")
        print("Vui long tao 'machine_translation.txt' va 'reference.txt' truoc khi chay.")
    else:
        evaluate_translation_quality(machine_path, reference_path)