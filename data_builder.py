from datasets import load_dataset
from deep_translator import GoogleTranslator
import os
import time

def build_dataset():
    print("1. Đang tải 1000 câu từ Hugging Face (Helsinki-NLP/opus-100)...")
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train[:1000]")

    machine_lines = []
    reference_lines = []

    print("2. Đang khởi động máy dịch (Google Translator)...")
    translator = GoogleTranslator(source='en', target='vi')

    print("Bắt đầu dịch tự động 1000 câu (Vui lòng đợi 5-10 phút)...")

    for index, item in enumerate(dataset['translation']):
        en_text = item['en'].strip()
        vi_human = item['vi'].strip()

        # FIX 1: Thêm retry loop — nếu Google trả về None hoặc lỗi thì thử lại tối đa 3 lần
        # Nguyên nhân crash gốc: translate() trả về None khi bị rate-limit,
        # sau đó None + "\n" gây TypeError ngay lập tức.
        vi_machine = None
        attempts = 0
        max_attempts = 3

        while vi_machine is None and attempts < max_attempts:
            try:
                result = translator.translate(en_text)

                # FIX 2: Kiểm tra None ngay sau khi nhận kết quả thay vì tin tưởng mù quáng
                # Google không crash khi bị rate-limit, nó chỉ âm thầm trả về None
                if result is not None:
                    vi_machine = result
                else:
                    attempts += 1
                    print(f"  [!] Câu {index + 1}: Google trả về None, thử lại lần {attempts}...")
                    # FIX 3: Chờ 2 giây trước khi thử lại để tránh bị chặn tiếp
                    time.sleep(2)

            except Exception as e:
                attempts += 1
                print(f"  [!] Câu {index + 1}: Lỗi '{e}', thử lại lần {attempts}...")
                time.sleep(2)

        # Nếu thử 3 lần vẫn thất bại thì ghi nhận và đi tiếp, không để crash cả chương trình
        if vi_machine is None:
            vi_machine = "[DICH_THAT_BAI]"
            print(f"  [X] Câu {index + 1}: Bỏ qua sau {max_attempts} lần thất bại.")

        machine_lines.append(vi_machine + "\n")
        reference_lines.append(vi_human + "\n")

        if (index + 1) % 50 == 0:
            print(f"  -> Đã dịch thành công {index + 1}/1000 câu...")
            # Nghỉ 1 giây mỗi 50 câu để Google không chặn
            time.sleep(1)

    print("\n3. Đang lưu 1000 kết quả vào thư mục data/...")

    os.makedirs("data", exist_ok=True)

    with open("data/machine_translation.txt", "w", encoding="utf-8") as f:
        f.writelines(machine_lines)

    with open("data/reference.txt", "w", encoding="utf-8") as f:
        f.writelines(reference_lines)

    print("Hoan tat! 1000 dong du lieu da san sang!")

if __name__ == "__main__":
    build_dataset()