from datasets import load_dataset
from deep_translator import GoogleTranslator
import os

def build_dataset():
    print("⏳ 1. Đang tải 1000 câu từ Hugging Face (Helsinki-NLP/opus-100)...")
    # ĐỔI THÀNH 1000 Ở ĐÂY
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train[:1000]")

    machine_lines = []
    reference_lines = []

    print("🤖 2. Đang khởi động 'Máy dịch' (Google Translator)...")
    translator = GoogleTranslator(source='en', target='vi')

    print("🔄 Bắt đầu dịch tự động 1000 câu (Vui lòng đợi 5-10 phút)...")
    
    # Dùng enumerate để đếm số thứ tự câu đang dịch
    for index, item in enumerate(dataset['translation']):
        en_text = item['en'].strip()
        vi_human = item['vi'].strip()
        
        try:
            vi_machine = translator.translate(en_text)
        except Exception as e:
            vi_machine = "lỗi dịch thuật"
        
        machine_lines.append(vi_machine + "\n")
        reference_lines.append(vi_human + "\n")
        
        # Cứ mỗi 50 câu thì in ra thông báo để biết máy không bị treo
        if (index + 1) % 50 == 0:
            print(f"  -> Đã dịch thành công {index + 1}/1000 câu...")

    print("\n💾 3. Đang lưu 1000 kết quả vào thư mục data/...")
    
    os.makedirs("data", exist_ok=True)

    with open("data/machine_translation.txt", "w", encoding="utf-8") as f:
        f.writelines(machine_lines)

    with open("data/reference.txt", "w", encoding="utf-8") as f:
        f.writelines(reference_lines)

    print("✅ Hoàn tất! 1000 dòng dữ liệu đã sẵn sàng để thử lửa!")

if __name__ == "__main__":
    build_dataset()