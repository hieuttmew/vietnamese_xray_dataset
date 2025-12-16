import os
import time
import torch
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from multiprocessing import cpu_count

# ==== CONFIG ====
root_dir = "../XQUANG"       # ⚙️ folder chứa X-ray
output_csv = "ocr_results.csv"
checkpoint_every = 200        # Lưu checkpoint sau mỗi 200 ảnh (OK)

# --- CẢI TIẾN ---
# Dùng nhiều nhân CPU hơn để đọc file
num_workers = min(4, cpu_count() * 2) 
# Batch size cho GPU: Tận dụng 12GB VRAM
# Thử 32 hoặc 64. Nếu lỗi "Out of Memory", giảm xuống 16.
GPU_BATCH_SIZE = 128 
# --- KẾT THÚC CẢI TIẾN ---

# ==== DEVICE & MODEL ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")
# CẢI TIẾN: Báo cho model biết batch size để nó tối ưu
model = ocr_predictor(pretrained=True).to(device)

# ==== RESUME FROM CHECKPOINT ====
processed_files = set()
if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    processed_files = set(df_existing["file_path"].tolist())
    print(f"🔁 Resume mode: {len(processed_files)} already processed")

# ==== COLLECT IMAGES ====
image_paths = []
for root, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            full_path = os.path.join(root, file)
            if full_path not in processed_files:
                image_paths.append(full_path)

print(f"📂 Found {len(image_paths)} new images to process")

# ==== UTILS ====
def load_doc(path):
    """Hàm load file (giữ nguyên)"""
    try:
        # DocumentFile.from_images là nơi CPU làm việc nặng
        return (path, DocumentFile.from_images(path))
    except Exception as e:
        # print(f"⚠️ Error loading {path}: {e}")
        return (path, None) # Trả về None để biết là lỗi

def extract_text_from_output(output):
    """Ghép toàn bộ text của ảnh thành 1 dòng (giữ nguyên)"""
    return " ".join([
        word.value
        for page in output.pages
        for block in page.blocks
        for line in block.lines
        for word in line.words
    ]).replace("\n", " ").strip()

# ==== MAIN LOOP (ĐÃ CẤU TRÚC LẠI) ====
results = []
start_time = time.time()
total_processed_count = 0

# Dùng ThreadPoolExecutor để quản lý các luồng I/O
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    
    # Chia danh sách tất cả các ảnh thành các lô (batch)
    for i in tqdm(range(0, len(image_paths), GPU_BATCH_SIZE), desc="🚀 Processing Batches"):
        
        batch_paths = image_paths[i : i + GPU_BATCH_SIZE]
        
        # 1. LOAD BATCH (DÙNG CPU SONG SONG)
        # executor.map sẽ chạy hàm load_doc cho mọi ảnh trong 
        # batch_paths, sử dụng num_workers luồng
        doc_data_batch = list(executor.map(load_doc, batch_paths))
        
        # Lọc ra các ảnh load thành công
        valid_docs = []
        valid_paths = []
        for path, doc in doc_data_batch:
            if doc:
                valid_docs.append(doc)
                valid_paths.append(path)
            else:
                print(f"⚠️ Bỏ qua file lỗi: {path}")

        if not valid_docs:
            continue

        # 2. INFERENCE (DÙNG GPU) VÀ 3. EXTRACT TEXT (DÙNG CPU)
        # Lặp qua từng (đường dẫn, tài liệu) đã được load thành công trong batch
        for path, doc in zip(valid_paths, valid_docs):
            try:
                # 🧠 Chạy model trên TỪNG ảnh (GPU làm việc)
                # CPU thread này sẽ chờ (nghỉ) trong khi GPU xử lý
                output = model(doc) 
                
                # 🧩 Extract text (CPU làm việc)
                text = extract_text_from_output(output)
                results.append({"file_path": path, "extracted_text": text})
                total_processed_count += 1
                
            except Exception as e:
                # Nếu 1 ảnh bị lỗi, in lỗi và bỏ qua CHỈ ảnh đó
                print(f"💥 Lỗi khi infer ảnh {path}: {e}")
                continue # Bỏ qua ảnh lỗi, tiếp tục ảnh tiếp theo

        # 4. CHECKPOINT (Logic giữ nguyên, nhưng hiệu quả hơn)
        # Lưu checkpoint khi đã tích đủ
        if len(results) >= checkpoint_every:
            df = pd.DataFrame(results)
            write_mode = "a" if os.path.exists(output_csv) else "w"
            header = not os.path.exists(output_csv)
            
            df.to_csv(output_csv, mode=write_mode, index=False, header=header)
            
            # print(f"💾 Checkpoint saved ({len(results)} new records)")
            results = [] # Xóa bộ đệm sau khi lưu

# ==== SAVE FINAL ====
if results:
    df = pd.DataFrame(results)
    write_mode = "a" if os.path.exists(output_csv) else "w"
    header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode=write_mode, index=False, header=header)

end_time = time.time()
print(f"✅ Done! Processed {total_processed_count} images.")
print(f"Total time: {(end_time - start_time)/60:.2f} min")