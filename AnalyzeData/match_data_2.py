import pandas as pd
import unicodedata
from datetime import timedelta
import difflib # Thư viện so sánh chuỗi mờ
import re

# --- 1. CẤU HÌNH ---
# Độ giống nhau tối thiểu của tên để chấp nhận (0.0 đến 1.0). 
# 0.75 nghĩa là giống 75% là chốt. Hạ xuống nếu OCR quá tệ, tăng lên nếu muốn chính xác tuyệt đối.
NAME_SIMILARITY_THRESHOLD = 0.7

# Khoảng thời gian tìm kiếm xung quanh giờ khám (giờ)
SEARCH_WINDOW_HOURS = 24 

# --- 2. HÀM HỖ TRỢ ---

def clean_text_advanced(input_str):
    """
    Chuẩn hóa tên mạnh tay hơn:
    - Bỏ dấu tiếng Việt
    - Bỏ ký tự đặc biệt (dấu chấm, phẩy, gạch ngang thường gặp trong tên dân tộc K'..., H'...)
    - Bỏ các từ rác OCR thường gặp (BN:, TEN:, ...)
    """
    if not isinstance(input_str, str):
        return ""
    
    # 1. Bỏ dấu
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    s = s.replace('Đ', 'D').replace('đ', 'd').upper().strip()
    
    # 2. Bỏ từ khóa rác thường dính vào tên do OCR
    garbage_patterns = [r'^BN:', r'^TEN:', r'^HO TEN:', r'^NAME:', r'[0-9]']
    for p in garbage_patterns:
        s = re.sub(p, '', s)

    # 3. Chỉ giữ lại chữ cái (A-Z) và khoảng trắng. 
    # Logic này giúp K'Ho -> KHO, Liêng Hót -> LIENG HOT (bỏ dấu ' để dễ so khớp)
    s = re.sub(r'[^A-Z\s]', '', s)
    
    # 4. Xóa khoảng trắng thừa
    return " ".join(s.split())

def get_similarity(s1, s2):
    """Tính độ giống nhau giữa 2 chuỗi (0 -> 1)"""
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def find_best_match_fuzzy(row, df_ocr_all):
    """
    Tìm ảnh khớp dựa trên thời gian trước, sau đó so khớp tên.
    """
    # 1. Xác định mốc thời gian mục tiêu
    target_time = row['Date start']
    if pd.isna(target_time):
        target_time = row['Date assign']
    
    if pd.isna(target_time):
        return None, None, None, 0.0 # Không có thời gian -> Bó tay

    # 2. LỌC NHANH: Chỉ lấy dữ liệu OCR trong khoảng ±WINDOW hours
    # Bước này cực quan trọng để tăng tốc độ và độ chính xác
    start_window = target_time - timedelta(hours=SEARCH_WINDOW_HOURS)
    end_window = target_time + timedelta(hours=SEARCH_WINDOW_HOURS)
    
    # Lấy tập ứng viên trong khung giờ này (Candidate Pool)
    # df_ocr_all cần được sort theo datetime trước để slice nhanh hơn (đã làm ở main)
    # Dùng mask boolean
    mask = (df_ocr_all['datetime'] >= start_window) & (df_ocr_all['datetime'] <= end_window)
    candidates = df_ocr_all.loc[mask]
    
    if candidates.empty:
        return None, None, None, 0.0

    # 3. SO KHỚP TÊN (FUZZY MATCHING) trong tập ứng viên
    target_name = row['Name_Clean']
    
    best_score = 0.0
    best_candidate = None
    
    # Duyệt qua các ứng viên trong khung giờ
    for idx, cand in candidates.iterrows():
        cand_name = cand['Name_Clean']
        
        # Tính độ giống tên
        score = get_similarity(target_name, cand_name)
        
        if score > best_score:
            best_score = score
            best_candidate = cand
            
            # Nếu giống tuyệt đối 100% thì dừng tìm luôn cho nhanh
            if score == 1.0:
                break
    
    # 4. Kiểm tra ngưỡng chấp nhận
    if best_score >= NAME_SIMILARITY_THRESHOLD and best_candidate is not None:
        return best_candidate['filepath'], best_candidate['xray_type'], best_candidate['datetime'], best_score
    
    return None, None, None, best_score

# --- 3. XỬ LÝ DỮ LIỆU CHÍNH ---

print("Đang đọc dữ liệu...")
df_ketqua = pd.read_csv('AnalyzeData/ketqua.csv')
df_ocr = pd.read_csv('AnalyzeData/dataset_xray_original_combined.csv')

print(f"Số lượng KẾT QUẢ: {len(df_ketqua)}")
print(f"Số lượng ẢNH OCR: {len(df_ocr)}")

# A. Xử lý thời gian
print("Đang chuẩn hóa thời gian...")
df_ketqua['Date start'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ketqua['Date assign'] = pd.to_datetime(df_ketqua['Date assign'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ocr['datetime'] = pd.to_datetime(df_ocr['datetime'], errors='coerce')

# B. Loại bỏ dòng không có thời gian ở OCR (vì chiến thuật này dựa vào thời gian)
df_ocr = df_ocr.dropna(subset=['datetime'])

# C. Chuẩn hóa tên (Advanced)
print("Đang chuẩn hóa tên (Advanced Cleaning)...")
df_ketqua['Name_Clean'] = df_ketqua['Name'].apply(clean_text_advanced)
df_ocr['Name_Clean'] = df_ocr['patient_name'].apply(clean_text_advanced)

# D. Sắp xếp file OCR theo thời gian (Để tối ưu tốc độ tìm kiếm)
df_ocr = df_ocr.sort_values('datetime')

# --- 4. MATCHING LOOP ---

print("Đang thực hiện Fuzzy Matching (chiến thuật Thời gian trước -> Tên sau)...")
print(f"Ngưỡng chấp nhận độ giống tên: {NAME_SIMILARITY_THRESHOLD*100}%")

matched_data = [] # List lưu dict kết quả

total_rows = len(df_ketqua)
matched_count = 0

for idx, row in df_ketqua.iterrows():
    if idx % 1000 == 0:
        print(f"  -> Đã quét {idx}/{total_rows} dòng... (Match được: {matched_count})")
        
    fpath, xtype, ocr_time, score = find_best_match_fuzzy(row, df_ocr)
    
    if fpath:
        matched_count += 1
        
    matched_data.append({
        'img_filepath': fpath,
        'img_xray_type': xtype,
        'img_datetime': ocr_time,
        'match_score': score # Lưu lại điểm để kiểm tra độ tin cậy
    })

# Tạo DataFrame từ kết quả matching
df_results = pd.DataFrame(matched_data)

# Ghép vào DataFrame gốc
df_final = pd.concat([df_ketqua.reset_index(drop=True), df_results], axis=1)

# --- 5. XUẤT FILE & THỐNG KÊ ---

# Xóa cột tạm Name_Clean
df_final = df_final.drop(columns=['Name_Clean'])

print("\n--- KẾT QUẢ CUỐI CÙNG ---")
final_match_count = df_final['img_filepath'].notna().sum()
print(f"Tổng số ca: {len(df_final)}")
print(f"Match thành công: {final_match_count} ({final_match_count/len(df_final)*100:.2f}%)")

# Lưu file Full
df_final.to_csv('AnalyzeData/result2/merged_dataset_fuzzy_full.csv', index=False)
print("Đã lưu: merged_dataset_fuzzy_full.csv")

# Lưu file để debug: Gồm tên gốc, tên OCR match được và điểm số
# Giúp bro xem tại sao nó match hoặc không match
debug_cols = ['Name', 'Date start', 'img_filepath', 'img_datetime', 'match_score']
# Nếu bro muốn xem cả tên trong OCR để so sánh thì phải merge lại hơi phức tạp, 
# nhưng match_score cao là an tâm.
df_debug = df_final[df_final['img_filepath'].notna()][debug_cols]
df_debug.to_csv('AnalyzeData/result2/debug_match_quality.csv', index=False)
print("Đã lưu file kiểm tra chất lượng: debug_match_quality.csv")