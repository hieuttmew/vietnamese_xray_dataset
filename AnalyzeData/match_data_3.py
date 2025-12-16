import pandas as pd
import unicodedata
from datetime import timedelta
import difflib
import re
import numpy as np

# --- CẤU HÌNH CHIẾN THUẬT ---

# Chiến thuật 1: Tìm kiếm tiêu chuẩn (Standard)
STRATEGY_STD_WINDOW = 24       # ±24 giờ
STRATEGY_STD_THRESHOLD = 0.65  # Tên giống > 65% (Chấp nhận sai sót OCR khá nhiều)

# Chiến thuật 2: Tìm kiếm mở rộng (Rescue - Cứu vớt ca sai ngày)
STRATEGY_WIDE_WINDOW = 7 * 24  # ±7 ngày
STRATEGY_WIDE_THRESHOLD = 0.85 # Tên giống > 85% (Yêu cầu tên chính xác cao để tránh nhầm người)

# --- HÀM HỖ TRỢ ---

def clean_text_advanced(input_str):
    """Chuẩn hóa tên triệt để"""
    if not isinstance(input_str, str):
        return ""
    # 1. Bỏ dấu
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    s = s.replace('Đ', 'D').replace('đ', 'd').upper().strip()
    # 2. Bỏ rác OCR
    garbage = [r'^BN:', r'^TEN:', r'^HO TEN:', r'^NAME:', r'[0-9]', r'[-_.]']
    for p in garbage:
        s = re.sub(p, ' ', s)
    # 3. Chỉ giữ chữ cái
    s = re.sub(r'[^A-Z\s]', '', s)
    return " ".join(s.split())

def get_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def find_matches_multi_layered(row, df_ocr_all):
    """
    Tìm kiếm đa tầng: Thử lớp 1, nếu thất bại thử lớp 2.
    Trả về list các ảnh khớp.
    """
    # 1. Xác định thời gian đích
    target_time = row['Date start']
    if pd.isna(target_time): target_time = row['Date assign']
    if pd.isna(target_time): return [], [], [] # Không có thời gian gốc

    target_name = row['Name_Clean']
    required_films = row['Film numbers']
    
    # Xử lý số lượng film yêu cầu (mặc định là 1 nếu không có data)
    try:
        limit_count = int(required_films) if pd.notna(required_films) and required_films > 0 else 1
    except:
        limit_count = 1

    # --- LỚP 1: TÌM KIẾM TIÊU CHUẨN ---
    matches = scan_candidates(target_time, target_name, df_ocr_all, 
                              window_hours=STRATEGY_STD_WINDOW, 
                              threshold=STRATEGY_STD_THRESHOLD)
    
    # --- LỚP 2: NẾU KHÔNG CÓ KẾT QUẢ -> TÌM KIẾM MỞ RỘNG (RESCUE) ---
    if not matches:
        matches = scan_candidates(target_time, target_name, df_ocr_all, 
                                  window_hours=STRATEGY_WIDE_WINDOW, 
                                  threshold=STRATEGY_WIDE_THRESHOLD)

    # --- XỬ LÝ KẾT QUẢ ---
    if not matches:
        return [], [], []

    # Sắp xếp kết quả: Ưu tiên điểm giống tên cao nhất -> sau đó đến thời gian gần nhất
    # (Để đảm bảo lấy đúng người trước, đúng giờ sau)
    matches.sort(key=lambda x: (-x['score'], abs((x['datetime'] - target_time).total_seconds())))

    # Lấy top N kết quả tốt nhất (N = Film numbers)
    # Nếu bro muốn lấy HẾT (dư cũng lấy) để ko sót dữ liệu, hãy comment dòng dưới lại
    # Tuy nhiên, lấy đúng số lượng thường tốt hơn để tránh duplicate
    
    # Logic thông minh: Nếu Film numbers = 1 nhưng tìm thấy 2 ảnh chụp cách nhau chỉ 1 giây -> Lấy cả 2 (có thể là chụp đúp)
    # Nếu cách nhau xa -> Lấy cái tốt nhất.
    # Ở đây tôi chọn giải pháp an toàn: Lấy tối đa = limit_count + 1 (Dư 1 chút cho chắc)
    selected_matches = matches[:limit_count + 2] 

    paths = [m['filepath'] for m in selected_matches]
    types = [m['xray_type'] for m in selected_matches]
    times = [m['datetime'] for m in selected_matches]

    return paths, types, times

def scan_candidates(target_time, target_name, df_source, window_hours, threshold):
    """Hàm quét core"""
    start_win = target_time - timedelta(hours=window_hours)
    end_win = target_time + timedelta(hours=window_hours)
    
    # Lọc nhanh theo thời gian (Vectorized operation - Siêu nhanh)
    mask = (df_source['datetime'] >= start_win) & (df_source['datetime'] <= end_win)
    candidates = df_source.loc[mask]
    
    if candidates.empty: return []

    results = []
    # Duyệt candidates
    for idx, cand in candidates.iterrows():
        score = get_similarity(target_name, cand['Name_Clean'])
        if score >= threshold:
            results.append({
                'filepath': cand['filepath'],
                'xray_type': cand['xray_type'],
                'datetime': cand['datetime'],
                'score': score
            })
    return results

# --- MAIN PROCESS ---

print("Đang đọc dữ liệu...")
df_ketqua = pd.read_csv('AnalyzeData/ketqua.csv')
df_ocr = pd.read_csv('AnalyzeData/dataset_xray_original_combined.csv')

# Pre-processing
print("Đang tiền xử lý...")
df_ketqua['Date start'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ketqua['Date assign'] = pd.to_datetime(df_ketqua['Date assign'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ocr['datetime'] = pd.to_datetime(df_ocr['datetime'], errors='coerce')

# Chỉ giữ dòng có datetime hợp lệ ở OCR
df_ocr = df_ocr.dropna(subset=['datetime']).sort_values('datetime')

# Advanced Name Cleaning
df_ketqua['Name_Clean'] = df_ketqua['Name'].apply(clean_text_advanced)
df_ocr['Name_Clean'] = df_ocr['patient_name'].apply(clean_text_advanced)

# --- MATCHING LOOP ---
print(f"Bắt đầu matching 2 lớp (Standard: ±{STRATEGY_STD_WINDOW}h, Rescue: ±{STRATEGY_WIDE_WINDOW/24:.0f} ngày)...")

final_paths = []
final_types = []
final_times = []

total = len(df_ketqua)
match_counter = 0

for idx, row in df_ketqua.iterrows():
    if idx % 2000 == 0:
        print(f" -> Xử lý {idx}/{total}... (Match: {match_counter})")
        
    paths, types, times = find_matches_multi_layered(row, df_ocr)
    
    if paths:
        match_counter += 1
        # Gộp list thành chuỗi phân cách bởi dấu ;
        final_paths.append(" ; ".join(str(x) for x in paths))
        final_types.append(" ; ".join(str(x) for x in types))
        final_times.append(" ; ".join(str(x) for x in times))
    else:
        final_paths.append(None)
        final_types.append(None)
        final_times.append(None)

# Gán kết quả
df_ketqua['img_filepaths'] = final_paths
df_ketqua['img_xray_types'] = final_types
df_ketqua['img_datetimes'] = final_times

# --- XUẤT FILE ---
df_final = df_ketqua.drop(columns=['Name_Clean'])
matched_total = df_final['img_filepaths'].notna().sum()

print("\n--- KẾT QUẢ FINAL ---")
print(f"Tổng số ca: {len(df_final)}")
print(f"Match thành công: {matched_total} ({matched_total/len(df_final)*100:.2f}%)")

df_final.to_csv('AnalyzeData/result3/merged_dataset_multi_image.csv', index=False)
print("Đã lưu file: merged_dataset_multi_image.csv")