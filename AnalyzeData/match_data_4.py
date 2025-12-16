import pandas as pd
import unicodedata
from difflib import SequenceMatcher
from datetime import timedelta

# 1. Load dữ liệu
df_xray = pd.read_csv('AnalyzeData/dataset_xray_original_combined.csv')
df_ketqua = pd.read_csv('AnalyzeData/ketqua.csv')

# --- PRE-PROCESSING ---

def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).upper()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.strip()

# Chuẩn hóa tên
df_xray['norm_name'] = df_xray['patient_name'].apply(normalize_text)
df_ketqua['norm_name'] = df_ketqua['Name'].apply(normalize_text)

# Xử lý thời gian
df_xray['datetime_obj'] = pd.to_datetime(df_xray['datetime'], errors='coerce')
# Tạo cột Date (chỉ ngày) để khoanh vùng
df_xray['date_only'] = df_xray['datetime_obj'].dt.date

df_ketqua['date_start_obj'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ketqua['date_only'] = df_ketqua['date_start_obj'].dt.date

# Xử lý Năm sinh cho file Ketqua (Trích xuất từ cột Female/Male)
# Logic: Lấy cột Female nếu có, không thì lấy Male, bỏ chữ " tuổi" và ép kiểu số
def extract_age_to_year(row):
    age_str = row['Female'] if pd.notna(row['Female']) else row['Male']
    if pd.isna(age_str): return None
    try:
        age = int(str(age_str).replace(' tuổi', '').strip())
        # Giả sử dữ liệu năm 2024/2025, ta lấy năm hiện tại của record đó trừ đi tuổi
        # Hoặc đơn giản lấy 2024 - tuổi (cần linh hoạt chỗ này)
        current_year = row['date_start_obj'].year if pd.notna(row['date_start_obj']) else 2024
        return current_year - age
    except:
        return None

df_ketqua['calc_birth_year'] = df_ketqua.apply(extract_age_to_year, axis=1)

# --- MATCHING ENGINE ---

def find_best_match(row_xray, df_lookup):
    """
    Hàm tìm kiếm thông minh:
    1. Filter theo ngày (+/- 0 ngày).
    2. Tính điểm tương đồng tên.
    3. Cộng điểm nếu trùng năm sinh.
    """
    if pd.isna(row_xray['date_only']):
        return None, 0, "No Date"

    # 1. BLOCKING: Chỉ lấy danh sách khám trong cùng ngày
    # (Có thể mở rộng +/- 1 ngày nếu cần, nhưng cùng ngày là chuẩn nhất)
    candidates = df_lookup[df_lookup['date_only'] == row_xray['date_only']].copy()
    
    if candidates.empty:
        return None, 0, "Date Mismatch"

    best_score = 0
    best_candidate_idx = None
    
    xray_name = row_xray['norm_name']
    xray_year = row_xray['birth_year']

    # Duyệt qua các ứng viên trong ngày đó
    for idx, cand in candidates.iterrows():
        cand_name = cand['norm_name']
        
        # 2. FUZZY NAME MATCHING
        # SequenceMatcher trả về ratio 0.0 -> 1.0
        name_score = SequenceMatcher(None, xray_name, cand_name).ratio()
        
        # 3. BOOST SCORE NẾU TRÙNG NĂM SINH
        # Nếu năm sinh khớp nhau (hoặc lệch 1 năm do cách tính tuổi), cộng thêm điểm tự tin
        cand_year = cand['calc_birth_year']
        if pd.notna(xray_year) and pd.notna(cand_year):
            if abs(xray_year - cand_year) <= 1:
                name_score += 0.2 # Boost 20% nếu trùng năm sinh
            elif abs(xray_year - cand_year) > 5:
                name_score -= 0.3 # Trừ điểm nặng nếu năm sinh lệch quá xa (khác người)

        if name_score > best_score:
            best_score = name_score
            best_candidate_idx = idx

    # Ngưỡng chấp nhận (Threshold)
    # Nếu điểm > 0.6 (60%) thì coi là match. Nếu có trùng năm sinh thì điểm thường > 1.0
    if best_score > 0.6: 
        return best_candidate_idx, best_score, "Matched"
    else:
        return None, best_score, "Low Score"

# --- RUNNING THE MATCH ---
# Tạo list kết quả để convert thành DataFrame cho nhanh
results = []

print("Đang xử lý matching... vui lòng đợi...")
total_rows = len(df_xray)

for i, row in df_xray.iterrows():
    if i % 1000 == 0: print(f"Processing row {i}/{total_rows}")
    
    match_idx, score, status = find_best_match(row, df_ketqua)
    
    res = {
        'xray_index': i,
        'match_score': score,
        'match_status': status
    }
    
    if match_idx is not None:
        # Lấy thông tin từ dòng match được
        match_data = df_ketqua.loc[match_idx].to_dict()
        # Prefix 'ketqua_' để phân biệt
        for k, v in match_data.items():
            res[f'ketqua_{k}'] = v
    
    results.append(res)

# Tạo DataFrame kết quả
df_results = pd.DataFrame(results)

# Ghép lại với file gốc
df_final = df_xray.join(df_results.set_index('xray_index'))

# --- THỐNG KÊ ---
matched_count = df_final[df_final['match_status'] == 'Matched'].shape[0]
print(f"\nTotal Records: {total_rows}")
print(f"Matched: {matched_count}")
print(f"Match Rate: {(matched_count/total_rows)*100:.2f}%")

# Lưu file
# Bỏ bớt các cột tính toán phụ nếu cần
df_final.to_csv('AnalyzeData/result4/merged_xray_smart_match.csv', index=False)
print("File saved: merged_xray_smart_match.csv")