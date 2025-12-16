import pandas as pd
import unicodedata
from datetime import timedelta
import numpy as np

# --- 1. HÀM HỖ TRỢ CHUẨN HÓA ---

def remove_accents(input_str):
    """
    Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu và viết hoa.
    """
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return s.replace('Đ', 'D').replace('đ', 'd').upper().strip()

def find_best_match(row, df_candidates, time_threshold_hours=24):
    """
    Phiên bản FIX LỖI: Kiểm tra kỹ dữ liệu thời gian trước khi tìm min.
    """
    # FIX: Lọc bỏ ngay các dòng candidates mà cột datetime bị NaT (Lỗi/Rỗng)
    # Nếu không lọc bước này, phép tính trừ bên dưới sẽ ra NaT, gây lỗi cho hàm idxmin()
    valid_candidates = df_candidates.dropna(subset=['datetime'])
    
    if valid_candidates.empty:
        return None, None, None

    # Lấy mốc thời gian từ file kết quả
    target_time = row['Date start']
    if pd.isna(target_time):
        target_time = row['Date assign']
    
    if pd.isna(target_time):
        return None, None, None

    # Tính khoảng cách thời gian
    # Sử dụng try-except để xử lý trường hợp lệch múi giờ (tz-aware vs tz-naive) nếu có
    try:
        time_diffs = (valid_candidates['datetime'] - target_time).abs()
    except TypeError:
        # Nếu lỗi timezone, quy tất cả về dạng không múi giờ (naive)
        c_times = pd.to_datetime(valid_candidates['datetime']).dt.tz_localize(None)
        t_time = pd.to_datetime(target_time).tz_localize(None)
        time_diffs = (c_times - t_time).abs()

    # FIX: Sau khi trừ xong, lọc bỏ các giá trị NaT trong kết quả (nếu có)
    time_diffs = time_diffs.dropna()

    if time_diffs.empty:
        return None, None, None
        
    min_diff = time_diffs.min()
    
    # Kiểm tra ngưỡng thời gian (ví dụ: chụp trong vòng 24h so với lúc chỉ định)
    if min_diff > timedelta(hours=time_threshold_hours):
        return None, None, None

    # Tìm index của dòng có khoảng cách thời gian nhỏ nhất
    best_match_idx = time_diffs.idxmin()
    best_match_row = valid_candidates.loc[best_match_idx]
    
    return best_match_row['filepath'], best_match_row['xray_type'], best_match_row['datetime']

# --- 2. LOAD VÀ XỬ LÝ DỮ LIỆU ---

print("Đang đọc dữ liệu...")
# Load file gốc (Kết quả chẩn đoán)
df_ketqua = pd.read_csv('AnalyzeData/ketqua.csv')

# Load file phụ (OCR từ ảnh)
df_ocr = pd.read_csv('AnalyzeData/dataset_xray_original_combined.csv')

print(f"- File gốc (ketqua): {len(df_ketqua)} dòng")
print(f"- File phụ (OCR): {len(df_ocr)} dòng")

# A. Xử lý thời gian (Quan trọng: errors='coerce' để biến lỗi thành NaT thay vì dừng chương trình)
print("Đang xử lý thời gian...")
df_ketqua['Date start'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y %H:%M', errors='coerce')
df_ketqua['Date assign'] = pd.to_datetime(df_ketqua['Date assign'], format='%d/%m/%Y %H:%M', errors='coerce')

# File OCR thường format YYYY-MM-DD HH:MM:SS, để auto detect cho an toàn
df_ocr['datetime'] = pd.to_datetime(df_ocr['datetime'], errors='coerce')

# B. Chuẩn hóa tên để so sánh
print("Đang chuẩn hóa tên bệnh nhân...")
df_ketqua['Name_Norm'] = df_ketqua['Name'].apply(remove_accents)
df_ocr['Name_Norm'] = df_ocr['patient_name'].apply(remove_accents)

# --- 3. MATCHING (KHỚP DỮ LIỆU) ---

print("Đang thực hiện ghép nối (Matching)...")
matched_filepaths = []
matched_xray_types = []
matched_ocr_datetimes = []

# Duyệt qua từng dòng của file kết quả
for idx, row in df_ketqua.iterrows():
    if idx % 1000 == 0 and idx > 0:
        print(f"  -> Đã xử lý {idx}/{len(df_ketqua)} dòng...")

    patient_name_norm = row['Name_Norm']
    
    if not patient_name_norm:
        matched_filepaths.append(None)
        matched_xray_types.append(None)
        matched_ocr_datetimes.append(None)
        continue
        
    # Lọc danh sách ứng viên trùng tên từ file OCR
    candidates = df_ocr[df_ocr['Name_Norm'] == patient_name_norm]
    
    # Tìm ứng viên khớp nhất về thời gian
    fpath, xtype, ocr_time = find_best_match(row, candidates, time_threshold_hours=24)
    
    matched_filepaths.append(fpath)
    matched_xray_types.append(xtype)
    matched_ocr_datetimes.append(ocr_time)

# Gán kết quả vào DataFrame gốc
df_ketqua['img_filepath'] = matched_filepaths
df_ketqua['img_xray_type'] = matched_xray_types
df_ketqua['img_datetime'] = matched_ocr_datetimes

# --- 4. XUẤT FILE ---

# Xóa cột tạm Name_Norm cho gọn
df_final = df_ketqua.drop(columns=['Name_Norm'])

# Thống kê kết quả
matched_count = df_final['img_filepath'].notna().sum()
print(f"\n--- HOÀN TẤT ---")
print(f"Tổng số ca khám: {len(df_final)}")
print(f"Số ca tìm thấy ảnh khớp: {matched_count} ({matched_count/len(df_final)*100:.2f}%)")

# Lưu file (Chỉ lưu những ca có ảnh khớp nếu muốn gọn, hoặc lưu hết)
output_filename = 'merged_dataset_final.csv'
df_final.to_csv(output_filename, index=False)
print(f"Đã lưu file kết quả tại: {output_filename}")

# Lưu riêng file chỉ chứa các ca khớp được ảnh (để dễ kiểm tra)
df_matched_only = df_final.dropna(subset=['img_filepath'])
df_matched_only.to_csv('merged_dataset_matched_only.csv', index=False)
print(f"Đã lưu file chỉ chứa các ca khớp ảnh tại: merged_dataset_matched_only.csv")