import pandas as pd
import re

# --- CẤU HÌNH TỪ KHÓA ---

# 1. Mapping X-Ray (Chuẩn hóa)
XRAY_MAPPING = {
    'CHEST': ['CHEST', 'THORAX', 'LUNG', 'RIB', 'CLAVICLE', 'HEST', 'CNEST', 'PHOI', 'GHEST', 'CHESTPA', 'CHESTAP'],
    'SKULL': ['SKULL', 'HEAD', 'SINUS', 'MANDIBLE', 'ZYGOMA', 'NASAL', 'FACE', 'SO', 'HAM'],
    'SPINE': ['SPINE', 'CERVICAL', 'LUMBAR', 'THORACIC', 'SACRUM', 'COCCYX', 'NECK', 'COT SONG', 'L-SPINE', 'C-SPINE', 'T-SPINE', 'LSPINE', 'CSPINE'],
    'ABDOMEN': ['ABDOMEN', 'ABD', 'KUB', 'BUNG'],
    'PELVIS': ['PELVIS', 'HIP', 'SI JOINT', 'CHAU'],
    'KNEE': ['KNEE', 'PATELLA', 'GOI'],
    'SHOULDER': ['SHOULDER', 'SCAPULA', 'VAI'],
    'HAND': ['HAND', 'FINGER', 'THUMB', 'DIGIT', 'BAN TAY', 'NGON TAY'],
    'WRIST': ['WRIST', 'CARPAL', 'CO TAY'],
    'ELBOW': ['ELBOW', 'KHUYU'],
    'FOREARM': ['FOREARM', 'RADIUS', 'ULNA', 'CANG TAY'],
    'FOOT': ['FOOT', 'TOE', 'HEEL', 'CALCANEUS', 'BAN CHAN'],
    'ANKLE': ['ANKLE', 'CO CHAN'],
    'HUMERUS': ['HUMERUS', 'ARM', 'CANH TAY'],
    'FEMUR': ['FEMUR', 'THIGH', 'DUI'],
    'LEG': ['LEG', 'TIBIA', 'FIBULA', 'CANG CHAN']
}

# 2. Nhận diện Tên người Việt (Họ phổ biến)
VN_SURNAMES = [
    'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG', 'HUYNH', 'PHAN', 'VU', 'VO', 'DANG', 
    'BUI', 'DO', 'HO', 'NGO', 'DUONG', 'LY', 'TRINH', 'DINH', 'LAM', 'MAI', 
    'LUONG', 'TRUONG', 'THAI', 'CAO', 'CHAU', 'TA', 'PHUNG', 'KHUONG', 'QUACH', 
    'LA', 'TON', 'DIEP', 'KIEU', 'DAM', 'SON', 'CHU', 'THI', 'VAN', 'TIT', 'TIT'
]

# 3. Nhận diện Rác địa chỉ (TTYTHUYENDUCTRONG và biến thể)
LOCATION_NOISE = [
    'TTYT', 'HUYEN', 'DUC', 'TRONG', 'YTE', 'HUTEN', 'DUG', 'RONG', 'LAM', 'DONG', 
    'CENTER', 'HOSPITAL', 'TYTHUYEN', 'TEHUYEN', 'LHUYEN', 'UHUYEN', 'HUTENDUC'
]

def analyze_label(label):
    """
    Phân tích nhãn để xác định Category và Clean_Label
    Trả về: (Category, Normalized_Value)
    """
    if not isinstance(label, str):
        return "OTHERS", str(label)
    
    label_upper = label.upper().strip()
    
    # 1. Kiểm tra X-Ray (Ưu tiên chuẩn hóa)
    for standard_label, keywords in XRAY_MAPPING.items():
        for kw in keywords:
            if kw in label_upper:
                return "XRAY_LABELS", standard_label
    
    # 2. Kiểm tra Địa chỉ/Rác OCR (TTYTHUYENDUCTRONG)
    # Check regex trước cho các biến thể dính chùm
    if re.search(r'(TTY|HUT|HUY|DUC|TRONG|YTE|HUYEN)', label_upper):
        return "LOCATION_NOISE", label # Giữ nguyên nhãn gốc vì là rác
        
    count_matches = sum(1 for kw in LOCATION_NOISE if kw in label_upper)
    if count_matches >= 1:
        return "LOCATION_NOISE", label

    # 3. Kiểm tra Tên người Việt
    for surname in VN_SURNAMES:
        if label_upper.startswith(surname):
            return "VIETNAMESE_NAMES", label

    # 4. Còn lại
    return "OTHERS", label

# --- XỬ LÝ CHÍNH ---

def process_dataset(input_file, output_files: dict = None):
    try:
        # Load dữ liệu
        df = pd.read_csv(input_file)
        print(f"Đã đọc file {input_file} với {len(df)} dòng.")
        
        # Kiểm tra cột xray_type có tồn tại không
        if 'xray_type' not in df.columns:
            print("Lỗi: Không tìm thấy cột 'xray_type' trong file CSV.")
            return

        # 1. Phân tích và tạo cột tạm để lưu kết quả phân loại
        # Hàm analyze_label trả về (Category, New_Label)
        analysis_result = df['xray_type'].apply(lambda x: analyze_label(x))
        
        # Tách kết quả thành 2 cột riêng
        df['Temp_Category'] = [res[0] for res in analysis_result]
        df['Temp_Clean_Label'] = [res[1] for res in analysis_result]

        # 2. Thay thế giá trị trong cột xray_type CHỈ KHI nó là nhãn X-Ray chuẩn
        # Các trường hợp rác (Tên, Địa chỉ) giữ nguyên giá trị gốc trong cột xray_type
        mask_valid = df['Temp_Category'] == 'XRAY_LABELS'
        df.loc[mask_valid, 'xray_type'] = df.loc[mask_valid, 'Temp_Clean_Label']

        # 3. Xuất file theo từng nhóm
        if output_files is None:
            output_files = {
                "XRAY_LABELS": "dataset_xray_original.csv",
                "LOCATION_NOISE": "dataset_noisy_address.csv",
                "VIETNAMESE_NAMES": "dataset_noisy_name.csv",
                "OTHERS": "dataset_noisy_other.csv"
            }

        # Danh sách các cột gốc cần giữ lại
        original_columns = ['filepath', 'patient_name', 'birth_year', 'xray_type', 'datetime']
        
        # Đảm bảo chỉ lấy các cột có thực tế trong file (phòng trường hợp file thiếu cột nào đó)
        valid_columns = [col for col in original_columns if col in df.columns]

        print("\n--- BẮT ĐẦU XUẤT FILE ---")
        for category, filename in output_files.items():
            # Lọc dữ liệu theo Category
            subset = df[df['Temp_Category'] == category].copy()
            
            # Chỉ lấy các cột gốc ban đầu (xray_type đã được chuẩn hóa nếu là XRAY)
            final_subset = subset[valid_columns]
            
            # Xuất file
            final_subset.to_csv(filename, index=False)
            print(f"-> Đã xuất: {filename} - Số lượng: {len(subset)}")
            
            if category == "XRAY_LABELS":
                print(f"   (Ghi chú: Cột xray_type trong file này đã được chuẩn hóa)")
            else:
                print(f"   (Ghi chú: Cột xray_type giữ nguyên gốc để kiểm tra)")

        print("\nHoàn tất xử lý!")

    except FileNotFoundError:
        print("Không tìm thấy file input. Đang tạo dữ liệu mẫu để test logic...")
        create_mock_data_and_test()

def create_mock_data_and_test():
    # Tạo dữ liệu giả lập để test nếu không có file thật
    data = {
        'filepath': ['/img/1.jpg', '/img/2.jpg', '/img/3.jpg', '/img/4.jpg', '/img/5.jpg'],
        'patient_name': ['A', 'B', 'C', 'D', 'E'],
        'birth_year': [1990, 1991, 1992, 1993, 1994],
        'xray_type': ['CHESTPA', 'NGUYEN', 'TTYTHUYENDUCTRONG', 'SKULLAP', 'TEST_DATA'],
        'datetime': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    df_mock = pd.DataFrame(data)
    df_mock.to_csv('dataset_mock.csv', index=False)
    process_dataset('dataset_mock.csv')