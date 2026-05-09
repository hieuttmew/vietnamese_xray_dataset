import os
import time
import logging
import re
import unicodedata
import ast
import json
import shutil
import traceback
import concurrent.futures
from datetime import timedelta, datetime
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import cv2
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import easyocr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import cpu_count

class MedicalXRayPipeline:
    """
    End-to-End Medical X-Ray Data Processing Pipeline.
    Quy trình xử lý dữ liệu X-Quang y tế tự động (từ Ảnh thô đến AI-ready dataset).
    """

    # --- CONSTANTS & MAPPINGS ---
    # NOISE_WORDS: Từ khóa nhiễu OCR thường gặp / Common OCR noise keywords
    NOISE_WORDS = [
        "TTYT", "HUYEN", "TRUNG", "TAM", "Y", "TE", "DUC", "TRONG",
        "CENTER", "BENH", "VIEN", "PKDK", "PHONG KHAM"
    ]
    NOISE_PATTERN = re.compile(r'\b(?:' + '|'.join(NOISE_WORDS) + r')\b', re.IGNORECASE)

    # Regex patterns
    RE_DATE = re.compile(r'(\d{2}/\d{2}/\d{4}\s*\d{0,2}:?\d{0,2}:?\d{0,2})')
    RE_NAME = re.compile(r'([A-ZÀ-Ỹ][A-ZÀ-Ỹ\s\.]+)\s+((19|20)\d{2})', re.UNICODE)
    RE_XRAY = re.compile(r'([A-Z\-]+\s*(?:AP|LAT|PA|OBL|VIEW)?)', re.UNICODE)

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

    VN_SURNAMES = [
        'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG', 'HUYNH', 'PHAN', 'VU', 'VO', 'DANG', 
        'BUI', 'DO', 'HO', 'NGO', 'DUONG', 'LY', 'TRINH', 'DINH', 'LAM', 'MAI', 
        'LUONG', 'TRUONG', 'THAI', 'CAO', 'CHAU', 'TA', 'PHUNG', 'KHUONG', 'QUACH', 
        'LA', 'TON', 'DIEP', 'KIEU', 'DAM', 'SON', 'CHU', 'THI', 'VAN', 'TIT'
    ]

    LOCATION_NOISE = [
        'TTYT', 'HUYEN', 'DUC', 'TRONG', 'YTE', 'HUTEN', 'DUG', 'RONG', 'LAM', 'DONG', 
        'CENTER', 'HOSPITAL', 'TYTHUYEN', 'TEHUYEN', 'LHUYEN', 'UHUYEN', 'HUTENDUC'
    ]

    LABEL_MAPPING_V2 = {
        "NORMAL": ["bình thường", "không thấy hình ảnh bệnh lý", "không thấy tổn thương", "tim phổi bình thường", "sáng đều", "trong giới hạn", "chưa thấy", "không có dấu hiệu", "kết quả bình thường", "không thấy hình ảnh"],
        "PNEUMONIA/INFILTRATION": ["viêm phổi", "đám mờ", "thâm nhiễm", "nốt mờ", "kính mờ", "phế bào", "tổn thương nhu mô", "đông đặc"],
        "TUBERCULOSIS": ["lao", "xơ", "vôi hóa", "đỉnh phổi", "nốt vôi"],
        "PLEURAL_EFFUSION": ["tràn dịch", "tù góc sườn hoành", "mờ góc sườn hoành", "dày màng phổi"],
        "CARDIOMEGALY": ["tim to", "bóng tim lớn", "chỉ số tim ngực", "bè ngang"],
        "BRONCHITIS": ["viêm phế quản", "dày thành phế quản", "tăng đậm vân phổi", "rốn phổi đậm", "nhánh phế quản"],
        "FRACTURE": ["gãy", "vỡ", "nứt", "di lệch", "gián đoạn", "đường sáng", "mất liên tục"],
        "DEGENERATION": ["thoái hóa", "gai xương", "đặc xương", "hẹp khe khớp", "mỏ xương", "xơ xương", "biến đổi thoái hóa"],
        "SPINAL_ALIGNMENT": ["cong", "vẹo", "trượt đốt sống", "gù", "mất đường cong", "thẳng trục"],
        "SPONDYLOSIS/DISC": ["xẹp đốt sống", "biến dạng", "thoát vị", "lún", "giảm chiều cao"],
        "SINUSITIS": ["viêm xoang", "mờ xoang", "dày niêm mạc", "dịch trong xoang", "ngách mũi"],
        "HARDWARE/SURGERY": ["kết hợp xương", "nẹp vít", "đinh", "dụng cụ", "cố định", "sau mổ", "phẫu thuật"],
        "SOFT_TISSUE": ["mô mềm", "dị vật", "phù nề", "sưng"]
    }

    # Các nhãn cần loại bỏ ở bước cân bằng dữ liệu / Labels to drop during balancing
    DROP_XRAY_TYPES = ['PELVIS', 'ELBOW', 'ABDOMEN', 'HUMERUS', 'FEMUR']


    def __init__(self, root_img_dir, hospital_csv_path, output_dir, ssd_resize_dir,
                 gpu_batch_size=128, img_size=256, ocr_checkpoint_every=200):
        """
        Khởi tạo Pipeline / Initialize the pipeline.
        """
        self.root_img_dir = root_img_dir
        self.hospital_csv_path = hospital_csv_path
        self.output_dir = output_dir
        self.ssd_resize_dir = ssd_resize_dir
        self.gpu_batch_size = gpu_batch_size
        self.img_size = img_size
        self.ocr_checkpoint_every = ocr_checkpoint_every
        
        # Tạo các thư mục lưu kết quả / Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ssd_resize_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ssd_resize_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.ssd_resize_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.ssd_resize_dir, 'test'), exist_ok=True)
        
        # Cấu hình logging / Setup logging
        self.logger = self._setup_logger()
        
        # File checkpoints
        self.ckpt_step1 = os.path.join(self.output_dir, "step1_ocr_raw.csv")
        self.ckpt_step2 = os.path.join(self.output_dir, "step2_cleaned_data.csv")
        self.ckpt_step3 = os.path.join(self.output_dir, "step3_matched_dataset.csv")
        self.ckpt_step4 = os.path.join(self.output_dir, "step4_labeled_dataset.csv")

    def _setup_logger(self):
        """Cấu hình logger để ghi ra file và in ra console / Setup logger for file & console."""
        logger = logging.getLogger("MedicalXRayPipeline")
        logger.setLevel(logging.INFO)
        
        # Nếu logger đã có handler thì xoá đi để không bị lặp / Remove handlers if exist
        if logger.hasHandlers():
            logger.handlers.clear()
            
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File Handler
        log_file = os.path.join(self.output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger

    # ==========================================
    # UTILITY FUNCTIONS / CÁC HÀM TIỆN ÍCH
    # ==========================================
    @staticmethod
    def load_doc_safe(path):
        """Hàm load file ảnh an toàn cho docTr / Safe image load for docTr"""
        try:
            return (path, DocumentFile.from_images(path))
        except Exception as e:
            return (path, None)

    @staticmethod
    def extract_text_from_doctr(output):
        """Trích xuất text từ output của docTr / Extract text from docTr output"""
        return " ".join([
            word.value
            for page in output.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
        ]).replace("\n", " ").strip()

    @staticmethod
    def clean_noise(text):
        """Xóa nhiễu cố định / Remove fixed noise"""
        text = re.sub(r'[-_]', ' ', text)
        text = re.sub(MedicalXRayPipeline.NOISE_PATTERN, '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def extract_datetime(text):
        """Trích xuất ngày giờ / Extract datetime"""
        m = MedicalXRayPipeline.RE_DATE.search(text)
        if m:
            s = m.group(1).strip()
            try:
                if len(s.split()) == 2:
                    return datetime.strptime(s, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
            except:
                pass
        return ""

    @staticmethod
    def extract_name_birth(text):
        """Trích xuất tên và năm sinh / Extract name and birth year"""
        m = MedicalXRayPipeline.RE_NAME.search(text)
        if m:
            return m.group(1).strip(), m.group(2)
        return "", ""

    @staticmethod
    def extract_xray_type(text):
        """Trích xuất loại ảnh X-Quang thô / Extract raw X-Ray type"""
        candidates = MedicalXRayPipeline.RE_XRAY.findall(text)
        filtered = [c.strip() for c in candidates if len(c) >= 3 and not re.match(r'^(SERIES|IMAGE|UNIT|PIXEL|ADMIN|W|L)$', c)]
        if filtered:
            return max(filtered, key=len) # Chọn cái dài nhất / Pick the longest
        return ""

    @staticmethod
    def normalize_and_classify(label):
        """Chuẩn hóa nhãn X-Ray / Normalize X-Ray label"""
        if not isinstance(label, str):
            return "OTHERS", str(label)
        
        original_label = label
        label_upper = label.upper().strip()
        
        # 1. Kiểm tra X-Ray
        for standard_label, keywords in MedicalXRayPipeline.XRAY_MAPPING.items():
            for kw in keywords:
                if kw in label_upper:
                    return "XRAY_LABELS", standard_label
        
        # 2. Kiểm tra Nhiễu địa chỉ
        if re.search(r'(TTY|HUT|HUY|DUC|TRONG|YTE|HUYEN)', label_upper):
            return "LOCATION_NOISE", original_label
        count_matches = sum(1 for kw in MedicalXRayPipeline.LOCATION_NOISE if kw in label_upper)
        if count_matches >= 1:
            return "LOCATION_NOISE", original_label

        # 3. Kiểm tra Tên người Việt
        for surname in MedicalXRayPipeline.VN_SURNAMES:
            if label_upper.startswith(surname):
                return "VIETNAMESE_NAMES", original_label

        return "OTHERS", original_label

    @staticmethod
    def normalize_text(text):
        """Chuẩn hóa text tiếng Việt không dấu / Normalize Vietnamese text"""
        if pd.isna(text): return ""
        text = str(text).upper()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.strip()

    @staticmethod
    def clean_result_text(text):
        """Làm sạch câu kết quả để dễ parse label / Clean result text"""
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def get_labels(text):
        """Tự động lấy nhãn bệnh lý từ kết quả / Auto extract labels from result"""
        if text == "":
            return ["UNKNOWN"]
        
        detected_labels = []
        for label, keywords in MedicalXRayPipeline.LABEL_MAPPING_V2.items():
            for kw in keywords:
                if kw in text:
                    detected_labels.append(label)
                    break
        
        if len(detected_labels) > 1 and "NORMAL" in detected_labels:
            detected_labels.remove("NORMAL")
            
        if not detected_labels:
            return ["OTHER"]
            
        return detected_labels


    # ==========================================
    # CORE PIPELINE STEPS / CÁC BƯỚC CHÍNH
    # ==========================================

    def step1_run_ocr(self):
        """
        BƯỚC 1: Quét OCR ảnh để lấy text / STEP 1: OCR Extraction
        Sử dụng docTr + Multiprocessing + GPU Batching.
        """
        self.logger.info("=== BƯỚC 1: BẮT ĐẦU TRÍCH XUẤT TEXT (OCR) ===")
        
        processed_files = set()
        if os.path.exists(self.ckpt_step1):
            df_existing = pd.read_csv(self.ckpt_step1)
            processed_files = set(df_existing["file_path"].tolist())
            self.logger.info(f"🔁 Chế độ Resume: Đã có {len(processed_files)} ảnh được xử lý.")

        image_paths = []
        for root, _, files in os.walk(self.root_img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    full_path = os.path.join(root, file)
                    # Chuyển về forward slash cho đồng bộ / Normalize slashes
                    full_path = full_path.replace('\\', '/') 
                    if full_path not in processed_files:
                        image_paths.append(full_path)

        self.logger.info(f"📂 Tìm thấy {len(image_paths)} ảnh mới cần xử lý.")
        if not image_paths:
            self.logger.info("✅ Không có ảnh mới. Bỏ qua OCR.")
            return pd.read_csv(self.ckpt_step1)

        # Load mô hình Doctr / Load Doctr Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"🧠 Đang sử dụng thiết bị: {device}")
        
        try:
            model = ocr_predictor(pretrained=True).to(device)
        except Exception as e:
            self.logger.error(f"❌ Lỗi load mô hình docTr: {e}")
            raise

        num_workers = min(4, cpu_count() * 2)
        results = []
        total_processed_count = 0

        # Processing loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in tqdm(range(0, len(image_paths), self.gpu_batch_size), desc="🚀 Processing Batches"):
                batch_paths = image_paths[i : i + self.gpu_batch_size]
                doc_data_batch = list(executor.map(self.load_doc_safe, batch_paths))
                
                valid_docs, valid_paths = [], []
                for path, doc in doc_data_batch:
                    if doc:
                        valid_docs.append(doc)
                        valid_paths.append(path)
                    else:
                        self.logger.warning(f"⚠️ Bỏ qua file lỗi khi load: {path}")

                if not valid_docs: continue

                for path, doc in zip(valid_paths, valid_docs):
                    try:
                        output = model(doc) 
                        text = self.extract_text_from_doctr(output)
                        results.append({"file_path": path, "extracted_text": text})
                        total_processed_count += 1
                    except Exception as e:
                        self.logger.error(f"💥 Lỗi khi infer ảnh {path}: {e}")
                        continue

                # Lưu checkpoint / Save checkpoint
                if len(results) >= self.ocr_checkpoint_every:
                    df_chunk = pd.DataFrame(results)
                    write_mode = "a" if os.path.exists(self.ckpt_step1) else "w"
                    header = not os.path.exists(self.ckpt_step1)
                    df_chunk.to_csv(self.ckpt_step1, mode=write_mode, index=False, header=header)
                    results = [] 

        # Lưu phần còn lại / Save remaining
        if results:
            df_chunk = pd.DataFrame(results)
            write_mode = "a" if os.path.exists(self.ckpt_step1) else "w"
            header = not os.path.exists(self.ckpt_step1)
            df_chunk.to_csv(self.ckpt_step1, mode=write_mode, index=False, header=header)

        self.logger.info(f"✅ Hoàn tất BƯỚC 1. Đã xử lý {total_processed_count} ảnh.")
        return pd.read_csv(self.ckpt_step1)


    def step2_clean_and_rescue_data(self, df_raw):
        """
        BƯỚC 2: Clean data & Cứu data lỗi OCR / STEP 2: Cleaning & Rescue
        Sử dụng Regex và EasyOCR cho những ca bị thiếu.
        """
        self.logger.info("=== BƯỚC 2: BẮT ĐẦU LÀM SẠCH VÀ CỨU DỮ LIỆU ===")
        
        if os.path.exists(self.ckpt_step2):
            self.logger.info(f"🔁 Load checkpoint Bước 2 từ file.")
            return pd.read_csv(self.ckpt_step2)

        results = []
        df_raw = df_raw.fillna("")

        self.logger.info("🧹 Đang phân tích cú pháp Regex (Lần 1)...")
        for i, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Regex Clean"):
            fp = row.get("file_path", "")
            text = str(row.get("extracted_text", ""))
            if not text:
                continue

            clean_txt = self.clean_noise(text)
            datetime_str = self.extract_datetime(clean_txt)
            name, birth = self.extract_name_birth(clean_txt)
            xray_type = self.extract_xray_type(clean_txt)

            results.append({
                "filepath": fp,
                "patient_name": name,
                "birth_year": birth,
                "xray_type": xray_type,
                "datetime": datetime_str
            })

        df_clean = pd.DataFrame(results)
        df_clean.replace("", np.nan, inplace=True)

        # -- VÒNG LẶP CỨU DỮ LIỆU (RESCUE LOOP) --
        # Xác định data thiếu / Identify missing
        mask_missing = df_clean['patient_name'].isnull() | df_clean['birth_year'].isnull() | df_clean['xray_type'].isnull()
        
        # Bỏ qua các nhãn không chuẩn / Filter invalid xray types
        valid_xray_regex = 'C-SPINE|CERVICAL|CHEST|THORAX|L-SPINE|T-SPINE|KUB|ABDOMEN|PELVIS|KNEE|SHOULDER|AP|LAT'
        df_clean.loc[mask_missing & ~df_clean['xray_type'].str.contains(valid_xray_regex, na=False, regex=True), 'xray_type'] = np.nan

        # Missing lần 2
        mask_missing = df_clean['patient_name'].isnull() | df_clean['birth_year'].isnull() | df_clean['xray_type'].isnull()
        df_missing = df_clean[mask_missing].copy()
        
        if len(df_missing) > 0:
            self.logger.info(f"🚑 Đang tiến hành vòng cứu hộ cho {len(df_missing)} ca bị thiếu bằng EasyOCR...")
            try:
                reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                
                for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing), desc="Rescue OCR"):
                    img_path = row['filepath']
                    try:
                        # Đảm bảo đường dẫn tuyệt đối đúng nếu cần / Ensure absolute path
                        full_img_path = img_path if os.path.isabs(img_path) else os.path.join(os.getcwd(), img_path)
                        img = cv2.imread(full_img_path)
                        if img is None: continue
                        
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Crop góc chứa thông tin / Crop header
                        h, w = gray.shape
                        crop_h = min(500, h)
                        crop_w = min(1000, w)
                        gray_crop = gray[0:crop_h, 0:crop_w]
                        
                        ocr_res = reader.readtext(gray_crop)
                        
                        # Heuristic đơn giản lấy dòng 2 và 3 (theo code gốc của người dùng)
                        # Trong thực tế có thể cần rule chắc chắn hơn, nhưng tôn trọng logic người dùng
                        if len(ocr_res) >= 3 and pd.isna(row['patient_name']):
                            df_missing.at[idx, 'patient_name'] = str(ocr_res[2][1])
                        if len(ocr_res) >= 4 and pd.isna(row['xray_type']):
                            df_missing.at[idx, 'xray_type'] = str(ocr_res[3][1])
                            
                    except Exception as e:
                        # Lỗi ảnh lẻ thì bỏ qua / Skip bad images
                        pass

                # Xóa rác ký tự / Clean characters for rescued names
                df_missing['patient_name'] = df_missing['patient_name'].astype(str).str.replace(r'[^a-zA-Z\s]', '', regex=True).str.upper()
                df_missing.replace("NAN", np.nan, inplace=True)
                df_missing.replace("", np.nan, inplace=True)
                
                # Combine lại vào df chính / Merge back
                df_clean.set_index('filepath', inplace=True)
                df_missing.set_index('filepath', inplace=True)
                df_clean = df_clean.combine_first(df_missing)
                df_clean.reset_index(inplace=True)
                
            except Exception as e:
                self.logger.error(f"❌ Lỗi vòng cứu hộ EasyOCR: {e}")

        # Chuẩn hóa cột XRAY TYPE và loại bỏ nhiễu / Normalize XRAY and remove noise
        self.logger.info("🏷️ Đang chuẩn hóa nhãn (Normalize)...")
        analysis_result = df_clean['xray_type'].apply(lambda x: self.normalize_and_classify(x))
        df_clean['Temp_Category'] = [res[0] for res in analysis_result]
        df_clean['Temp_Clean_Label'] = [res[1] for res in analysis_result]

        mask_valid = df_clean['Temp_Category'] == 'XRAY_LABELS'
        df_clean.loc[mask_valid, 'xray_type'] = df_clean.loc[mask_valid, 'Temp_Clean_Label']
        
        # Chỉ lấy những record hợp lệ hoặc có khả năng khớp được
        df_clean = df_clean[['filepath', 'patient_name', 'birth_year', 'xray_type', 'datetime']].copy()

        df_clean.to_csv(self.ckpt_step2, index=False)
        self.logger.info("✅ Hoàn tất BƯỚC 2. Đã lưu file làm sạch.")
        return df_clean


    def step3_match_hospital_records(self, df_xray):
        """
        BƯỚC 3: Matching với file bệnh viện / STEP 3: Hospital Records Matching
        """
        self.logger.info("=== BƯỚC 3: BẮT ĐẦU MATCHING KẾT QUẢ BỆNH VIỆN ===")
        
        if os.path.exists(self.ckpt_step3):
            self.logger.info(f"🔁 Load checkpoint Bước 3 từ file.")
            return pd.read_csv(self.ckpt_step3)

        try:
            df_ketqua = pd.read_csv(self.hospital_csv_path)
        except Exception as e:
            self.logger.error(f"❌ Không tìm thấy file {self.hospital_csv_path}. {e}")
            raise

        # Preprocessing cho Matching
        df_xray['norm_name'] = df_xray['patient_name'].apply(self.normalize_text)
        df_ketqua['norm_name'] = df_ketqua['Name'].apply(self.normalize_text)

        df_xray['datetime_obj'] = pd.to_datetime(df_xray['datetime'], errors='coerce')
        df_xray['date_only'] = df_xray['datetime_obj'].dt.date

        df_ketqua['date_start_obj'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y %H:%M', errors='coerce')
        df_ketqua['date_only'] = df_ketqua['date_start_obj'].dt.date

        # Hàm tính năm sinh từ cột Female/Male của file ketqua
        def extract_age_to_year(row):
            age_str = row['Female'] if pd.notna(row.get('Female')) else row.get('Male')
            if pd.isna(age_str): return None
            try:
                age = int(str(age_str).replace(' tuổi', '').strip())
                current_year = row['date_start_obj'].year if pd.notna(row['date_start_obj']) else datetime.now().year
                return current_year - age
            except:
                return None

        df_ketqua['calc_birth_year'] = df_ketqua.apply(extract_age_to_year, axis=1)

        # HÀM MATCHING THÔNG MINH
        def find_match(row_xray, df_lookup, mode="strict"):
            if pd.isna(row_xray['date_only']):
                return None, 0, "No Date"
            
            xray_date = row_xray['date_only']
            xray_name = str(row_xray['norm_name'])
            xray_year = row_xray['birth_year']

            # Lọc cửa sổ thời gian
            if mode == "strict":
                candidates = df_lookup[df_lookup['date_only'] == xray_date].copy()
                threshold = 0.6
            else: # rescue mode (+/- 3 days)
                start_w = xray_date - timedelta(days=3)
                end_w = xray_date + timedelta(days=3)
                candidates = df_lookup[(df_lookup['date_only'] >= start_w) & (df_lookup['date_only'] <= end_w)].copy()
                threshold = 0.65 # Khắt khe hơn với chuỗi tên / Stricter for name matching
            
            if candidates.empty:
                return None, 0, "No Candidates"

            best_score = 0
            best_idx = None

            for idx, cand in candidates.iterrows():
                cand_name = cand['norm_name']
                name_score = SequenceMatcher(None, xray_name, cand_name).ratio()
                
                # Boost năm sinh
                cand_year = cand['calc_birth_year']
                if pd.notna(xray_year) and pd.notna(cand_year):
                    try:
                        x_year = float(xray_year)
                        if abs(x_year - cand_year) <= 1:
                            name_score += 0.2
                        elif abs(x_year - cand_year) > 5:
                            name_score -= 0.3
                    except: pass

                if name_score > best_score:
                    best_score = name_score
                    best_idx = idx

            status = "Matched" if mode == "strict" else "Matched (Rescue +/-3d)"
            if best_score > threshold:
                return best_idx, best_score, status
            return None, best_score, "Low Score"

        self.logger.info("🔍 Đang chạy Matching Pass 1 (Strict)...")
        results = []
        for i, row in tqdm(df_xray.iterrows(), total=len(df_xray), desc="Strict Match"):
            match_idx, score, status = find_match(row, df_ketqua, mode="strict")
            res = {'xray_index': i, 'match_score': score, 'match_status': status}
            if match_idx is not None:
                match_data = df_ketqua.loc[match_idx].to_dict()
                for k, v in match_data.items(): res[f'ketqua_{k}'] = v
            results.append(res)
            
        df_results = pd.DataFrame(results)
        df_final = df_xray.join(df_results.set_index('xray_index'))

        self.logger.info("🚑 Đang chạy Matching Pass 2 (Rescue +/- 3 ngày)...")
        mask_miss = df_final['match_status'] != 'Matched'
        indices_to_rescue = df_final[mask_miss].index
        
        rescued_count = 0
        for i in tqdm(indices_to_rescue, desc="Rescue Match"):
            row = df_final.loc[i]
            match_idx, score, status = find_match(row, df_ketqua, mode="rescue")
            if match_idx is not None:
                df_final.at[i, 'match_score'] = score
                df_final.at[i, 'match_status'] = status
                match_data = df_ketqua.loc[match_idx]
                for col in df_ketqua.columns:
                    if col not in ['norm_name', 'date_start_obj', 'date_only', 'calc_birth_year']:
                        df_final.at[i, f"ketqua_{col}"] = match_data[col]
                rescued_count += 1

        # Chỉ giữ lại các ca đã Match
        df_matched_only = df_final[df_final['match_status'].str.contains('Matched', na=False)].copy()
        
        self.logger.info(f"✅ Hoàn tất BƯỚC 3. Cứu được {rescued_count} ca. Tổng số match: {len(df_matched_only)}")
        df_matched_only.to_csv(self.ckpt_step3, index=False)
        return df_matched_only


    def step4_auto_labeling(self, df_matched):
        """
        BƯỚC 4: Gán nhãn tự động / STEP 4: Auto Labeling
        """
        self.logger.info("=== BƯỚC 4: BẮT ĐẦU GÁN NHÃN TỰ ĐỘNG ===")
        
        if os.path.exists(self.ckpt_step4):
            self.logger.info(f"🔁 Load checkpoint Bước 4 từ file.")
            return pd.read_csv(self.ckpt_step4)

        df_matched['clean_result'] = df_matched['ketqua_Result'].apply(self.clean_result_text)
        df_matched['label'] = df_matched['clean_result'].apply(self.get_labels)
        
        # Convert list to string for saving
        df_matched['label_str'] = df_matched['label'].apply(json.dumps)

        df_matched.to_csv(self.ckpt_step4, index=False)
        self.logger.info(f"✅ Hoàn tất BƯỚC 4. Đã gán nhãn cho {len(df_matched)} bản ghi.")
        return df_matched


    def step5_feature_engineering_and_split(self, df_labeled):
        """
        BƯỚC 5: Trích xuất đặc trưng (Tuổi, Giới tính) và chia Train/Val/Test
        STEP 5: Feature Engineering & Dataset Split
        """
        self.logger.info("=== BƯỚC 5: BẮT ĐẦU TẠO FEATURE & SPLIT DATA ===")
        
        # Convert string label back to list
        df_labeled['label'] = df_labeled['label_str'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        def extract_demographics(row):
            age, gender = 0, 1 # Mặc định 1: Nữ / Default Female
            # Check Female
            if pd.notna(row.get('ketqua_Female')) and str(row.get('ketqua_Female')).strip() != "":
                gender = 1
                age_str = str(row['ketqua_Female'])
            # Check Male
            elif pd.notna(row.get('ketqua_Male')) and str(row.get('ketqua_Male')).strip() != "":
                gender = 0
                age_str = str(row['ketqua_Male'])
            else:
                curr_year = datetime.now().year
                by = row.get('birth_year', row.get('ketqua_calc_birth_year', curr_year))
                age = curr_year - by if pd.notna(by) else 0
                return pd.Series([age, 0])

            try:
                age = int(re.search(r'\d+', age_str).group())
            except:
                curr_year = datetime.now().year
                by = row.get('birth_year', row.get('ketqua_calc_birth_year', curr_year))
                age = curr_year - by if pd.notna(by) else 0
            return pd.Series([age, gender])

        df_labeled[['Age', 'Sex']] = df_labeled.apply(extract_demographics, axis=1)

        keep_cols = [
            'ketqua_ID', 'filepath', 'ketqua_Diagnose', 'xray_type', 
            'Age', 'Sex', 'clean_result', 'label'
        ]
        
        # Lấy các cột tồn tại / Get existing columns
        valid_cols = [c for c in keep_cols if c in df_labeled.columns]
        df_clean = df_labeled[valid_cols].copy()
        df_clean['filepath'] = df_clean['filepath'].astype(str).apply(lambda p: p.replace('\\', '/').replace('../', ''))

        # Gom nhóm theo ca khám (Group by Study ID)
        self.logger.info("🧬 Đang gom nhóm theo ketqua_ID (Study ID)...")
        df_grouped = df_clean.groupby('ketqua_ID').agg({
            'filepath': list,
            'ketqua_Diagnose': 'first',
            'xray_type': 'first',
            'Age': 'max',
            'Sex': 'first',
            'clean_result': 'first',
            'label': 'first'
        }).reset_index()

        df_grouped = df_grouped[df_grouped['filepath'].map(len) > 0]
        self.logger.info(f"Tổng số ca khám (Study) sau gom: {len(df_grouped)}")

        # Tạo Class Mapping
        all_labels = set()
        for labels in df_grouped['label']:
            if isinstance(labels, list): all_labels.update(labels)
        
        if 'UNKNOWN' in all_labels: all_labels.remove('UNKNOWN')
        label2id = {label: i for i, label in enumerate(sorted(list(all_labels)))}
        
        map_path = os.path.join(self.output_dir, 'class_mapping.json')
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(label2id, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Đã lưu class_mapping.json với {len(label2id)} classes.")

        # Stratified Split (70/10/20)
        stratify_col = df_grouped['label'].apply(lambda x: x[0] if len(x) > 0 and x[0] in label2id else "OTHER")
        
        # Split TrainVal / Test (80/20)
        train_val_df, test_df = train_test_split(df_grouped, test_size=0.2, random_state=42, stratify=stratify_col)
        
        # Split Train / Val (87.5% của 80% = 70% tổng, 12.5% của 80% = 10% tổng)
        stratify_col_remain = train_val_df['label'].apply(lambda x: x[0] if len(x) > 0 and x[0] in label2id else "OTHER")
        train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=stratify_col_remain)

        self.logger.info(f"✅ Hoàn tất BƯỚC 5. Split: Train({len(train_df)}) | Val({len(val_df)}) | Test({len(test_df)})")
        return train_df, val_df, test_df


    def step6_7_balance_and_resize(self, train_df, val_df, test_df):
        """
        BƯỚC 6 & 7: Cân bằng nhãn và Resize ảnh / STEP 6 & 7: Balance & Resize
        """
        self.logger.info("=== BƯỚC 6 & 7: BẮT ĐẦU CÂN BẰNG DATA VÀ RESIZE ẢNH SANG SSD ===")
        
        def process_split(df, split_name):
            # 1. Balancing (Lọc dữ liệu) / Filter Data
            self.logger.info(f"[{split_name.upper()}] Đang lọc dữ liệu...")
            df = df[~df['xray_type'].isin(self.DROP_XRAY_TYPES)].copy()
            df = df[(df['Age'] >= 7) & (df['Age'] <= 90)].copy()
            
            output_folder = os.path.join(self.ssd_resize_dir, split_name)
            new_filepaths = []
            
            self.logger.info(f"[{split_name.upper()}] Đang resize {len(df)} studies (Size: {self.img_size}x{self.img_size})...")
            
            # 2. Resizing & Moving
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Resize {split_name}"):
                original_paths = ast.literal_eval(row['filepath']) if isinstance(row['filepath'], str) else row['filepath']
                resized_paths = []
                study_id = str(row['ketqua_ID'])
                
                for i, img_path in enumerate(original_paths):
                    # Xây dựng absolute path dựa trên root_dir
                    full_src_path = img_path if os.path.isabs(img_path) else os.path.join(os.getcwd(), img_path)
                    
                    if os.path.exists(full_src_path):
                        fname = f"{study_id}_{i}.jpg"
                        save_path = os.path.join(output_folder, fname)
                        
                        if not os.path.exists(save_path): # Checkpoint resize
                            try:
                                img = cv2.imread(full_src_path)
                                if img is not None:
                                    img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                                    # Lưu nén JPG quality 90
                                    cv2.imwrite(save_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                    
                                    # Normalize đường dẫn lưu vào CSV
                                    save_path_norm = save_path.replace('\\', '/')
                                    resized_paths.append(save_path_norm)
                            except Exception as e:
                                self.logger.warning(f"Lỗi resize ảnh {full_src_path}: {e}")
                        else:
                            save_path_norm = save_path.replace('\\', '/')
                            resized_paths.append(save_path_norm)
                
                new_filepaths.append(resized_paths)
            
            # Cập nhật DataFrame
            df['filepath'] = new_filepaths
            # Bỏ các row bị lỗi mất sạch ảnh
            df = df[df['filepath'].map(len) > 0]
            
            final_csv_path = os.path.join(self.output_dir, f"final_dataset_{split_name}.csv")
            df.to_csv(final_csv_path, index=False)
            self.logger.info(f"[{split_name.upper()}] Đã lưu: {final_csv_path}")
            return df

        # Chạy cho cả 3 tập
        train_final = process_split(train_df, "train")
        val_final = process_split(val_df, "val")
        test_final = process_split(test_df, "test")

        self.logger.info("🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
        return train_final, val_final, test_final

    
    def run_pipeline(self):
        """Hàm chạy toàn bộ pipeline / Execute full pipeline"""
        self.logger.info("🚀 BẮT ĐẦU END-TO-END MEDICAL X-RAY PIPELINE")
        start_time = time.time()
        
        try:
            df_ocr = self.step1_run_ocr()
            df_cleaned = self.step2_clean_and_rescue_data(df_ocr)
            df_matched = self.step3_match_hospital_records(df_cleaned)
            df_labeled = self.step4_auto_labeling(df_matched)
            train_df, val_df, test_df = self.step5_feature_engineering_and_split(df_labeled)
            self.step6_7_balance_and_resize(train_df, val_df, test_df)
            
            end_time = time.time()
            self.logger.info(f"💯 PIPELINE SUCCESSFUL. Total time: {(end_time - start_time)/60:.2f} mins")
            
        except Exception as e:
            self.logger.error(f"❌ PIPELINE FAILED OR INTERRUPTED. Lỗi hệ thống: {e}")
            self.logger.error(traceback.format_exc())

# === END OF CLASS ===
