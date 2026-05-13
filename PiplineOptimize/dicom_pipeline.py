import os
import time
import logging
import json
import re
import unicodedata
import traceback
from datetime import timedelta, datetime
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MedicalDicomPipeline:
    """
    End-to-End Medical DICOM Pipeline.
    Xử lý trực tiếp file .dcm, bỏ qua hoàn toàn OCR.
    """

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

    DROP_XRAY_TYPES = ['PELVIS', 'ELBOW', 'ABDOMEN', 'HUMERUS', 'FEMUR']

    def __init__(self, root_dcm_dir, hospital_csv_path, output_dir, ssd_resize_dir, img_size=256):
        self.root_dcm_dir = root_dcm_dir
        self.hospital_csv_path = hospital_csv_path
        self.output_dir = output_dir
        self.ssd_resize_dir = ssd_resize_dir
        self.img_size = img_size
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ssd_resize_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ssd_resize_dir, 'images'), exist_ok=True) # Lưu chung ảnh vào 1 folder
        
        self.logger = self._setup_logger()
        
        self.ckpt_step1 = os.path.join(self.output_dir, "step1_dicom_metadata.csv")
        self.ckpt_step2 = os.path.join(self.output_dir, "step2_matched_dataset.csv")
        self.ckpt_step3 = os.path.join(self.output_dir, "step3_labeled_dataset.csv")

    def _setup_logger(self):
        logger = logging.getLogger("MedicalDicomPipeline")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        log_file = os.path.join(self.output_dir, f"dicom_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger

    # ==========================================
    # UTILITY FUNCTIONS
    # ==========================================
    @staticmethod
    def normalize_text(text):
        if pd.isna(text) or text is None: return ""
        text = str(text).upper()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        # Loại bỏ các số (nếu có tuổi dính vào tên) / Remove numbers from name
        text = re.sub(r'\d+', '', text)
        return text.strip()

    @staticmethod
    def parse_dicom_date(date_str):
        if not date_str or date_str == 'None': return ""
        try:
            return datetime.strptime(str(date_str), "%Y%m%d").strftime("%Y-%m-%d")
        except:
            return ""

    @staticmethod
    def extract_age_from_name(name_str):
        """Trích xuất tuổi nếu nó bị dính vào tên (VD: 'K HEU 63' -> 63)"""
        m = re.search(r'\d+', str(name_str))
        if m:
            return int(m.group())
        return None

    # ==========================================
    # PIPELINE STEPS
    # ==========================================
    def step1_extract_dicom_and_resize(self):
        """
        BƯỚC 1: Đọc DICOM -> Trích xuất Metadata -> Resize ảnh -> Lưu JPG.
        Gộp 3 bước (OCR, Clean, Resize) của quy trình cũ thành 1.
        """
        self.logger.info("=== BƯỚC 1: TRÍCH XUẤT DICOM & TIỀN XỬ LÝ ẢNH ===")
        if os.path.exists(self.ckpt_step1):
            self.logger.info("🔁 Load checkpoint Bước 1 từ file.")
            return pd.read_csv(self.ckpt_step1)

        dcm_files = []
        for root, _, files in os.walk(self.root_dcm_dir):
            for file in files:
                # Chỉ lấy file ảnh gốc DICOM (Original)
                if file.endswith('_O.dcm'):
                    dcm_files.append(os.path.join(root, file))

        self.logger.info(f"📂 Tìm thấy {len(dcm_files)} file _O.dcm.")
        results = []

        for dcm_path in tqdm(dcm_files, desc="DICOM Extraction"):
            try:
                ds = pydicom.dcmread(dcm_path)
                
                # 1. Trích xuất Metadata
                patient_name = str(getattr(ds, "PatientName", ""))
                study_date = getattr(ds, "StudyDate", "")
                body_part = str(getattr(ds, "BodyPartExamined", ""))
                patient_id = str(getattr(ds, "PatientID", ""))
                
                # Sửa lỗi nhập liệu bệnh viện (Tên dính tuổi)
                age_from_name = self.extract_age_from_name(patient_name)
                clean_name = self.normalize_text(patient_name)
                fmt_date = self.parse_dicom_date(study_date)

                # 2. Xử lý ảnh và Lưu xuống SSD
                try:
                    pixels = ds.pixel_array
                    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                        pixels = np.amax(pixels) - pixels
                        
                    # Áp dụng Windowing chuẩn y khoa
                    try:
                        windowed = apply_voi_lut(pixels, ds)
                    except:
                        windowed = pixels # Fallback

                    windowed = windowed.astype(float)
                    windowed -= np.min(windowed)
                    if np.max(windowed) > 0:
                        windowed /= np.max(windowed)
                    windowed *= 255.0
                    img_8bit = windowed.astype(np.uint8)

                    # Resize 
                    img_resized = cv2.resize(img_8bit, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                    
                    # Tạo tên file mới
                    save_name = f"{patient_id}_{os.path.basename(dcm_path).replace('.dcm', '.jpg')}"
                    save_path = os.path.join(self.ssd_resize_dir, 'images', save_name).replace('\\', '/')
                    
                    cv2.imwrite(save_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    results.append({
                        "filepath": save_path,
                        "dicom_id": patient_id,
                        "patient_name": clean_name,
                        "raw_name": patient_name,
                        "dicom_age": age_from_name,
                        "xray_type": body_part,
                        "datetime": fmt_date,
                        "date_only": fmt_date
                    })

                except Exception as img_e:
                    self.logger.warning(f"Lỗi ảnh {dcm_path}: {img_e}")
                    
            except Exception as e:
                self.logger.error(f"Lỗi đọc DICOM {dcm_path}: {e}")

        df_metadata = pd.DataFrame(results)
        df_metadata.to_csv(self.ckpt_step1, index=False)
        self.logger.info(f"✅ Hoàn tất BƯỚC 1. Đã trích xuất {len(df_metadata)} bản ghi.")
        return df_metadata

    def step2_match_hospital_records(self, df_dicom):
        """
        BƯỚC 2: Matching metadata của DICOM với ketqua.csv
        """
        self.logger.info("=== BƯỚC 2: MATCHING VỚI KẾT QUẢ BỆNH VIỆN ===")
        if os.path.exists(self.ckpt_step2):
            self.logger.info("🔁 Load checkpoint Bước 2 từ file.")
            return pd.read_csv(self.ckpt_step2)

        df_ketqua = pd.read_csv(self.hospital_csv_path)
        
        # Tiền xử lý ketqua
        df_ketqua['norm_name'] = df_ketqua['Name'].apply(self.normalize_text)
        df_ketqua['date_start_obj'] = pd.to_datetime(df_ketqua['Date start'], format='%d/%m/%Y', errors='coerce')
        df_ketqua['date_only'] = df_ketqua['date_start_obj'].dt.date.astype(str)
        
        # Nếu file DICOM ID khớp với Excel ID thì quá tuyệt vời
        df_ketqua['ketqua_ID_str'] = df_ketqua['ID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        results = []
        for i, row in tqdm(df_dicom.iterrows(), total=len(df_dicom), desc="Matching"):
            d_date = str(row['date_only'])
            d_name = str(row['patient_name'])
            d_id = str(row['dicom_id'])
            
            # Logic 1: Khớp bằng ID (Nhanh và chính xác nhất nếu BV nhập chuẩn)
            match_candidates = df_ketqua[df_ketqua['ketqua_ID_str'] == d_id]
            
            # Logic 2: Khớp bằng Date + Tên (Phòng hờ ID sai)
            if match_candidates.empty:
                candidates_by_date = df_ketqua[df_ketqua['date_only'] == d_date]
                best_score = 0
                best_idx = None
                for idx, cand in candidates_by_date.iterrows():
                    score = SequenceMatcher(None, d_name, str(cand['norm_name'])).ratio()
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_score > 0.65 and best_idx is not None:
                    match_candidates = df_ketqua.loc[[best_idx]]
                    
            res = row.to_dict()
            if not match_candidates.empty:
                res['match_status'] = 'Matched'
                match_data = match_candidates.iloc[0].to_dict()
                for k, v in match_data.items(): res[f'ketqua_{k}'] = v
            else:
                res['match_status'] = 'Unmatched'
                
            results.append(res)
            
        df_final = pd.DataFrame(results)
        df_matched = df_final[df_final['match_status'] == 'Matched'].copy()
        
        self.logger.info(f"✅ Hoàn tất BƯỚC 2. Match thành công {len(df_matched)}/{len(df_dicom)} ca.")
        df_matched.to_csv(self.ckpt_step2, index=False)
        return df_matched

    def step3_auto_labeling(self, df_matched):
        """
        BƯỚC 3: Gán nhãn tự động từ ketqua_Result
        """
        self.logger.info("=== BƯỚC 3: GÁN NHÃN TỰ ĐỘNG ===")
        if os.path.exists(self.ckpt_step3):
            return pd.read_csv(self.ckpt_step3)

        def clean_res(t):
            if pd.isna(t): return ""
            return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(t).lower())).strip()

        def get_lbls(t):
            if not t: return ["UNKNOWN"]
            lbls = []
            for lbl, kws in self.LABEL_MAPPING_V2.items():
                if any(kw in t for kw in kws): lbls.append(lbl)
            if len(lbls) > 1 and "NORMAL" in lbls: lbls.remove("NORMAL")
            return lbls if lbls else ["OTHER"]

        df_matched['clean_result'] = df_matched['ketqua_Result'].apply(clean_res)
        df_matched['label'] = df_matched['clean_result'].apply(get_lbls)
        df_matched['label_str'] = df_matched['label'].apply(json.dumps)
        
        df_matched.to_csv(self.ckpt_step3, index=False)
        self.logger.info("✅ Hoàn tất BƯỚC 3.")
        return df_matched

    def step4_feature_engineering_and_split(self, df_labeled):
        """
        BƯỚC 4: Feature Engineering & Split Train/Val/Test
        """
        self.logger.info("=== BƯỚC 4: FEATURE ENGINEERING & SPLITTING ===")
        
        df_labeled['label'] = df_labeled['label_str'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # Tính tuổi: Nếu dicom có sẵn tuổi thì dùng, không thì lấy từ Excel
        def get_age(row):
            if pd.notna(row.get('dicom_age')): return row['dicom_age']
            age_str = row.get('ketqua_Female', row.get('ketqua_Male', ''))
            try: return int(re.search(r'\d+', str(age_str)).group())
            except: return 0
            
        df_labeled['Age'] = df_labeled.apply(get_age, axis=1)
        df_labeled['Sex'] = df_labeled['ketqua_Female'].apply(lambda x: 1 if pd.notna(x) else 0)

        # Gom nhóm theo ca chụp
        df_grouped = df_labeled.groupby('ketqua_ID').agg({
            'filepath': list,
            'ketqua_Diagnose': 'first',
            'xray_type': 'first',
            'Age': 'max',
            'Sex': 'first',
            'clean_result': 'first',
            'label': 'first'
        }).reset_index()

        # Class Mapping
        all_lbls = set()
        for lbls in df_grouped['label']: all_lbls.update(lbls)
        if 'UNKNOWN' in all_lbls: all_lbls.remove('UNKNOWN')
        label2id = {lbl: i for i, lbl in enumerate(sorted(list(all_lbls)))}
        with open(os.path.join(self.output_dir, 'class_mapping.json'), 'w') as f:
            json.dump(label2id, f, indent=4)

        # Split
        strat = df_grouped['label'].apply(lambda x: x[0] if len(x)>0 and x[0] in label2id else "OTHER")
        tv_df, test_df = train_test_split(df_grouped, test_size=0.2, random_state=42, stratify=strat)
        strat_remain = tv_df['label'].apply(lambda x: x[0] if len(x)>0 and x[0] in label2id else "OTHER")
        train_df, val_df = train_test_split(tv_df, test_size=0.125, random_state=42, stratify=strat_remain)
        
        self.logger.info(f"✅ Hoàn tất BƯỚC 4. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def step5_balance_data(self, train_df, val_df, test_df):
        """
        BƯỚC 5: Lọc Data (Balancing) và Lưu CSV cuối cùng
        """
        self.logger.info("=== BƯỚC 5: CÂN BẰNG VÀ XUẤT DATA ===")
        
        def balance(df, name):
            # Lọc loại Xray và Tuổi
            df = df[~df['xray_type'].isin(self.DROP_XRAY_TYPES)]
            df = df[(df['Age'] >= 7) & (df['Age'] <= 90)]
            
            # Format list of strings proper for csv
            df['filepath'] = df['filepath'].apply(lambda x: str(x))
            out_file = os.path.join(self.output_dir, f"final_dicom_{name}.csv")
            df.to_csv(out_file, index=False)
            self.logger.info(f"[{name.upper()}] Đã lưu: {out_file} ({len(df)} records)")
            return df
            
        return balance(train_df, "train"), balance(val_df, "val"), balance(test_df, "test")

    def run_pipeline(self):
        self.logger.info("🚀 BẮT ĐẦU DICOM PIPELINE")
        t0 = time.time()
        try:
            df_dcm = self.step1_extract_dicom_and_resize()
            df_match = self.step2_match_hospital_records(df_dcm)
            df_lbl = self.step3_auto_labeling(df_match)
            tr, vl, ts = self.step4_feature_engineering_and_split(df_lbl)
            self.step5_balance_data(tr, vl, ts)
            self.logger.info(f"💯 PIPELINE THÀNH CÔNG. Thời gian: {(time.time()-t0)/60:.2f} phút.")
        except Exception as e:
            self.logger.error(f"❌ PIPELINE THẤT BẠI: {e}")
            self.logger.error(traceback.format_exc())
