import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2

class DicomAnalyzer:
    """
    Công cụ phân tích file DICOM (Medical Imaging).
    Đọc metadata (thẻ DICOM) và trích xuất/chuẩn hóa hình ảnh (Pixel Array).
    """

    def __init__(self, dicom_path):
        """Khởi tạo với đường dẫn tới file DICOM."""
        self.dicom_path = dicom_path
        self.dataset = None
        self.metadata = {}

    def load(self):
        """Đọc file DICOM vào bộ nhớ."""
        try:
            self.dataset = pydicom.dcmread(self.dicom_path)
            return True
        except Exception as e:
            print(f"Lỗi khi đọc file DICOM {self.dicom_path}: {e}")
            return False

    def extract_metadata(self):
        """Trích xuất các thông tin quan trọng từ DICOM Tags."""
        if self.dataset is None:
            print("Vui lòng gọi hàm load() trước.")
            return {}

        def get_tag_value(tag_name, default="N/A"):
            """Hàm helper để lấy giá trị tag an toàn."""
            return getattr(self.dataset, tag_name, default)

        self.metadata = {
            # --- Thông tin bệnh nhân (Patient Info) ---
            "Patient Name": str(get_tag_value("PatientName")),
            "Patient ID": str(get_tag_value("PatientID")),
            "Patient Birth Date": str(get_tag_value("PatientBirthDate")),
            "Patient Sex": str(get_tag_value("PatientSex")),
            
            # --- Thông tin ca chụp (Study Info) ---
            "Study Date": str(get_tag_value("StudyDate")),
            "Study Time": str(get_tag_value("StudyTime")),
            "Body Part Examined": str(get_tag_value("BodyPartExamined")),
            "Modality": str(get_tag_value("Modality")),
            "Study Description": str(get_tag_value("StudyDescription")),
            
            # --- Thông tin hình ảnh (Image Meta) ---
            "Photometric Interpretation": str(get_tag_value("PhotometricInterpretation")),
            "Rows": get_tag_value("Rows"),
            "Columns": get_tag_value("Columns"),
            "Bits Allocated": get_tag_value("BitsAllocated"),
            "Bits Stored": get_tag_value("BitsStored"),
            "Pixel Representation": get_tag_value("PixelRepresentation"),
            "Window Center": get_tag_value("WindowCenter"),
            "Window Width": get_tag_value("WindowWidth"),
        }
        return self.metadata

    def get_pixel_array(self, apply_windowing=True):
        """
        Lấy mảng pixel (hình ảnh). 
        Hỗ trợ Windowing (VOI LUT) để chuyển đổi độ tương phản chuẩn y khoa.
        """
        if self.dataset is None:
            return None

        try:
            pixels = self.dataset.pixel_array
            
            # Nếu Photometric Interpretation là MONOCHROME1, ảnh sẽ bị âm bản (đen thành trắng)
            # Cần đảo ngược lại để hiển thị đúng (chuẩn là MONOCHROME2).
            is_inverted = False
            if self.dataset.PhotometricInterpretation == "MONOCHROME1":
                pixels = np.amax(pixels) - pixels
                is_inverted = True

            windowed_pixels = pixels
            
            # Áp dụng Windowing chuẩn từ máy chụp (VOI LUT)
            if apply_windowing:
                try:
                    # Thử áp dụng VOI LUT (Window Center/Width) có sẵn trong file
                    windowed_pixels = apply_voi_lut(pixels, self.dataset)
                except Exception as e:
                    print(f"Không thể áp dụng VOI LUT tự động: {e}. Đang dùng thuật toán thủ công...")
                    windowed_pixels = self._manual_windowing(pixels)
            
            # Chuẩn hóa về 0-255 (8-bit) để hiển thị mượt mà trên matplotlib
            windowed_pixels = windowed_pixels.astype(float)
            windowed_pixels -= np.min(windowed_pixels)
            if np.max(windowed_pixels) > 0:
                windowed_pixels /= np.max(windowed_pixels)
            windowed_pixels *= 255.0
            
            return windowed_pixels.astype(np.uint8), is_inverted

        except Exception as e:
            print(f"Lỗi trích xuất pixel: {e}")
            return None, False

    def _manual_windowing(self, img):
        """Áp dụng Windowing thủ công dựa trên metadata nếu hàm tự động thất bại."""
        window_center = self.metadata.get("Window Center")
        window_width = self.metadata.get("Window Width")
        
        # Nếu có nhiều giá trị, lấy giá trị đầu tiên
        if isinstance(window_center, pydicom.multival.MultiValue): window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue): window_width = window_width[0]
            
        if window_center == "N/A" or window_width == "N/A":
            return img # Trả về ảnh gốc nếu không có thẻ Windowing
            
        window_center, window_width = float(window_center), float(window_width)
        
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        
        windowed_img = np.clip(img, img_min, img_max)
        return windowed_img

    def print_all_tags(self):
        """In toàn bộ các thẻ (Tags) thô có trong file để tham khảo."""
        if self.dataset:
            # Chỉ in metadata, không in Pixel Data (vì quá dài)
            for elem in self.dataset:
                if elem.name != 'Pixel Data':
                    print(elem)
