# re_clean_raw_data.py
import re
import pandas as pd
from datetime import datetime

INPUT = "raw_data.csv"
OUTPUT = "cleaned_data_2.csv"

# regex mẫu
RE_DATE = re.compile(r'(\d{2}/\d{2}/\d{4}\s*\d{0,2}:?\d{0,2}:?\d{0,2})')
RE_NAME = re.compile(r'([A-ZÀ-Ỹ][A-ZÀ-Ỹ\s\.]+)\s+((19|20)\d{2})', re.UNICODE)
RE_XRAY = re.compile(r'([A-Z\-]+\s*(?:AP|LAT|PA|OBL|VIEW)?)', re.UNICODE)

# danh sách các cụm nhiễu thường gặp
NOISE_WORDS = [
    "TTYT", "HUYEN", "TRUNG", "TAM", "Y", "TE", "DUC", "TRONG",
    "CENTER", "BENH", "VIEN", "PKDK", "PHONG KHAM"
]
NOISE_PATTERN = re.compile(r'\b(?:' + '|'.join(NOISE_WORDS) + r')\b', re.IGNORECASE)


def clean_noise(text):
    """Xóa các cụm chữ nhiễu (TTYT-HUYEN DUC TRONG, ...) nhưng giữ nguyên phần còn lại"""
    text = re.sub(r'[-_]', ' ', text)
    text = re.sub(NOISE_PATTERN, '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def extract_datetime(text):
    m = RE_DATE.search(text)
    if m:
        s = m.group(1).strip()
        # normalize format
        try:
            if len(s.split()) == 2:
                return datetime.strptime(s, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            else:
                return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")
        except:
            pass
    return ""


def extract_name_birth(text):
    m = RE_NAME.search(text)
    if m:
        return m.group(1).strip(), m.group(2)
    return "", ""


def extract_xray_type(text):
    """
    Tìm loại ảnh X-quang dựa trên pattern in hoa như 'C-SPINE AP', 'CHEST PA', ...
    """
    candidates = RE_XRAY.findall(text)
    filtered = [c.strip() for c in candidates if len(c) >= 3 and not re.match(r'^(SERIES|IMAGE|UNIT|PIXEL|ADMIN|W|L)$', c)]
    if filtered:
        # chọn cái dài nhất
        return max(filtered, key=len)
    return ""


def main():
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    results = []

    for i, row in df.iterrows():
        fp = row.get("file_path", "")
        text = row.get("extracted_text", "")
        if not text:
            continue

        text = clean_noise(text)
        datetime_str = extract_datetime(text)
        name, birth = extract_name_birth(text)
        xray_type = extract_xray_type(text)

        results.append({
            "filepath": fp,
            "patient_name": name,
            "birth_year": birth,
            "xray_type": xray_type,
            "datetime": datetime_str
        })

        if i % 5000 == 0 and i > 0:
            print(f"Processed {i} rows...")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)
    print(f"✅ Done! Saved recovered file to: {OUTPUT}")


if __name__ == "__main__":
    main()
