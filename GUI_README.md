# RamanSPy GUI - Hướng dẫn sử dụng

Giao diện đồ họa đơn giản và dễ sử dụng cho phân tích phổ Raman với RamanSPy.

## Cài đặt

### 1. Cài đặt RamanSPy (nếu chưa có)
```bash
pip install ramanspy
```

### 2. Cài đặt Streamlit và các dependencies cho GUI
```bash
pip install -r requirements_gui.txt
```

Hoặc cài đặt trực tiếp:
```bash
pip install streamlit ramanspy numpy matplotlib scipy scikit-learn pandas
```

## Chạy ứng dụng

Từ thư mục chứa file `ramanspy_gui.py`, chạy lệnh:

```bash
streamlit run ramanspy_gui.py
```

Ứng dụng sẽ tự động mở trong trình duyệt web tại địa chỉ `http://localhost:8501`

## Các chức năng chính

### 1. 📂 Tải dữ liệu

Ba cách để tải dữ liệu:

#### **Tải từ file**
- Hỗ trợ nhiều định dạng: WITec, Renishaw, CSV/Text, NumPy
- Chọn đúng định dạng thiết bị đo
- Upload file và xem preview

#### **Dữ liệu mẫu**
- Tải dữ liệu mẫu từ RamanSPy datasets
- Bao gồm: THP-1 cells, MCF-7 cells
- Tự động tải về từ repository

#### **Dữ liệu tổng hợp**
- Tạo dữ liệu phổ giả lập để thử nghiệm
- Điều chỉnh số phổ, số điểm, mức nhiễu, số peaks
- Hữu ích cho testing và học tập

### 2. ⚙️ Tiền xử lý

Xây dựng pipeline tiền xử lý với các bước:

#### **Cropping (Cắt vùng phổ)**
- Chọn vùng wavenumber quan tâm
- Thường dùng: 700-1800 cm⁻¹ (fingerprint region)

#### **Despike (Loại bỏ Cosmic Ray)**
- WhitakerHayes: Phương pháp phổ biến
- Median: Phương pháp đơn giản

#### **Denoising (Khử nhiễu)**
- SavGol: Savitzky-Golay filter (điều chỉnh window length và polynomial order)
- Gaussian: Gaussian smoothing (điều chỉnh sigma)
- Wavelet: Wavelet denoising

#### **Baseline Correction (Hiệu chỉnh Baseline)**
- ASPLS: Adaptive Smoothness Penalized Least Squares
- ASLS: Asymmetric Least Squares
- Poly: Polynomial fitting (điều chỉnh polynomial order)

#### **Normalization (Chuẩn hóa)**
- MinMax: Chuẩn hóa 0-1
- AUC: Area Under Curve
- Vector: Unit vector normalization
- SNV: Standard Normal Variate

**Tính năng:**
- ✅ Chọn bật/tắt từng bước
- ✅ Điều chỉnh tham số cho mỗi phương pháp
- ✅ Áp dụng pipeline với một click
- ✅ So sánh trước/sau xử lý
- ✅ Reset pipeline khi cần

### 3. 🔬 Phân tích

Ba phương pháp phân tích:

#### **Spectral Unmixing (N-FINDR)**
- Phân tách phổ thành các thành phần endmember
- Tạo abundance maps cho mỗi endmember
- Điều chỉnh số endmembers (2-10)
- Hữu ích cho phân tích thành phần hóa học/sinh học

#### **Peak Detection**
- Tự động tìm peaks trong phổ
- Điều chỉnh prominence và distance
- Hiển thị vị trí và cường độ peaks
- Export danh sách peaks

#### **PCA (Principal Component Analysis)**
- Phân tích thành phần chính
- Giảm chiều dữ liệu
- Chọn số components (2-10)
- Xem explained variance, scores, loadings

### 4. 📊 Trực quan hóa

Hiển thị kết quả phân tích:

#### **Unmixing Results**
- Endmember spectra (stacked plot)
- Abundance maps cho mỗi endmember
- Hỗ trợ dữ liệu 2D và 3D (volumetric)
- Chọn layer để xem (cho dữ liệu 3D)

#### **Peak Detection Results**
- Plot phổ với peaks được đánh dấu
- Bảng thông tin chi tiết về peaks
- Vị trí (wavenumber) và cường độ

#### **PCA Results**
- Scree plot (explained variance)
- Score plot (PC1 vs PC2)
- Loading plots cho các PCs
- Trực quan hóa phân bố dữ liệu

## Tips sử dụng

### Workflow khuyến nghị:

1. **Tải dữ liệu** → Bắt đầu với dữ liệu tổng hợp nếu đang học
2. **Tiền xử lý** → Luôn tiền xử lý trước khi phân tích
3. **Phân tích** → Chọn phương pháp phù hợp với mục tiêu
4. **Trực quan hóa** → Xem và diễn giải kết quả

### Các bước tiền xử lý tiêu chuẩn:

```
1. Cropping (700-1800 cm⁻¹)
2. Despike (WhitakerHayes)
3. Denoising (SavGol: window=9, polyorder=3)
4. Baseline (ASPLS)
5. Normalization (MinMax)
```

### Xử lý lỗi phổ biến:

- **"Vui lòng tải dữ liệu trước"**: Quay lại tab "Tải dữ liệu"
- **Lỗi khi load file**: Kiểm tra định dạng file có đúng không
- **Lỗi khi phân tích**: Thử tiền xử lý dữ liệu trước

## Ví dụ sử dụng

### Ví dụ 1: Phân tích nhanh dữ liệu tổng hợp

1. Vào tab **"Tải dữ liệu"** → **"Dữ liệu tổng hợp"**
2. Tạo 100 phổ với 500 điểm, 3 peaks, nhiễu 0.1
3. Vào tab **"Tiền xử lý"**
4. Bật: Cropping (700-1800), Despike, SavGol, ASPLS, MinMax
5. Click **"Áp dụng Pipeline"**
6. Vào tab **"Phân tích"** → Chọn **"Spectral Unmixing"**
7. Đặt 3 endmembers → Click **"Chạy Unmixing"**
8. Vào tab **"Trực quan hóa"** để xem kết quả

### Ví dụ 2: Tìm peaks trong phổ

1. Tải dữ liệu của bạn
2. Tiền xử lý (khuyến nghị làm trước)
3. Vào tab **"Phân tích"** → **"Peak Detection"**
4. Điều chỉnh prominence = 0.1, distance = 20
5. Click **"Tìm Peaks"**
6. Xem kết quả với peaks được đánh dấu và bảng thông tin

### Ví dụ 3: PCA để giảm chiều

1. Tải dữ liệu đã tiền xử lý
2. Vào tab **"Phân tích"** → **"PCA"**
3. Chọn 3-5 components
4. Click **"Chạy PCA"**
5. Xem scree plot, score plot và loadings

## Giao diện

### Cấu trúc:
- **Sidebar**: Menu điều hướng và thông tin
- **Main area**: Nội dung chính của từng trang
- **Wide layout**: Tối ưu cho hiển thị plots và charts

### Navigation:
- Sử dụng radio buttons ở sidebar để chuyển trang
- Mỗi trang có chức năng riêng biệt
- Dữ liệu được lưu trong session state

## Troubleshooting

### GUI không mở được?
```bash
# Kiểm tra Streamlit đã cài chưa
streamlit --version

# Nếu chưa có, cài đặt
pip install streamlit
```

### Import error?
```bash
# Cài đặt tất cả dependencies
pip install -r requirements_gui.txt
```

### Dữ liệu mẫu không tải được?
- Dữ liệu mẫu cần internet để tải về
- Thử dùng dữ liệu tổng hợp để test
- Hoặc tải file dữ liệu của riêng bạn

## Tài liệu tham khảo

- **RamanSPy Documentation**: https://ramanspy.readthedocs.io
- **Streamlit Documentation**: https://docs.streamlit.io
- **Paper**: [Georgiev et al., Analytical Chemistry 2024](https://pubs.acs.org/doi/10.1021/acs.analchem.4c00383)

## Phát triển thêm

Các tính năng có thể thêm trong tương lai:
- Export kết quả ra file
- Batch processing nhiều files
- Thêm phương pháp phân tích khác
- Save/Load preprocessing pipelines
- Integrated report generation

## Liên hệ & Đóng góp

- **Issues**: [GitHub Issues](https://github.com/barahona-research-group/RamanSPy/issues)
- **Documentation**: [ReadTheDocs](https://ramanspy.readthedocs.io)

---

**RamanSPy GUI v1.0** - Được xây dựng với ❤️ sử dụng Streamlit
