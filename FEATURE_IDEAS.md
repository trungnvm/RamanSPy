# Feature Ideas cho RamanSPy GUI v2.0

## 1. Multi-File Management

### Yêu cầu:
- Tải nhiều file phổ Raman
- Đặt tên cho mỗi phổ
- Hiển thị danh sách phổ đã tải
- Chọn một hoặc nhiều phổ để phân tích
- Xóa phổ không cần
- PCA và các phương pháp khác có thể hoạt động với nhiều phổ

### Thiết kế:
```
Tab "Quản lý phổ":
├── Upload file
│   ├── Tên phổ: [input text]
│   ├── Choose file: [file uploader]
│   └── [Thêm vào collection]
├── Danh sách phổ
│   ├── ☑ File1.txt (1024 points)  [Preview] [Xóa]
│   ├── ☑ File2.txt (1024 points)  [Preview] [Xóa]
│   └── ☐ File3.txt (1024 points)  [Preview] [Xóa]
├── Chế độ phân tích
│   ├── ○ Phổ đơn (chọn 1)
│   └── ● Nhiều phổ (chọn nhiều)
└── [Sử dụng phổ đã chọn]
```

### Implementation:
- `st.session_state.spectra_collection` = [
    {
        'name': 'File1',
        'data': Spectrum object,
        'preprocessed': None,
        'selected': True
    },
    ...
]
- Khi chọn nhiều phổ → combine thành SpectralContainer
- PCA/Unmixing sẽ dùng SpectralContainer này

## 2. Despike Options

### Yêu cầu:
- Không fix cứng WhitakerHayes
- Cho phép điều chỉnh parameters

### Thiết kế:
```
Bước 2: Loại bỏ Cosmic Ray
☑ Sử dụng Despike
Phương pháp: [WhitakerHayes ▼]
├── Kernel size: [3]
└── Threshold: [8]
```

### Available methods in RamanSPy:
- WhitakerHayes (kernel_size, threshold)
- Có thể thêm: Median filter, Moving average

## 3. Export Results

### Yêu cầu:
- Export peaks ra CSV
- Export processed spectra
- Export PCA results
- Export plots as PNG

## 4. Batch Processing

### Yêu cầu:
- Áp dụng cùng pipeline cho tất cả phổ
- Export tất cả kết quả

## Priority:
1. **HIGH**: Multi-file management (để PCA hoạt động)
2. **HIGH**: Despike options
3. MEDIUM: Export results
4. LOW: Batch processing
