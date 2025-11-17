"""
RamanSPy GUI - Giao diện đồ họa cho phân tích phổ Raman
Chạy ứng dụng: streamlit run ramanspy_gui.py
"""

import streamlit as st
import ramanspy as rp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

# Cấu hình trang
st.set_page_config(
    page_title="RamanSPy GUI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 RamanSPy - Phân tích phổ Raman</p>', unsafe_allow_html=True)

# Khởi tạo session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'pipeline_steps' not in st.session_state:
    st.session_state.pipeline_steps = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar - Navigation
st.sidebar.title("📋 Menu")
page = st.sidebar.radio(
    "Chọn chức năng:",
    ["Tải dữ liệu", "Tiền xử lý", "Phân tích", "Trực quan hóa"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**RamanSPy GUI v1.0**

Công cụ phân tích phổ Raman đơn giản và dễ sử dụng.

[Tài liệu](https://ramanspy.readthedocs.io)
""")

# ==================== TRANG TẢI DỮ LIỆU ====================
if page == "Tải dữ liệu":
    st.markdown('<p class="sub-header">📂 Tải dữ liệu phổ Raman</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 Tải từ file", "📊 Dữ liệu mẫu", "🎲 Dữ liệu tổng hợp"])

    # Tab 1: Tải từ file
    with tab1:
        st.write("### Tải dữ liệu từ file")

        col1, col2 = st.columns([2, 1])

        with col1:
            file_format = st.selectbox(
                "Chọn định dạng file:",
                ["CSV/Text (tùy chỉnh)", "WITec", "Renishaw", "NumPy (.npy)"]
            )

        with col2:
            st.info("💡 Chọn đúng định dạng của thiết bị đo")

        # Hướng dẫn cho CSV/Text
        if file_format == "CSV/Text (tùy chỉnh)":
            with st.expander("ℹ️ Định dạng CSV/Text được hỗ trợ"):
                st.write("""
                **Định dạng file được hỗ trợ:**
                - Header (tùy chọn) với metadata
                - Dữ liệu 2 cột: Wavenumber và Intensity
                - Phân cách bằng: `;` , `,` , tab hoặc khoảng trắng

                **Ví dụ:**
                ```
                Name=Andor Spectra
                X=Raman Shift, 1/cm
                Y=Intensity, Counts
                2.37; 2405
                6.04; 2446
                9.70; 2369
                ...
                ```
                """)

        uploaded_file = st.file_uploader(
            "Chọn file dữ liệu:",
            type=['txt', 'csv', 'wdf', 'npy', 'npz', 'dat'],
            help="Hỗ trợ nhiều định dạng file từ các thiết bị Raman khác nhau"
        )

        if uploaded_file is not None:
            try:
                # Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                with st.spinner("Đang tải dữ liệu..."):
                    # Load dữ liệu theo định dạng
                    if file_format == "WITec":
                        st.session_state.data = rp.load.witec(tmp_path)
                    elif file_format == "Renishaw":
                        st.session_state.data = rp.load.renishaw(tmp_path)
                    elif file_format == "NumPy (.npy)":
                        data_array = np.load(tmp_path)
                        st.session_state.data = rp.Spectrum(data_array)
                    else:
                        # CSV/Text (tùy chỉnh)
                        # Đọc file và xử lý
                        with open(tmp_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # Tìm dòng bắt đầu dữ liệu (bỏ qua header)
                        data_start = 0
                        for i, line in enumerate(lines):
                            # Kiểm tra nếu dòng chứa số (dữ liệu)
                            if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-'):
                                data_start = i
                                break

                        # Parse dữ liệu
                        wavenumbers = []
                        intensities = []

                        for line in lines[data_start:]:
                            line = line.strip()
                            if not line:
                                continue

                            # Thử các delimiter khác nhau
                            if ';' in line:
                                parts = line.split(';')
                            elif ',' in line:
                                parts = line.split(',')
                            elif '\t' in line:
                                parts = line.split('\t')
                            else:
                                parts = line.split()

                            if len(parts) >= 2:
                                try:
                                    wavenumbers.append(float(parts[0].strip()))
                                    intensities.append(float(parts[1].strip()))
                                except ValueError:
                                    continue

                        # Tạo Spectrum object
                        if len(wavenumbers) > 0 and len(intensities) > 0:
                            st.session_state.data = rp.Spectrum(
                                np.array(intensities),
                                spectral_axis=np.array(wavenumbers)
                            )
                        else:
                            raise ValueError("Không thể đọc dữ liệu từ file. Vui lòng kiểm tra định dạng.")

                os.unlink(tmp_path)
                st.success(f"✅ Đã tải thành công: {uploaded_file.name}")

                # Hiển thị thông tin dữ liệu
                st.write("### Thông tin dữ liệu")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Kiểu dữ liệu", type(st.session_state.data).__name__)
                with col2:
                    st.metric("Kích thước", str(st.session_state.data.shape))
                with col3:
                    if hasattr(st.session_state.data, 'spectral_axis'):
                        st.metric("Số điểm phổ", len(st.session_state.data.spectral_axis))

            except Exception as e:
                st.error(f"❌ Lỗi khi tải file: {str(e)}")

    # Tab 2: Dữ liệu mẫu
    with tab2:
        st.write("### Tải dữ liệu mẫu từ RamanSPy")

        st.info("📥 Dữ liệu sẽ được tải xuống tự động từ repository")

        dataset_type = st.selectbox(
            "Chọn loại dữ liệu:",
            ["THP-1 cells", "MCF-7 cells", "Bacteria dataset"]
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("📥 Tải dữ liệu mẫu", type="primary"):
                try:
                    with st.spinner("Đang tải dữ liệu mẫu..."):
                        data_dir = "./data/kallepitis_data"

                        # Chọn cell type
                        if dataset_type == "THP-1 cells":
                            cell_type = 'THP-1'
                        elif dataset_type == "MCF-7 cells":
                            cell_type = 'MCF-7'
                        else:
                            cell_type = 'THP-1'

                        volumes = rp.datasets.volumetric_cells(cell_type=cell_type, folder=data_dir)
                        st.session_state.data = volumes[0]

                        st.success(f"✅ Đã tải dữ liệu mẫu: {dataset_type}")
                        st.rerun()

                except Exception as e:
                    st.warning(f"⚠️ Không thể tải dữ liệu mẫu: {str(e)}")
                    st.info("Dữ liệu mẫu cần được tải về trước. Bạn có thể sử dụng dữ liệu tổng hợp để thử nghiệm.")

        with col2:
            if dataset_type == "THP-1 cells":
                st.write("**THP-1 cells**: Dữ liệu phổ Raman 3D của tế bào THP-1")
            elif dataset_type == "MCF-7 cells":
                st.write("**MCF-7 cells**: Dữ liệu phổ Raman 3D của tế bào MCF-7")

    # Tab 3: Dữ liệu tổng hợp
    with tab3:
        st.write("### Tạo dữ liệu tổng hợp để thử nghiệm")

        col1, col2 = st.columns(2)

        with col1:
            n_spectra = st.slider("Số phổ:", 10, 1000, 100)
            n_points = st.slider("Số điểm phổ:", 100, 2000, 500)

        with col2:
            noise_level = st.slider("Mức nhiễu:", 0.0, 0.5, 0.1, 0.05)
            n_peaks = st.slider("Số peak:", 1, 10, 3)

        if st.button("🎲 Tạo dữ liệu tổng hợp", type="primary"):
            with st.spinner("Đang tạo dữ liệu..."):
                # Tạo trục wavenumber
                wavenumbers = np.linspace(400, 2000, n_points)

                # Tạo phổ với nhiều peaks
                spectra_list = []
                for _ in range(n_spectra):
                    spectrum = np.zeros(n_points)

                    # Thêm peaks ngẫu nhiên
                    for _ in range(n_peaks):
                        center = np.random.uniform(600, 1800)
                        width = np.random.uniform(20, 80)
                        height = np.random.uniform(0.5, 1.0)

                        # Gaussian peak
                        spectrum += height * np.exp(-((wavenumbers - center) ** 2) / (2 * width ** 2))

                    # Thêm baseline
                    baseline = np.random.uniform(0.1, 0.3) * np.ones(n_points)
                    spectrum += baseline

                    # Thêm nhiễu
                    noise = np.random.normal(0, noise_level, n_points)
                    spectrum += noise

                    spectra_list.append(spectrum)

                # Tạo SpectralContainer
                data_array = np.array(spectra_list)
                st.session_state.data = rp.SpectralContainer(data_array, spectral_axis=wavenumbers)

                st.success(f"✅ Đã tạo {n_spectra} phổ tổng hợp với {n_points} điểm")

    # Preview dữ liệu nếu đã load
    if st.session_state.data is not None:
        st.markdown("---")
        st.write("### 👀 Preview dữ liệu")

        try:
            # Lấy một vài phổ để hiển thị
            data_type = type(st.session_state.data).__name__

            if data_type == 'Spectrum':
                # Spectrum đơn lẻ
                sample_spectra = st.session_state.data
            elif hasattr(st.session_state.data, 'flat'):
                # Volumetric data
                sample_spectra = st.session_state.data.flat[0:5]
            elif hasattr(st.session_state.data, '__len__') and len(st.session_state.data.shape) > 1:
                sample_spectra = st.session_state.data[0:5]
            else:
                sample_spectra = st.session_state.data

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            rp.plot.spectra(sample_spectra, ax=ax, plot_type='single')
            ax.set_title("Preview phổ Raman")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Không thể hiển thị preview: {str(e)}")

# ==================== TRANG TIỀN XỬ LÝ ====================
elif page == "Tiền xử lý":
    st.markdown('<p class="sub-header">⚙️ Tiền xử lý dữ liệu</p>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu trước!")
        st.stop()

    st.write("### Xây dựng Pipeline tiền xử lý")

    # Sidebar cho việc chọn các bước preprocessing
    st.write("#### Chọn các bước tiền xử lý:")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Bước 1: Cắt vùng phổ (Cropping)**")
        use_cropping = st.checkbox("Sử dụng Cropping", value=True)
        if use_cropping:
            crop_min = st.number_input("Wavenumber min (cm⁻¹):", 400, 2000, 700, 50)
            crop_max = st.number_input("Wavenumber max (cm⁻¹):", 400, 2000, 1800, 50)

    with col2:
        st.write("**Bước 2: Loại bỏ Cosmic Ray**")
        use_despike = st.checkbox("Sử dụng Despike", value=True)
        if use_despike:
            st.info("💡 Sử dụng phương pháp WhitakerHayes")

    col3, col4 = st.columns([1, 1])

    with col3:
        st.write("**Bước 3: Khử nhiễu (Denoising)**")
        use_denoise = st.checkbox("Sử dụng Denoising", value=True)
        if use_denoise:
            denoise_method = st.selectbox(
                "Phương pháp khử nhiễu:",
                ["SavGol", "Gaussian", "Wavelet"]
            )

            if denoise_method == "SavGol":
                window_length = st.slider("Window length:", 3, 21, 9, 2)
                polyorder = st.slider("Polynomial order:", 1, 5, 3)
            elif denoise_method == "Gaussian":
                sigma = st.slider("Sigma:", 0.5, 5.0, 1.0, 0.5)

    with col4:
        st.write("**Bước 4: Hiệu chỉnh Baseline**")
        use_baseline = st.checkbox("Sử dụng Baseline Correction", value=True)
        if use_baseline:
            baseline_method = st.selectbox(
                "Phương pháp baseline:",
                ["ASPLS", "ASLS", "Poly"]
            )

            if baseline_method == "Poly":
                poly_order = st.slider("Polynomial order:", 1, 5, 3)

    col5, col6 = st.columns([1, 1])

    with col5:
        st.write("**Bước 5: Chuẩn hóa (Normalization)**")
        use_normalize = st.checkbox("Sử dụng Normalization", value=True)
        if use_normalize:
            normalize_method = st.selectbox(
                "Phương pháp chuẩn hóa:",
                ["MinMax", "AUC", "Vector", "SNV"]
            )

    # Nút áp dụng pipeline
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        if st.button("▶️ Áp dụng Pipeline", type="primary", use_container_width=True):
            try:
                with st.spinner("Đang xử lý..."):
                    # Xây dựng pipeline
                    steps = []

                    if use_cropping:
                        steps.append(rp.preprocessing.misc.Cropper(region=(crop_min, crop_max)))

                    if use_despike:
                        steps.append(rp.preprocessing.despike.WhitakerHayes())

                    if use_denoise:
                        if denoise_method == "SavGol":
                            steps.append(rp.preprocessing.denoise.SavGol(window_length=window_length, polyorder=polyorder))
                        elif denoise_method == "Gaussian":
                            steps.append(rp.preprocessing.denoise.Gaussian(sigma=sigma))
                        else:
                            steps.append(rp.preprocessing.denoise.Wavelet())

                    if use_baseline:
                        if baseline_method == "ASPLS":
                            steps.append(rp.preprocessing.baseline.ASPLS())
                        elif baseline_method == "ASLS":
                            steps.append(rp.preprocessing.baseline.ASLS())
                        else:
                            steps.append(rp.preprocessing.baseline.Poly(poly_order=poly_order))

                    if use_normalize:
                        if normalize_method == "MinMax":
                            steps.append(rp.preprocessing.normalise.MinMax())
                        elif normalize_method == "AUC":
                            steps.append(rp.preprocessing.normalise.AUC())
                        elif normalize_method == "Vector":
                            steps.append(rp.preprocessing.normalise.Vector())
                        else:
                            steps.append(rp.preprocessing.normalise.SNV())

                    # Tạo và áp dụng pipeline
                    pipeline = rp.preprocessing.Pipeline(steps)
                    st.session_state.preprocessed_data = pipeline.apply(st.session_state.data)
                    st.session_state.pipeline_steps = steps

                    st.success(f"✅ Đã áp dụng {len(steps)} bước tiền xử lý!")

            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý: {str(e)}")

    with col_btn2:
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            st.session_state.preprocessed_data = None
            st.session_state.pipeline_steps = []
            st.info("Đã reset pipeline")

    # So sánh trước/sau
    if st.session_state.preprocessed_data is not None:
        st.markdown("---")
        st.write("### 📊 So sánh trước và sau xử lý")

        col_before, col_after = st.columns(2)

        try:
            # Lấy phổ mẫu
            data_type = type(st.session_state.data).__name__

            if data_type == 'Spectrum':
                # Spectrum đơn lẻ
                raw_spectrum = st.session_state.data
                processed_spectrum = st.session_state.preprocessed_data
            elif hasattr(st.session_state.data, 'flat'):
                # Volumetric data
                raw_spectrum = st.session_state.data.flat[0]
                processed_spectrum = st.session_state.preprocessed_data.flat[0]
            elif hasattr(st.session_state.data, '__len__') and len(st.session_state.data.shape) > 1:
                # Multi-spectrum data
                raw_spectrum = st.session_state.data[0]
                processed_spectrum = st.session_state.preprocessed_data[0]
            else:
                raw_spectrum = st.session_state.data
                processed_spectrum = st.session_state.preprocessed_data

            with col_before:
                st.write("**Trước xử lý**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                rp.plot.spectra(raw_spectrum, ax=ax1)
                ax1.set_title("Phổ gốc")
                st.pyplot(fig1)
                plt.close()

            with col_after:
                st.write("**Sau xử lý**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                rp.plot.spectra(processed_spectrum, ax=ax2)
                ax2.set_title("Phổ đã xử lý")
                st.pyplot(fig2)
                plt.close()

        except Exception as e:
            st.error(f"Lỗi khi hiển thị so sánh: {str(e)}")

# ==================== TRANG PHÂN TÍCH ====================
elif page == "Phân tích":
    st.markdown('<p class="sub-header">🔬 Phân tích phổ</p>', unsafe_allow_html=True)

    # Kiểm tra dữ liệu
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu trước!")
        st.stop()

    # Sử dụng dữ liệu đã xử lý nếu có, không thì dùng dữ liệu gốc
    data_to_analyze = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.data

    if st.session_state.preprocessed_data is None:
        st.info("💡 Đang sử dụng dữ liệu gốc. Khuyến nghị tiền xử lý dữ liệu trước khi phân tích.")

    st.write("### Chọn phương pháp phân tích")

    analysis_method = st.selectbox(
        "Phương pháp:",
        ["Spectral Unmixing (N-FINDR)", "Peak Detection", "Component Analysis (PCA)"]
    )

    # Spectral Unmixing
    if analysis_method == "Spectral Unmixing (N-FINDR)":
        st.write("#### Spectral Unmixing - N-FINDR")
        st.write("Phân tách phổ thành các thành phần endmember và bản đồ phong phú")

        col1, col2 = st.columns([2, 1])

        with col1:
            n_endmembers = st.slider("Số endmembers:", 2, 10, 5)

        with col2:
            st.info("💡 Số endmembers là số thành phần cơ bản trong mẫu")

        if st.button("▶️ Chạy Unmixing", type="primary"):
            try:
                with st.spinner("Đang phân tích..."):
                    unmixer = rp.analysis.unmix.NFINDR(n_endmembers=n_endmembers)
                    abundance_maps, endmembers = unmixer.apply(data_to_analyze)

                    st.session_state.analysis_results = {
                        'type': 'unmixing',
                        'abundance_maps': abundance_maps,
                        'endmembers': endmembers,
                        'data': data_to_analyze
                    }

                    st.success(f"✅ Đã phân tách thành {n_endmembers} endmembers!")

            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {str(e)}")

    # Peak Detection
    elif analysis_method == "Peak Detection":
        st.write("#### Peak Detection")
        st.write("Tìm các peak trong phổ")

        col1, col2 = st.columns(2)

        with col1:
            prominence = st.slider("Prominence:", 0.01, 1.0, 0.1, 0.01)

        with col2:
            distance = st.slider("Distance (số điểm):", 5, 100, 20)

        if st.button("▶️ Tìm Peaks", type="primary"):
            try:
                from scipy.signal import find_peaks

                with st.spinner("Đang tìm peaks..."):
                    # Lấy phổ đầu tiên để phân tích
                    if hasattr(data_to_analyze, 'flat'):
                        spectrum = data_to_analyze.flat[0]
                    elif hasattr(data_to_analyze, '__getitem__'):
                        spectrum = data_to_analyze[0]
                    else:
                        spectrum = data_to_analyze

                    # Tìm peaks
                    intensities = spectrum.spectral_data if hasattr(spectrum, 'spectral_data') else spectrum
                    peaks, properties = find_peaks(intensities, prominence=prominence, distance=distance)

                    st.session_state.analysis_results = {
                        'type': 'peaks',
                        'spectrum': spectrum,
                        'peaks': peaks,
                        'properties': properties
                    }

                    st.success(f"✅ Đã tìm thấy {len(peaks)} peaks!")

            except Exception as e:
                st.error(f"❌ Lỗi khi tìm peaks: {str(e)}")

    # PCA
    elif analysis_method == "Component Analysis (PCA)":
        st.write("#### Principal Component Analysis (PCA)")
        st.write("Phân tích thành phần chính")

        n_components = st.slider("Số components:", 2, 10, 3)

        if st.button("▶️ Chạy PCA", type="primary"):
            try:
                from sklearn.decomposition import PCA

                with st.spinner("Đang phân tích PCA..."):
                    # Chuẩn bị dữ liệu
                    if hasattr(data_to_analyze, 'flat'):
                        data_matrix = data_to_analyze.flat.spectral_data
                    else:
                        data_matrix = data_to_analyze.spectral_data if hasattr(data_to_analyze, 'spectral_data') else data_to_analyze

                    # Reshape nếu cần
                    if len(data_matrix.shape) > 2:
                        original_shape = data_matrix.shape
                        data_matrix = data_matrix.reshape(-1, data_matrix.shape[-1])

                    # Chạy PCA
                    pca = PCA(n_components=n_components)
                    scores = pca.fit_transform(data_matrix)
                    loadings = pca.components_

                    st.session_state.analysis_results = {
                        'type': 'pca',
                        'scores': scores,
                        'loadings': loadings,
                        'explained_variance': pca.explained_variance_ratio_,
                        'data': data_to_analyze
                    }

                    st.success(f"✅ Đã hoàn thành PCA với {n_components} components!")

            except Exception as e:
                st.error(f"❌ Lỗi khi chạy PCA: {str(e)}")

# ==================== TRANG TRỰC QUAN HÓA ====================
elif page == "Trực quan hóa":
    st.markdown('<p class="sub-header">📊 Trực quan hóa kết quả</p>', unsafe_allow_html=True)

    if st.session_state.analysis_results is None:
        st.warning("⚠️ Vui lòng chạy phân tích trước!")
        st.stop()

    results = st.session_state.analysis_results
    result_type = results['type']

    # Hiển thị theo loại phân tích
    if result_type == 'unmixing':
        st.write("### Kết quả Spectral Unmixing")

        endmembers = results['endmembers']
        abundance_maps = results['abundance_maps']
        data = results['data']

        # Plot endmembers
        st.write("#### 🔬 Endmembers")
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        if hasattr(data, 'spectral_axis'):
            rp.plot.spectra(endmembers, wavenumber_axis=data.spectral_axis, ax=ax1, plot_type='single stacked')
        else:
            rp.plot.spectra(endmembers, ax=ax1, plot_type='single stacked')

        ax1.set_title("Endmember Spectra")
        st.pyplot(fig1)
        plt.close()

        # Plot abundance maps
        st.write("#### 🗺️ Abundance Maps")

        try:
            # Nếu là volumetric data, lấy một layer
            if len(abundance_maps[0].shape) == 3:
                layer_idx = st.slider("Chọn layer:", 0, abundance_maps[0].shape[2]-1, abundance_maps[0].shape[2]//2)

                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4))
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap[:, :, layer_idx], cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                st.pyplot(fig2)
                plt.close()

            else:
                # 2D data
                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4))
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap, cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                st.pyplot(fig2)
                plt.close()

        except Exception as e:
            st.warning(f"Không thể hiển thị abundance maps: {str(e)}")

    elif result_type == 'peaks':
        st.write("### Kết quả Peak Detection")

        spectrum = results['spectrum']
        peaks = results['peaks']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot spectrum
        if hasattr(spectrum, 'spectral_axis'):
            x_axis = spectrum.spectral_axis
            y_data = spectrum.spectral_data
        else:
            x_axis = np.arange(len(spectrum))
            y_data = spectrum

        ax.plot(x_axis, y_data, 'b-', linewidth=1.5, label='Spectrum')
        ax.plot(x_axis[peaks], y_data[peaks], 'ro', markersize=8, label='Peaks')

        # Đánh dấu peaks
        for peak in peaks:
            ax.axvline(x_axis[peak], color='r', linestyle='--', alpha=0.3)
            ax.text(x_axis[peak], y_data[peak], f'{x_axis[peak]:.0f}',
                   rotation=45, ha='right', va='bottom', fontsize=8)

        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Peak Detection - {len(peaks)} peaks found')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

        # Bảng thông tin peaks
        st.write("#### 📋 Danh sách Peaks")
        peak_data = {
            'Peak #': range(1, len(peaks)+1),
            'Position (index)': peaks,
            'Wavenumber (cm⁻¹)': [f"{x_axis[p]:.2f}" for p in peaks],
            'Intensity': [f"{y_data[p]:.4f}" for p in peaks]
        }

        import pandas as pd
        df = pd.DataFrame(peak_data)
        st.dataframe(df, use_container_width=True)

    elif result_type == 'pca':
        st.write("### Kết quả PCA")

        scores = results['scores']
        loadings = results['loadings']
        explained_variance = results['explained_variance']

        col1, col2 = st.columns(2)

        with col1:
            # Scree plot
            st.write("#### 📊 Explained Variance")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.bar(range(1, len(explained_variance)+1), explained_variance * 100)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot')
            st.pyplot(fig1)
            plt.close()

        with col2:
            # Score plot
            st.write("#### 🎯 Score Plot (PC1 vs PC2)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(scores[:, 0], scores[:, 1], alpha=0.6)
            ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
            ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
            ax2.set_title('PCA Score Plot')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()

        # Loading plot
        st.write("#### 📈 Loading Plots")
        fig3, axes = plt.subplots(1, min(3, len(loadings)), figsize=(12, 4))
        if len(loadings) == 1:
            axes = [axes]

        for i, ax in enumerate(axes[:len(loadings)]):
            ax.plot(loadings[i])
            ax.set_title(f'PC{i+1} Loading')
            ax.set_xlabel('Wavenumber index')
            ax.set_ylabel('Loading')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>RamanSPy GUI v1.0 | Được xây dựng với Streamlit</p>
    <p>Tài liệu: <a href='https://ramanspy.readthedocs.io'>ramanspy.readthedocs.io</a></p>
</div>
""", unsafe_allow_html=True)
