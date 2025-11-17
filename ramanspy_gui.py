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
if 'spectra_collection' not in st.session_state:
    st.session_state.spectra_collection = []  # List of {'name': str, 'data': Spectrum, 'preprocessed': None}

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

        uploaded_files = st.file_uploader(
            "Chọn file dữ liệu:",
            type=['txt', 'csv', 'wdf', 'npy', 'npz', 'dat'],
            help="Hỗ trợ nhiều định dạng file từ các thiết bị Raman khác nhau",
            accept_multiple_files=True
        )

        # Checkbox để tự động thêm vào collection
        auto_add_to_collection = st.checkbox(
            "Tự động thêm vào Collection sau khi tải",
            value=True,
            help="Tự động thêm các file đã tải vào collection để dễ quản lý"
        )

        if uploaded_files:
            loaded_count = 0
            failed_files = []

            for uploaded_file in uploaded_files:
                try:
                    # Lưu file tạm
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    with st.spinner(f"Đang tải {uploaded_file.name}..."):
                        # Load dữ liệu theo định dạng
                        if file_format == "WITec":
                            loaded_data = rp.load.witec(tmp_path)
                        elif file_format == "Renishaw":
                            loaded_data = rp.load.renishaw(tmp_path)
                        elif file_format == "NumPy (.npy)":
                            data_array = np.load(tmp_path)
                            loaded_data = rp.Spectrum(data_array)
                        else:
                            # CSV/Text (tùy chỉnh)
                            with open(tmp_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()

                            # Tìm dòng bắt đầu dữ liệu
                            data_start = 0
                            for i, line in enumerate(lines):
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
                                loaded_data = rp.Spectrum(
                                    np.array(intensities),
                                    spectral_axis=np.array(wavenumbers)
                                )
                            else:
                                raise ValueError("Không thể đọc dữ liệu từ file.")

                    os.unlink(tmp_path)

                    # Lưu vào st.session_state.data (file cuối cùng)
                    st.session_state.data = loaded_data

                    # Tự động thêm vào collection nếu được chọn
                    if auto_add_to_collection:
                        # Lấy tên file (không có extension)
                        file_base_name = Path(uploaded_file.name).stem

                        # Kiểm tra trùng tên
                        existing_names = [s['name'] for s in st.session_state.spectra_collection]
                        final_name = file_base_name
                        counter = 1
                        while final_name in existing_names:
                            final_name = f"{file_base_name}_{counter}"
                            counter += 1

                        st.session_state.spectra_collection.append({
                            'name': final_name,
                            'original_filename': uploaded_file.name,
                            'data': loaded_data,
                            'preprocessed': None,
                            'selected': True
                        })

                    loaded_count += 1

                except Exception as e:
                    failed_files.append((uploaded_file.name, str(e)))

            # Hiển thị kết quả
            if loaded_count > 0:
                st.success(f"✅ Đã tải thành công {loaded_count} file(s)")
                if auto_add_to_collection:
                    st.info(f"💡 Đã thêm {loaded_count} phổ vào Collection. Mở rộng '📚 Quản lý Collection Phổ' để xem.")

            if failed_files:
                st.error(f"❌ Lỗi khi tải {len(failed_files)} file(s):")
                for fname, error in failed_files:
                    st.write(f"- {fname}: {error}")

            # Hiển thị thông tin phổ cuối cùng được load
            if st.session_state.data is not None and loaded_count > 0:
                st.write("### Thông tin phổ cuối cùng được tải")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Kiểu dữ liệu", type(st.session_state.data).__name__)
                with col2:
                    st.metric("Kích thước", str(st.session_state.data.shape))
                with col3:
                    if hasattr(st.session_state.data, 'spectral_axis'):
                        st.metric("Số điểm phổ", len(st.session_state.data.spectral_axis))

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

    # ==================== QUẢN LÝ NHIỀU PHỔ ====================
    st.markdown("---")
    with st.expander("📚 Quản lý Collection Phổ (để chạy PCA với nhiều phổ)", expanded=False):
        st.write("### Thêm phổ hiện tại vào collection")

        col_name, col_add = st.columns([3, 1])

        with col_name:
            spectrum_name = st.text_input(
                "Tên phổ:",
                value=f"Spectrum_{len(st.session_state.spectra_collection)+1}",
                help="Đặt tên để dễ quản lý"
            )

        with col_add:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("➕ Thêm vào Collection"):
                if st.session_state.data is not None:
                    # Kiểm tra trùng tên
                    existing_names = [s['name'] for s in st.session_state.spectra_collection]
                    if spectrum_name in existing_names:
                        st.error(f"❌ Tên '{spectrum_name}' đã tồn tại!")
                    else:
                        st.session_state.spectra_collection.append({
                            'name': spectrum_name,
                            'data': st.session_state.data,
                            'preprocessed': st.session_state.preprocessed_data,
                            'selected': True
                        })
                        st.success(f"✅ Đã thêm '{spectrum_name}' vào collection!")
                        st.rerun()
                else:
                    st.warning("⚠️ Chưa có dữ liệu để thêm. Vui lòng tải file trước!")

        # Hiển thị collection
        if len(st.session_state.spectra_collection) > 0:
            st.write(f"### 📋 Collection ({len(st.session_state.spectra_collection)} phổ)")

            # Selection mode
            col_mode1, col_mode2 = st.columns(2)
            with col_mode1:
                if st.button("✅ Chọn tất cả"):
                    for spec in st.session_state.spectra_collection:
                        spec['selected'] = True
                    st.rerun()
            with col_mode2:
                if st.button("☐ Bỏ chọn tất cả"):
                    for spec in st.session_state.spectra_collection:
                        spec['selected'] = False
                    st.rerun()

            # List spectra với rename
            for i, spec in enumerate(st.session_state.spectra_collection):
                col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 0.6, 0.6])

                with col1:
                    new_selected = st.checkbox(
                        "☑",
                        value=spec['selected'],
                        key=f"select_{i}",
                        label_visibility="collapsed"
                    )
                    if new_selected != spec['selected']:
                        spec['selected'] = new_selected

                with col2:
                    # Hiển thị tên file gốc nếu có
                    original_name = spec.get('original_filename', '')
                    if original_name:
                        st.write(f"📄 `{original_name}`")
                    else:
                        st.write(f"Phổ #{i+1}")

                with col3:
                    # Editable name
                    new_name = st.text_input(
                        "Tên:",
                        value=spec['name'],
                        key=f"name_{i}",
                        label_visibility="collapsed",
                        placeholder="Đặt tên..."
                    )
                    if new_name != spec['name'] and new_name.strip():
                        # Check duplicate
                        existing = [s['name'] for idx, s in enumerate(st.session_state.spectra_collection) if idx != i]
                        if new_name not in existing:
                            spec['name'] = new_name

                    # Status
                    data_shape = spec['data'].shape if hasattr(spec['data'], 'shape') else "N/A"
                    preprocessed_status = "✅" if spec['preprocessed'] is not None else "⚪"
                    st.caption(f"{preprocessed_status} {data_shape}")

                with col4:
                    if st.button("🗑️", key=f"del_{i}", help="Xóa"):
                        st.session_state.spectra_collection.pop(i)
                        st.rerun()

                with col5:
                    if st.button("👁️", key=f"view_{i}", help="Load"):
                        st.session_state.data = spec['data']
                        st.session_state.preprocessed_data = spec['preprocessed']
                        st.success(f"Đã load '{spec['name']}'")
                        st.rerun()

            # Actions
            st.markdown("---")
            selected_count = sum(1 for s in st.session_state.spectra_collection if s['selected'])
            st.info(f"**Đã chọn:** {selected_count} phổ")

            if selected_count > 0:
                col_action1, col_action2 = st.columns(2)

                with col_action1:
                    # Batch preprocessing
                    if st.button("⚙️ Tiền xử lý hàng loạt", use_container_width=True, help="Áp dụng pipeline cho các phổ đã chọn"):
                        # Chuyển sang tab preprocessing với flag
                        st.session_state['batch_preprocess_mode'] = True
                        st.info("💡 Chuyển sang tab 'Tiền xử lý', thiết lập pipeline, và click 'Áp dụng cho Collection'")

                with col_action2:
                    # Combine spectra
                    if selected_count > 1:
                        if st.button("🔗 Kết hợp để chạy PCA", type="primary", use_container_width=True):
                            # Get selected items
                            selected_items = [s for s in st.session_state.spectra_collection if s['selected']]

                            try:
                                # Ưu tiên dùng preprocessed data nếu có
                                spectra_arrays = []
                                using_preprocessed = False

                                for item in selected_items:
                                    # Dùng preprocessed nếu có, không thì dùng raw
                                    spec = item['preprocessed'] if item['preprocessed'] is not None else item['data']

                                    if item['preprocessed'] is not None:
                                        using_preprocessed = True

                                    if hasattr(spec, 'spectral_data'):
                                        spectra_arrays.append(spec.spectral_data)
                                    else:
                                        spectra_arrays.append(np.array(spec))

                                combined_array = np.stack(spectra_arrays)

                                # Get common spectral axis
                                first_spec = selected_items[0]['preprocessed'] if selected_items[0]['preprocessed'] is not None else selected_items[0]['data']
                                if hasattr(first_spec, 'spectral_axis'):
                                    spectral_axis = first_spec.spectral_axis
                                else:
                                    spectral_axis = np.arange(combined_array.shape[-1])

                                # Create SpectralContainer
                                st.session_state.data = rp.SpectralContainer(combined_array, spectral_axis=spectral_axis)
                                st.session_state.preprocessed_data = None

                                if using_preprocessed:
                                    st.success(f"✅ Đã kết hợp {selected_count} phổ (sử dụng dữ liệu đã tiền xử lý)!")
                                else:
                                    st.success(f"✅ Đã kết hợp {selected_count} phổ (dữ liệu gốc)!")
                                    st.warning("⚠️ Một số phổ chưa được tiền xử lý. Khuyến nghị tiền xử lý trước khi phân tích.")

                                st.info("💡 Chuyển sang tab 'Phân tích' để chạy PCA.")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Lỗi khi kết hợp phổ: {str(e)}")
                    else:
                        st.info("💡 Chọn ít nhất 2 phổ để kết hợp và chạy PCA.")

            elif selected_count == 1:
                st.info("💡 Chỉ chọn 1 phổ. Sử dụng Peak Detection để phân tích phổ đơn.")
        else:
            st.info("Collection trống. Tải file và chọn 'Tự động thêm vào Collection' khi upload.")

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
            st.write("Phương pháp: **WhitakerHayes**")
            with st.expander("⚙️ Tùy chỉnh parameters"):
                despike_kernel = st.slider("Kernel size:", 1, 9, 3, 2, help="Kích thước kernel để detect spikes")
                despike_threshold = st.slider("Threshold:", 1.0, 20.0, 8.0, 1.0, help="Ngưỡng để xác định spike")

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
                        steps.append(rp.preprocessing.despike.WhitakerHayes(
                            kernel_size=despike_kernel,
                            threshold=despike_threshold
                        ))

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

    with col_btn3:
        # Batch preprocessing for collection
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 0:
            if st.button(f"⚙️ Áp dụng cho Collection ({len(selected_in_collection)})", use_container_width=True):
                try:
                    with st.spinner(f"Đang xử lý {len(selected_in_collection)} phổ..."):
                        # Xây dựng pipeline
                        steps = []

                        if use_cropping:
                            steps.append(rp.preprocessing.misc.Cropper(region=(crop_min, crop_max)))

                        if use_despike:
                            steps.append(rp.preprocessing.despike.WhitakerHayes(
                                kernel_size=despike_kernel,
                                threshold=despike_threshold
                            ))

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

                        # Tạo pipeline
                        pipeline = rp.preprocessing.Pipeline(steps)

                        # Áp dụng cho từng phổ được chọn
                        success_count = 0
                        for item in st.session_state.spectra_collection:
                            if item['selected']:
                                try:
                                    item['preprocessed'] = pipeline.apply(item['data'])
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"Lỗi khi xử lý '{item['name']}': {str(e)}")

                        st.success(f"✅ Đã xử lý {success_count}/{len(selected_in_collection)} phổ với {len(steps)} bước!")
                        st.info("💡 Giờ bạn có thể kết hợp các phổ đã xử lý để chạy PCA.")

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý: {str(e)}")
        else:
            st.info("💡 Chọn phổ trong Collection để xử lý hàng loạt")

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

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(raw_spectrum, 'spectral_axis') and hasattr(raw_spectrum, 'spectral_data'):
                    ax1.plot(raw_spectrum.spectral_axis, raw_spectrum.spectral_data, linewidth=1.5)
                    ax1.set_xlabel("Wavenumber (cm⁻¹)")
                    ax1.set_ylabel("Intensity")
                else:
                    rp.plot.spectra(raw_spectrum, ax=ax1)

                ax1.set_title("Phổ gốc")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close()

            with col_after:
                st.write("**Sau xử lý**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(processed_spectrum, 'spectral_axis') and hasattr(processed_spectrum, 'spectral_data'):
                    ax2.plot(processed_spectrum.spectral_axis, processed_spectrum.spectral_data, linewidth=1.5)
                    ax2.set_xlabel("Wavenumber (cm⁻¹)")
                    ax2.set_ylabel("Intensity")
                else:
                    rp.plot.spectra(processed_spectrum, ax=ax2)

                ax2.set_title("Phổ đã xử lý")
                ax2.grid(True, alpha=0.3)
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

    # Debug info
    st.write(f"**Loại dữ liệu phân tích:** {type(data_to_analyze).__name__}")

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
                # Kiểm tra nếu là Spectrum đơn lẻ
                data_type = type(data_to_analyze).__name__

                # Kiểm tra shape để xác định có phải single spectrum không
                if data_type == 'Spectrum':
                    if hasattr(data_to_analyze, 'spectral_data'):
                        data_shape = data_to_analyze.spectral_data.shape
                    else:
                        data_shape = data_to_analyze.shape if hasattr(data_to_analyze, 'shape') else (1,)

                    # Nếu là 1D hoặc shape[0] == 1, là single spectrum
                    if len(data_shape) == 1 or (len(data_shape) > 1 and data_shape[0] == 1):
                        st.error("❌ Spectral Unmixing cần nhiều phổ (ảnh hoặc volumetric data).")
                        st.info("💡 Với 1 phổ đơn lẻ, sử dụng Peak Detection thay vì Unmixing.")
                        st.stop()

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
                    # Lấy phổ để phân tích
                    data_type = type(data_to_analyze).__name__

                    if data_type == 'Spectrum':
                        # Spectrum đơn lẻ
                        spectrum = data_to_analyze
                    elif hasattr(data_to_analyze, 'flat'):
                        # Volumetric data
                        spectrum = data_to_analyze.flat[0]
                    elif hasattr(data_to_analyze, '__len__') and len(data_to_analyze.shape) > 1:
                        # Multi-spectrum data
                        spectrum = data_to_analyze[0]
                    else:
                        spectrum = data_to_analyze

                    # Lấy intensities một cách an toàn
                    if hasattr(spectrum, 'spectral_data'):
                        intensities = spectrum.spectral_data
                    elif hasattr(spectrum, 'flat'):
                        intensities = spectrum.flat
                    elif isinstance(spectrum, np.ndarray):
                        intensities = spectrum
                    else:
                        # Fallback: chuyển về numpy array
                        intensities = np.array(spectrum)

                    # Đảm bảo là 1D array
                    if len(intensities.shape) > 1:
                        intensities = intensities.flatten()

                    # Tìm peaks
                    peaks, properties = find_peaks(intensities, prominence=prominence, distance=distance)

                    # Lấy spectral axis an toàn
                    if hasattr(spectrum, 'spectral_axis'):
                        spectral_axis = spectrum.spectral_axis
                    else:
                        # Nếu không có, tạo index array
                        spectral_axis = np.arange(len(intensities))

                    st.session_state.analysis_results = {
                        'type': 'peaks',
                        'spectrum': spectrum,
                        'peaks': peaks,
                        'properties': properties,
                        'intensities': intensities,
                        'spectral_axis': spectral_axis
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
                    data_type = type(data_to_analyze).__name__

                    if data_type == 'Spectrum':
                        # Spectrum đơn lẻ - PCA cần ít nhất 2 mẫu
                        st.error("❌ PCA cần ít nhất 2 phổ. Dữ liệu hiện tại chỉ có 1 phổ đơn lẻ.")
                        st.info("💡 Sử dụng dữ liệu tổng hợp hoặc tải nhiều phổ để chạy PCA.")
                        st.stop()
                    elif hasattr(data_to_analyze, 'flat'):
                        # Volumetric data
                        data_matrix = data_to_analyze.flat.spectral_data
                    else:
                        data_matrix = data_to_analyze.spectral_data if hasattr(data_to_analyze, 'spectral_data') else data_to_analyze

                    # Reshape nếu cần
                    if len(data_matrix.shape) > 2:
                        original_shape = data_matrix.shape
                        data_matrix = data_matrix.reshape(-1, data_matrix.shape[-1])
                    elif len(data_matrix.shape) == 1:
                        # Nếu chỉ có 1 phổ, không thể chạy PCA
                        st.error("❌ PCA cần ít nhất 2 phổ để phân tích.")
                        st.stop()

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

        peaks = results['peaks']
        intensities = results['intensities']
        spectral_axis = results['spectral_axis']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot spectrum
        ax.plot(spectral_axis, intensities, 'b-', linewidth=1.5, label='Spectrum')
        ax.plot(spectral_axis[peaks], intensities[peaks], 'ro', markersize=8, label='Peaks')

        # Đánh dấu peaks
        for peak in peaks:
            ax.axvline(spectral_axis[peak], color='r', linestyle='--', alpha=0.3)
            ax.text(spectral_axis[peak], intensities[peak], f'{spectral_axis[peak]:.0f}',
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
            'Wavenumber (cm⁻¹)': [f"{spectral_axis[p]:.2f}" for p in peaks],
            'Intensity': [f"{intensities[p]:.4f}" for p in peaks]
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
