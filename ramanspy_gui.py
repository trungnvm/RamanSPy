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
import io
import pandas as pd

# Helper functions
def plot_with_download(fig, filename="plot.png", download_label="📥 Tải plot"):
    """Display plot with download button"""
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # Display plot
    st.pyplot(fig)

    # Download button
    st.download_button(
        label=download_label,
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

    plt.close(fig)

def create_csv_download(dataframe, filename="data.csv", label="📥 Tải CSV"):
    """Create download button for CSV data"""
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)

    st.download_button(
        label=label,
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

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
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids = set()  # Track which files have been added to avoid duplicates

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
        st.markdown("---")
        auto_add_to_collection = st.checkbox(
            "✅ Tự động thêm TẤT CẢ vào Collection sau khi tải",
            value=True,
            help="Khuyến nghị BẬT option này để tất cả files được thêm vào Collection tự động"
        )
        if auto_add_to_collection:
            st.caption("💡 Tất cả files upload sẽ tự động được thêm vào Collection để quản lý và xử lý hàng loạt")
        else:
            st.caption("⚠️ Files sẽ KHÔNG được thêm vào Collection - bạn phải thêm thủ công")

        # Hiển thị số files đã được xử lý
        if len(st.session_state.processed_file_ids) > 0:
            st.info(f"📊 Đã xử lý {len(st.session_state.processed_file_ids)} file(s) trong session này. Mỗi file chỉ được thêm vào Collection 1 lần duy nhất.")

        if uploaded_files:
            loaded_count = 0
            failed_files = []

            for uploaded_file in uploaded_files:
                # Tạo unique ID cho file (dựa trên tên file + kích thước + file_id nếu có)
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if hasattr(uploaded_file, 'file_id'):
                    file_id = f"{uploaded_file.file_id}_{uploaded_file.name}_{uploaded_file.size}"

                # Kiểm tra xem file đã được xử lý chưa
                if file_id in st.session_state.processed_file_ids:
                    continue  # Skip file này vì đã xử lý rồi

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

                    # Đánh dấu file này đã được xử lý
                    st.session_state.processed_file_ids.add(file_id)

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
            # Thống kê collection
            total_count = len(st.session_state.spectra_collection)
            preprocessed_count = sum(1 for s in st.session_state.spectra_collection if s['preprocessed'] is not None)

            st.write(f"### 📋 Collection ({total_count} phổ)")

            # Progress bar cho preprocessing status
            if total_count > 0:
                progress = preprocessed_count / total_count
                st.progress(progress, text=f"Đã tiền xử lý: {preprocessed_count}/{total_count} phổ ({progress*100:.0f}%)")

            st.markdown("")  # Spacing

            # Selection mode
            col_mode1, col_mode2, col_mode3, col_mode4 = st.columns(4)
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
            with col_mode3:
                if st.button("🗑️ Xóa tất cả"):
                    st.session_state.spectra_collection = []
                    st.session_state.processed_file_ids = set()
                    st.rerun()
            with col_mode4:
                if st.button("🔄 Reset cache"):
                    st.session_state.processed_file_ids = set()
                    st.info("Đã xóa cache upload")
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

                    # Status với màu sắc rõ ràng
                    data_shape = spec['data'].shape if hasattr(spec['data'], 'shape') else "N/A"
                    if spec['preprocessed'] is not None:
                        st.caption(f"✅ **Đã xử lý** | {data_shape}")
                    else:
                        st.caption(f"⚪ *Chưa xử lý* | {data_shape}")

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

            # Hiển thị thống kê chi tiết
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Đã chọn", f"{selected_count}/{total_count}")
            with col_stat2:
                selected_preprocessed = sum(1 for s in st.session_state.spectra_collection if s['selected'] and s['preprocessed'] is not None)
                st.metric("Đã xử lý", f"{selected_preprocessed}/{selected_count}")
            with col_stat3:
                selected_raw = selected_count - selected_preprocessed
                if selected_raw > 0:
                    st.metric("Chưa xử lý", selected_raw, delta="Cần xử lý", delta_color="off")
                else:
                    st.metric("Chưa xử lý", "0", delta="✓", delta_color="normal")

            st.markdown("")

            if selected_count > 0:
                # Batch preprocessing section
                st.write("#### ⚙️ Tiền xử lý hàng loạt")
                st.info("💡 Chuyển sang tab **'Tiền xử lý'** để:")
                st.markdown("""
                1. Thiết lập pipeline tiền xử lý
                2. Click **'⚙️ Áp dụng cho Collection'** để xử lý hàng loạt
                3. Click **'🔗 Kết hợp phổ'** để tạo SpectralContainer cho PCA
                """)

                col_quick = st.columns(1)[0]
                with col_quick:
                    # Quick access button
                    if st.button("📍 Đi tới Tiền xử lý", use_container_width=True, type="primary"):
                        st.session_state['show_batch_hint'] = True
                        st.info("Chuyển sang tab 'Tiền xử lý' bên trên!")

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
                # Spectrum đơn lẻ - plot với màu đẹp
                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

                if hasattr(st.session_state.data, 'spectral_axis') and hasattr(st.session_state.data, 'spectral_data'):
                    ax.plot(st.session_state.data.spectral_axis, st.session_state.data.spectral_data,
                           color='#1f77b4', linewidth=1.5)
                    ax.set_xlabel("Wavenumber (cm⁻¹)")
                    ax.set_ylabel("Intensity")
                else:
                    rp.plot.spectra(st.session_state.data, ax=ax, plot_type='single')

                ax.set_title("Preview phổ Raman")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            elif data_type == 'SpectralContainer' or (hasattr(st.session_state.data, 'spectral_data') and hasattr(st.session_state.data.spectral_data, 'shape') and len(st.session_state.data.spectral_data.shape) > 1):
                # SpectralContainer hoặc multi-spectrum data
                if hasattr(st.session_state.data, 'spectral_data'):
                    n_spectra = min(5, len(st.session_state.data.spectral_data))
                else:
                    n_spectra = min(5, len(st.session_state.data))

                # Lấy tên từ Collection nếu có
                spectrum_labels = []
                if len(st.session_state.spectra_collection) > 0:
                    selected_items = [s for s in st.session_state.spectra_collection if s['selected']]
                    if len(selected_items) > 0:
                        spectrum_labels = [item['name'] for item in selected_items[:n_spectra]]

                # Fallback labels nếu không có từ collection
                if len(spectrum_labels) == 0:
                    spectrum_labels = [f'Phổ {i+1}' for i in range(n_spectra)]

                colors = plt.cm.tab10(np.linspace(0, 1, n_spectra))

                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                for i in range(n_spectra):
                    # Get spectrum
                    if hasattr(st.session_state.data, 'spectral_data'):
                        y_data = st.session_state.data.spectral_data[i]
                        x_data = st.session_state.data.spectral_axis if hasattr(st.session_state.data, 'spectral_axis') else np.arange(len(y_data))
                    else:
                        spec = st.session_state.data[i]
                        if hasattr(spec, 'spectral_axis') and hasattr(spec, 'spectral_data'):
                            x_data = spec.spectral_axis
                            y_data = spec.spectral_data
                        else:
                            y_data = spec if isinstance(spec, np.ndarray) else np.array(spec)
                            x_data = np.arange(len(y_data))

                    # Flatten if needed
                    if hasattr(y_data, 'shape') and len(y_data.shape) > 1:
                        y_data = y_data.flatten()

                    # Use name from collection
                    label = spectrum_labels[i] if i < len(spectrum_labels) else f'Phổ {i+1}'
                    ax.plot(x_data, y_data, color=colors[i], linewidth=1.5, alpha=0.7, label=label)

                ax.set_title(f"Preview phổ Raman ({n_spectra} phổ)")
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            elif hasattr(st.session_state.data, 'flat'):
                # Volumetric data - plot 5 phổ đầu với màu khác nhau
                sample_spectra = st.session_state.data.flat[0:5]
                n_samples = len(sample_spectra)
                colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                for i in range(n_samples):
                    spec = sample_spectra[i]
                    if hasattr(spec, 'spectral_axis') and hasattr(spec, 'spectral_data'):
                        ax.plot(spec.spectral_axis, spec.spectral_data,
                               color=colors[i], linewidth=1.5, alpha=0.7, label=f'Phổ {i+1}')

                ax.set_title("Preview phổ Raman (5 phổ đầu)")
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            else:
                # Fallback
                sample_spectra = st.session_state.data
                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                rp.plot.spectra(sample_spectra, ax=ax, plot_type='single')
                ax.set_title("Preview phổ Raman")
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Không thể hiển thị preview: {str(e)}")
            st.info(f"Debug: Data type = {type(st.session_state.data).__name__}")

# ==================== TRANG TIỀN XỬ LÝ ====================
elif page == "Tiền xử lý":
    st.markdown('<p class="sub-header">⚙️ Tiền xử lý dữ liệu</p>', unsafe_allow_html=True)

    # Hiển thị thông tin collection nếu có
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 0:
            preprocessed_in_selected = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)
            raw_in_selected = len(selected_in_collection) - preprocessed_in_selected

            if raw_in_selected > 0:
                st.info(f"📚 Collection: {len(selected_in_collection)} phổ đã chọn | ✅ {preprocessed_in_selected} đã xử lý | ⚪ {raw_in_selected} chưa xử lý")
            else:
                st.success(f"📚 Collection: {len(selected_in_collection)} phổ đã chọn | ✅ Tất cả đã xử lý!")

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
            crop_min = st.number_input("Wavenumber min (cm⁻¹):", 0, 4000, 700, 50)
            crop_max = st.number_input("Wavenumber max (cm⁻¹):", 0, 4000, 1800, 50)

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
            button_label = f"⚙️ Áp dụng cho Collection ({len(selected_in_collection)} phổ)"
            if st.button(button_label, use_container_width=True, type="primary"):
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

                        # Áp dụng cho từng phổ được chọn với progress bar
                        success_count = 0
                        progress_bar = st.progress(0, text="Bắt đầu xử lý...")

                        for idx, item in enumerate(st.session_state.spectra_collection):
                            if item['selected']:
                                try:
                                    progress = (idx + 1) / len(selected_in_collection)
                                    progress_bar.progress(progress, text=f"Đang xử lý {item['name']}... ({idx + 1}/{len(selected_in_collection)})")

                                    item['preprocessed'] = pipeline.apply(item['data'])
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"Lỗi khi xử lý '{item['name']}': {str(e)}")

                        progress_bar.progress(1.0, text="✅ Hoàn thành!")

                        st.success(f"✅ Đã xử lý thành công {success_count}/{len(selected_in_collection)} phổ với {len(steps)} bước!")

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý: {str(e)}")
        else:
            st.info("💡 Chọn phổ trong Collection để xử lý hàng loạt")

    # Combine section - Kết hợp phổ để phân tích (DI CHUYỂN TỪ TAB TẢI DỮ LIỆU)
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 1:
            st.markdown("---")
            st.write("### 🔗 Kết hợp phổ để phân tích")

            col_combine1, col_combine2 = st.columns([2, 1])

            # Calculate stats
            selected_preprocessed = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)
            selected_raw = len(selected_in_collection) - selected_preprocessed

            with col_combine1:
                # Hiển thị warning nếu có phổ chưa xử lý
                if selected_raw > 0:
                    st.warning(f"⚠️ Có {selected_raw} phổ chưa tiền xử lý. Khuyến nghị xử lý trước khi kết hợp.")
                else:
                    st.success(f"✅ Tất cả {len(selected_in_collection)} phổ đã được tiền xử lý!")
                    st.info("💡 Click 'Kết hợp phổ' để tạo SpectralContainer cho PCA")

            with col_combine2:
                # Combine spectra button
                if st.button("🔗 Kết hợp phổ", type="primary", use_container_width=True, key="combine_preprocessing"):
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
                            st.success(f"✅ Đã kết hợp {len(selected_in_collection)} phổ (sử dụng dữ liệu đã tiền xử lý)!")
                        else:
                            st.success(f"✅ Đã kết hợp {len(selected_in_collection)} phổ (dữ liệu gốc)!")
                            st.warning("⚠️ Một số phổ chưa được tiền xử lý. Khuyến nghị tiền xử lý trước khi phân tích.")

                        st.info("💡 Chuyển sang tab 'Phân tích' để chạy PCA.")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Lỗi khi kết hợp phổ: {str(e)}")

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
                fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(raw_spectrum, 'spectral_axis') and hasattr(raw_spectrum, 'spectral_data'):
                    ax1.plot(raw_spectrum.spectral_axis, raw_spectrum.spectral_data, linewidth=1.5)
                    ax1.set_xlabel("Wavenumber (cm⁻¹)")
                    ax1.set_ylabel("Intensity")
                else:
                    rp.plot.spectra(raw_spectrum, ax=ax1)

                ax1.set_title("Phổ gốc")
                ax1.grid(True, alpha=0.3)
                plot_with_download(fig1, "raw_spectrum.png", "📥 Tải phổ gốc")

            with col_after:
                st.write("**Sau xử lý**")
                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(processed_spectrum, 'spectral_axis') and hasattr(processed_spectrum, 'spectral_data'):
                    ax2.plot(processed_spectrum.spectral_axis, processed_spectrum.spectral_data, linewidth=1.5)
                    ax2.set_xlabel("Wavenumber (cm⁻¹)")
                    ax2.set_ylabel("Intensity")
                else:
                    rp.plot.spectra(processed_spectrum, ax=ax2)

                ax2.set_title("Phổ đã xử lý")
                ax2.grid(True, alpha=0.3)
                plot_with_download(fig2, "preprocessed_spectrum.png", "📥 Tải phổ đã xử lý")

        except Exception as e:
            st.error(f"Lỗi khi hiển thị so sánh: {str(e)}")

        # CSV Export cho preprocessing data
        st.markdown("---")
        st.write("### 📥 Tải dữ liệu tiền xử lý")

        col_csv1, col_csv2 = st.columns(2)

        with col_csv1:
            st.write("**Phổ gốc**")
            if hasattr(raw_spectrum, 'spectral_axis') and hasattr(raw_spectrum, 'spectral_data'):
                raw_df = pd.DataFrame({
                    'Wavenumber (cm⁻¹)': raw_spectrum.spectral_axis,
                    'Intensity': raw_spectrum.spectral_data
                })
                create_csv_download(raw_df, "raw_spectrum.csv", "📥 Tải phổ gốc CSV")
            else:
                st.info("Không thể export dữ liệu này")

        with col_csv2:
            st.write("**Phổ đã xử lý**")
            if hasattr(processed_spectrum, 'spectral_axis') and hasattr(processed_spectrum, 'spectral_data'):
                processed_df = pd.DataFrame({
                    'Wavenumber (cm⁻¹)': processed_spectrum.spectral_axis,
                    'Intensity': processed_spectrum.spectral_data
                })
                create_csv_download(processed_df, "preprocessed_spectrum.csv", "📥 Tải phổ đã xử lý CSV")
            else:
                st.info("Không thể export dữ liệu này")

    # Overlay plots cho batch preprocessing
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        preprocessed_count = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)

        if len(selected_in_collection) > 1 and preprocessed_count > 0:
            st.markdown("---")
            st.write("### 📊 So sánh chồng phổ (Overlay)")
            st.info(f"Hiển thị {len(selected_in_collection)} phổ đã chọn trong Collection ({preprocessed_count} đã xử lý)")

            try:
                # Tạo figure với 2 subplots
                fig, (ax_raw, ax_processed) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

                # Colormap cho màu sắc đẹp
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_in_collection)))

                # Plot raw spectra (bên trái)
                ax_raw.set_title("Phổ gốc (Raw Spectra)", fontsize=12, fontweight='bold')
                ax_raw.set_xlabel("Wavenumber (cm⁻¹)")
                ax_raw.set_ylabel("Intensity")
                ax_raw.grid(True, alpha=0.3)

                for idx, item in enumerate(selected_in_collection):
                    raw_spec = item['data']

                    # Extract data safely
                    if hasattr(raw_spec, 'spectral_axis') and hasattr(raw_spec, 'spectral_data'):
                        x_data = raw_spec.spectral_axis
                        y_data = raw_spec.spectral_data
                    elif hasattr(raw_spec, 'spectral_axis'):
                        x_data = raw_spec.spectral_axis
                        y_data = np.array(raw_spec)
                    else:
                        y_data = np.array(raw_spec)
                        x_data = np.arange(len(y_data))

                    # Flatten if needed
                    if len(y_data.shape) > 1:
                        y_data = y_data.flatten()

                    ax_raw.plot(x_data, y_data, color=colors[idx], linewidth=1.5, alpha=0.7, label=item['name'])

                ax_raw.legend(loc='best', fontsize=8, framealpha=0.9)

                # Plot preprocessed spectra (bên phải)
                ax_processed.set_title("Phổ đã xử lý (Preprocessed Spectra)", fontsize=12, fontweight='bold')
                ax_processed.set_xlabel("Wavenumber (cm⁻¹)")
                ax_processed.set_ylabel("Intensity")
                ax_processed.grid(True, alpha=0.3)

                preprocessed_items = [item for item in selected_in_collection if item['preprocessed'] is not None]

                if len(preprocessed_items) > 0:
                    # Re-create colors for preprocessed items
                    proc_colors = plt.cm.tab10(np.linspace(0, 1, len(preprocessed_items)))

                    for idx, item in enumerate(preprocessed_items):
                        proc_spec = item['preprocessed']

                        # Extract data safely
                        if hasattr(proc_spec, 'spectral_axis') and hasattr(proc_spec, 'spectral_data'):
                            x_data = proc_spec.spectral_axis
                            y_data = proc_spec.spectral_data
                        elif hasattr(proc_spec, 'spectral_axis'):
                            x_data = proc_spec.spectral_axis
                            y_data = np.array(proc_spec)
                        else:
                            y_data = np.array(proc_spec)
                            x_data = np.arange(len(y_data))

                        # Flatten if needed
                        if len(y_data.shape) > 1:
                            y_data = y_data.flatten()

                        ax_processed.plot(x_data, y_data, color=proc_colors[idx], linewidth=1.5, alpha=0.7, label=item['name'])

                    ax_processed.legend(loc='best', fontsize=8, framealpha=0.9)
                else:
                    ax_processed.text(0.5, 0.5, 'Chưa có phổ nào được xử lý',
                                     ha='center', va='center', transform=ax_processed.transAxes,
                                     fontsize=12, style='italic')

                plt.tight_layout()
                plot_with_download(fig, "batch_preprocessing_overlay.png", "📥 Tải overlay plots")

                # CSV Export cho batch preprocessing
                st.markdown("---")
                st.write("### 📥 Tải dữ liệu batch preprocessing")

                col_batch1, col_batch2 = st.columns(2)

                with col_batch1:
                    st.write("**Tất cả phổ gốc**")
                    try:
                        # Tạo DataFrame cho tất cả phổ raw
                        raw_data_dict = {}
                        wavenumbers = None

                        for item in selected_in_collection:
                            raw_spec = item['data']
                            if hasattr(raw_spec, 'spectral_axis') and hasattr(raw_spec, 'spectral_data'):
                                if wavenumbers is None:
                                    wavenumbers = raw_spec.spectral_axis
                                y_data = raw_spec.spectral_data
                                if len(y_data.shape) > 1:
                                    y_data = y_data.flatten()
                                raw_data_dict[item['name']] = y_data

                        if wavenumbers is not None and len(raw_data_dict) > 0:
                            raw_batch_df = pd.DataFrame(raw_data_dict)
                            raw_batch_df.insert(0, 'Wavenumber (cm⁻¹)', wavenumbers)
                            create_csv_download(raw_batch_df, "batch_raw_spectra.csv", "📥 Tải tất cả phổ gốc CSV")
                        else:
                            st.info("Không có dữ liệu để export")
                    except Exception as e:
                        st.warning(f"Không thể export: {str(e)}")

                with col_batch2:
                    st.write("**Tất cả phổ đã xử lý**")
                    if len(preprocessed_items) > 0:
                        try:
                            # Tạo DataFrame cho tất cả phổ preprocessed
                            proc_data_dict = {}
                            wavenumbers = None

                            for item in preprocessed_items:
                                proc_spec = item['preprocessed']
                                if hasattr(proc_spec, 'spectral_axis') and hasattr(proc_spec, 'spectral_data'):
                                    if wavenumbers is None:
                                        wavenumbers = proc_spec.spectral_axis
                                    y_data = proc_spec.spectral_data
                                    if len(y_data.shape) > 1:
                                        y_data = y_data.flatten()
                                    proc_data_dict[item['name']] = y_data

                            if wavenumbers is not None and len(proc_data_dict) > 0:
                                proc_batch_df = pd.DataFrame(proc_data_dict)
                                proc_batch_df.insert(0, 'Wavenumber (cm⁻¹)', wavenumbers)
                                create_csv_download(proc_batch_df, "batch_preprocessed_spectra.csv", "📥 Tải tất cả phổ đã xử lý CSV")
                            else:
                                st.info("Không có dữ liệu để export")
                        except Exception as e:
                            st.warning(f"Không thể export: {str(e)}")
                    else:
                        st.info("Chưa có phổ nào được tiền xử lý")

            except Exception as e:
                st.error(f"Lỗi khi hiển thị overlay plots: {str(e)}")

        # Stacked plots cho batch preprocessing
        if len(selected_in_collection) > 1:
            st.markdown("---")
            st.write("### 📚 So sánh phổ dạng Stacked")
            st.info("Stacked plot giúp so sánh từng phổ riêng lẻ bằng cách xếp chồng với offset theo trục Y")

            # Options
            col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1])

            with col_opt1:
                stack_mode = st.radio(
                    "Chọn dữ liệu:",
                    ["Phổ gốc", "Phổ đã xử lý", "So sánh cả 2"],
                    horizontal=True
                )

            with col_opt2:
                offset_multiplier = st.slider(
                    "Khoảng cách giữa các phổ:",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Điều chỉnh khoảng cách offset giữa các phổ"
                )

            with col_opt3:
                reverse_order = st.checkbox("Đảo ngược thứ tự", value=False)

            try:
                if stack_mode == "So sánh cả 2":
                    fig, (ax_raw_stack, ax_proc_stack) = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
                    axes_list = [ax_raw_stack, ax_proc_stack]
                    titles = ["Phổ gốc (Stacked)", "Phổ đã xử lý (Stacked)"]
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
                    axes_list = [ax]
                    titles = [f"{stack_mode} (Stacked)"]

                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_in_collection)))

                # Determine order
                items_to_plot = list(reversed(selected_in_collection)) if reverse_order else selected_in_collection

                for ax_idx, ax_current in enumerate(axes_list):
                    # Determine which data to use
                    if stack_mode == "Phổ gốc" or (stack_mode == "So sánh cả 2" and ax_idx == 0):
                        use_raw = True
                    else:
                        use_raw = False

                    # Filter items based on data availability
                    if use_raw:
                        items = items_to_plot
                    else:
                        items = [item for item in items_to_plot if item['preprocessed'] is not None]

                    if len(items) == 0:
                        ax_current.text(0.5, 0.5, 'Không có dữ liệu để hiển thị',
                                      ha='center', va='center', transform=ax_current.transAxes,
                                      fontsize=12, style='italic')
                        ax_current.set_title(titles[ax_idx], fontsize=12, fontweight='bold')
                        continue

                    # Calculate offset
                    max_intensity = 0
                    all_intensities = []
                    for item in items:
                        spec = item['data'] if use_raw else item['preprocessed']
                        if hasattr(spec, 'spectral_data'):
                            y = spec.spectral_data
                        else:
                            y = np.array(spec)
                        if len(y.shape) > 1:
                            y = y.flatten()
                        all_intensities.append(y)
                        max_intensity = max(max_intensity, np.max(y) - np.min(y))

                    offset = max_intensity * offset_multiplier

                    # Plot each spectrum with offset
                    for idx, item in enumerate(items):
                        spec = item['data'] if use_raw else item['preprocessed']

                        # Extract data safely
                        if hasattr(spec, 'spectral_axis') and hasattr(spec, 'spectral_data'):
                            x_data = spec.spectral_axis
                            y_data = spec.spectral_data
                        elif hasattr(spec, 'spectral_axis'):
                            x_data = spec.spectral_axis
                            y_data = np.array(spec)
                        else:
                            y_data = np.array(spec)
                            x_data = np.arange(len(y_data))

                        # Flatten if needed
                        if len(y_data.shape) > 1:
                            y_data = y_data.flatten()

                        # Apply offset
                        y_offset = y_data + (idx * offset)

                        # Find color index in original list
                        original_idx = selected_in_collection.index(item)
                        color = colors[original_idx]

                        # Plot
                        ax_current.plot(x_data, y_offset, color=color, linewidth=1.2, alpha=0.9, label=item['name'])

                        # Add text label - đặt gần cuối phổ, vào trong plot một chút
                        # Dùng 92% của chiều dài để text nằm hoàn toàn trong plot
                        label_idx = int(len(x_data) * 0.92)
                        # Thêm offset nhỏ để text cao hơn data một chút (10% của offset giữa các phổ)
                        text_y_offset = y_offset[label_idx] + (offset * 0.1)
                        ax_current.text(x_data[label_idx], text_y_offset, item['name'],
                                      fontsize=8, va='bottom', ha='right', color=color, fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

                    ax_current.set_title(titles[ax_idx], fontsize=12, fontweight='bold')
                    ax_current.set_xlabel("Wavenumber (cm⁻¹)")
                    ax_current.set_ylabel("Intensity (offset)")
                    ax_current.grid(True, alpha=0.2, axis='x')
                    ax_current.legend(loc='upper right', fontsize=8, framealpha=0.9)

                plt.tight_layout()

                # Download button
                if stack_mode == "So sánh cả 2":
                    plot_with_download(fig, "batch_stacked_comparison.png", "📥 Tải Stacked Plots")
                elif stack_mode == "Phổ gốc":
                    plot_with_download(fig, "batch_stacked_raw.png", "📥 Tải Stacked Plot")
                else:
                    plot_with_download(fig, "batch_stacked_preprocessed.png", "📥 Tải Stacked Plot")

            except Exception as e:
                st.error(f"Lỗi khi hiển thị stacked plots: {str(e)}")

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

    # Warning về N-FINDR compatibility
    with st.expander("⚠️ Lưu ý về Spectral Unmixing (N-FINDR)", expanded=False):
        st.warning("""
        **N-FINDR có thể gặp lỗi compatibility với scipy!**

        Nếu bạn gặp lỗi `module 'scipy.linalg' has no attribute '_flinalg'`,
        đây là vấn đề đã biết của RamanSPy với một số phiên bản scipy.

        **Khuyến nghị**: Sử dụng **PCA (Component Analysis)** thay thế -
        hoạt động tốt, ổn định, và cho kết quả tương tự!
        """)

    analysis_method = st.selectbox(
        "Phương pháp:",
        ["Component Analysis (PCA)", "Peak Detection", "Spectral Unmixing (N-FINDR)"]
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
                    try:
                        unmixer = rp.analysis.unmix.NFINDR(n_endmembers=n_endmembers)
                        abundance_maps, endmembers = unmixer.apply(data_to_analyze)
                    except AttributeError as ae:
                        if "_flinalg" in str(ae) or "scipy.linalg" in str(ae):
                            st.error("❌ Lỗi scipy compatibility với N-FINDR")
                            st.warning("""
                            **Nguyên nhân**: RamanSPy's N-FINDR implementation có vấn đề compatibility với phiên bản scipy hiện tại.

                            **Giải pháp**:
                            1. Sử dụng **PCA** thay vì N-FINDR cho việc phân tích thành phần
                            2. Hoặc cập nhật RamanSPy/scipy:
                               ```
                               pip install --upgrade ramanspy scipy
                               ```

                            **Khuyến nghị**: Sử dụng PCA (Component Analysis) - hoạt động tốt và ổn định hơn!
                            """)
                            st.stop()
                        else:
                            raise ae

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

        # Check if we have Collection with multiple spectra
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        has_collection = len(selected_in_collection) > 0

        # Options for multi-spectrum analysis
        if has_collection and len(selected_in_collection) > 1:
            st.info(f"📊 Phát hiện {len(selected_in_collection)} phổ trong Collection")

            col_mode1, col_mode2 = st.columns([2, 1])
            with col_mode1:
                peak_mode = st.radio(
                    "Chế độ phân tích:",
                    ["Phổ đơn lẻ", "So sánh tất cả phổ"],
                    horizontal=True,
                    help="Chọn phân tích một phổ riêng lẻ hoặc xem tất cả peaks cùng lúc"
                )

            if peak_mode == "Phổ đơn lẻ":
                spectrum_options = [item['name'] for item in selected_in_collection]
                selected_spectrum_name = st.selectbox(
                    "Chọn phổ để phân tích:",
                    spectrum_options
                )
                selected_spectrum_idx = spectrum_options.index(selected_spectrum_name)
            else:
                selected_spectrum_idx = None  # Analyze all
        else:
            peak_mode = "Phổ đơn lẻ"
            selected_spectrum_idx = 0
            if has_collection:
                st.info(f"📊 Phân tích phổ: {selected_in_collection[0]['name']}")

        col1, col2 = st.columns(2)

        with col1:
            prominence = st.slider("Prominence:", 0.01, 1.0, 0.1, 0.01)

        with col2:
            distance = st.slider("Distance (số điểm):", 5, 100, 20)

        if st.button("▶️ Tìm Peaks", type="primary"):
            try:
                from scipy.signal import find_peaks

                with st.spinner("Đang tìm peaks..."):
                    # Prepare results storage
                    all_peaks_results = []

                    # Determine which spectra to analyze
                    if has_collection:
                        if peak_mode == "So sánh tất cả phổ":
                            spectra_to_analyze = selected_in_collection
                        else:
                            spectra_to_analyze = [selected_in_collection[selected_spectrum_idx]]
                    else:
                        # No collection, use single spectrum
                        data_type = type(data_to_analyze).__name__
                        if data_type == 'Spectrum':
                            spectrum = data_to_analyze
                        elif hasattr(data_to_analyze, 'flat'):
                            spectrum = data_to_analyze.flat[0]
                        elif hasattr(data_to_analyze, '__len__') and len(data_to_analyze.shape) > 1:
                            spectrum = data_to_analyze[0]
                        else:
                            spectrum = data_to_analyze

                        spectra_to_analyze = [{'name': 'Phổ', 'data': spectrum}]

                    # Analyze each spectrum
                    for item in spectra_to_analyze:
                        spectrum = item['data']
                        spectrum_name = item['name']

                        # Use preprocessed if available
                        if 'preprocessed' in item and item['preprocessed'] is not None:
                            spectrum = item['preprocessed']

                        # Lấy intensities một cách an toàn
                        if hasattr(spectrum, 'spectral_data'):
                            intensities = spectrum.spectral_data
                        elif hasattr(spectrum, 'flat'):
                            intensities = spectrum.flat
                        elif isinstance(spectrum, np.ndarray):
                            intensities = spectrum
                        else:
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
                            spectral_axis = np.arange(len(intensities))

                        all_peaks_results.append({
                            'name': spectrum_name,
                            'spectrum': spectrum,
                            'peaks': peaks,
                            'properties': properties,
                            'intensities': intensities,
                            'spectral_axis': spectral_axis
                        })

                    st.session_state.analysis_results = {
                        'type': 'peaks',
                        'all_peaks': all_peaks_results,
                        'mode': peak_mode
                    }

                    total_peaks = sum(len(r['peaks']) for r in all_peaks_results)
                    if peak_mode == "So sánh tất cả phổ":
                        st.success(f"✅ Đã tìm thấy tổng cộng {total_peaks} peaks trong {len(all_peaks_results)} phổ!")
                    else:
                        st.success(f"✅ Đã tìm thấy {total_peaks} peaks!")

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

                    # Lấy tên phổ từ Collection nếu có
                    spectrum_names = []
                    if len(st.session_state.spectra_collection) > 0:
                        # Lấy selected items
                        selected_items = [s for s in st.session_state.spectra_collection if s['selected']]
                        if len(selected_items) > 0:
                            spectrum_names = [item['name'] for item in selected_items]

                    st.session_state.analysis_results = {
                        'type': 'pca',
                        'scores': scores,
                        'loadings': loadings,
                        'explained_variance': pca.explained_variance_ratio_,
                        'data': data_to_analyze,
                        'spectrum_names': spectrum_names,
                        'n_components': n_components
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
        fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

        if hasattr(data, 'spectral_axis'):
            rp.plot.spectra(endmembers, wavenumber_axis=data.spectral_axis, ax=ax1, plot_type='single stacked')
        else:
            rp.plot.spectra(endmembers, ax=ax1, plot_type='single stacked')

        ax1.set_title("Endmember Spectra")
        plot_with_download(fig1, "unmixing_endmembers.png", "📥 Tải Endmembers")

        # Plot abundance maps
        st.write("#### 🗺️ Abundance Maps")

        try:
            # Nếu là volumetric data, lấy một layer
            if len(abundance_maps[0].shape) == 3:
                layer_idx = st.slider("Chọn layer:", 0, abundance_maps[0].shape[2]-1, abundance_maps[0].shape[2]//2)

                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4), dpi=150)
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap[:, :, layer_idx], cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                plot_with_download(fig2, "unmixing_abundance_maps.png", "📥 Tải Abundance Maps")

            else:
                # 2D data
                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4), dpi=150)
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap, cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                plot_with_download(fig2, "unmixing_abundance_maps.png", "📥 Tải Abundance Maps")

        except Exception as e:
            st.warning(f"Không thể hiển thị abundance maps: {str(e)}")

    elif result_type == 'peaks':
        st.write("### Kết quả Peak Detection")

        all_peaks = results['all_peaks']
        mode = results.get('mode', 'Phổ đơn lẻ')

        # Single spectrum mode
        if mode == "Phổ đơn lẻ" or len(all_peaks) == 1:
            result = all_peaks[0]
            peaks = result['peaks']
            intensities = result['intensities']
            spectral_axis = result['spectral_axis']
            spectrum_name = result['name']

            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

            # Plot spectrum with name in legend
            ax.plot(spectral_axis, intensities, 'b-', linewidth=1.5, label=spectrum_name)
            ax.plot(spectral_axis[peaks], intensities[peaks], 'ro', markersize=8, label=f'Peaks ({len(peaks)})')

            # Đánh dấu peaks
            for peak in peaks:
                ax.axvline(spectral_axis[peak], color='r', linestyle='--', alpha=0.3)
                ax.text(spectral_axis[peak], intensities[peak], f'{spectral_axis[peak]:.0f}',
                       rotation=45, ha='right', va='bottom', fontsize=8)

            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Peak Detection: {spectrum_name} - {len(peaks)} peaks')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_with_download(fig, f"peak_detection_{spectrum_name}.png", "📥 Tải Peak Detection")

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

            # CSV Export for peak detection
            st.markdown("---")
            create_csv_download(df, f"peak_detection_{spectrum_name}.csv", "📥 Tải Peak Detection CSV")

        # Compare all spectra mode
        else:
            st.info(f"📊 So sánh peaks của {len(all_peaks)} phổ")

            # Plot all spectra with peaks
            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

            # Color palette
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_peaks)))

            for idx, result in enumerate(all_peaks):
                peaks = result['peaks']
                intensities = result['intensities']
                spectral_axis = result['spectral_axis']
                spectrum_name = result['name']
                color = colors[idx]

                # Plot spectrum
                ax.plot(spectral_axis, intensities, '-', linewidth=1.5,
                       color=color, label=f"{spectrum_name} ({len(peaks)} peaks)", alpha=0.7)

                # Plot peaks
                ax.plot(spectral_axis[peaks], intensities[peaks], 'o',
                       markersize=6, color=color)

            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Peak Detection - So sánh {len(all_peaks)} phổ')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plot_with_download(fig, "peak_detection_comparison.png", "📥 Tải Peak Comparison")

            # Bảng thông tin tổng hợp
            st.write("#### 📋 Tổng hợp Peaks")

            # Create combined table
            all_peak_data = []
            for result in all_peaks:
                spectrum_name = result['name']
                peaks = result['peaks']
                intensities = result['intensities']
                spectral_axis = result['spectral_axis']

                for i, peak in enumerate(peaks):
                    all_peak_data.append({
                        'Mẫu': spectrum_name,
                        'Peak #': i+1,
                        'Wavenumber (cm⁻¹)': f"{spectral_axis[peak]:.2f}",
                        'Intensity': f"{intensities[peak]:.4f}"
                    })

            import pandas as pd
            df_all = pd.DataFrame(all_peak_data)
            st.dataframe(df_all, use_container_width=True)

            # CSV Export
            st.markdown("---")
            create_csv_download(df_all, "peak_detection_all_spectra.csv", "📥 Tải tất cả Peaks CSV")

            # Summary statistics
            st.write("#### 📊 Thống kê")
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                total_peaks = sum(len(r['peaks']) for r in all_peaks)
                st.metric("Tổng số peaks", total_peaks)

            with col_stat2:
                avg_peaks = total_peaks / len(all_peaks)
                st.metric("Trung bình peaks/phổ", f"{avg_peaks:.1f}")

            with col_stat3:
                st.metric("Số phổ", len(all_peaks))

    elif result_type == 'pca':
        st.write("### Kết quả PCA")

        scores = results['scores']
        loadings = results['loadings']
        explained_variance = results['explained_variance']
        spectrum_names = results.get('spectrum_names', [])
        n_components = results.get('n_components', len(loadings))

        # Giải thích PCA
        with st.expander("ℹ️ PCA là gì? Cách đọc kết quả", expanded=False):
            st.markdown("""
            ### Principal Component Analysis (PCA)

            PCA giúp **giảm chiều dữ liệu** và tìm ra **sự khác biệt chính** giữa các phổ.

            #### 📊 **Explained Variance (Scree Plot)**
            - Cho biết mỗi PC "giải thích" bao nhiêu % sự biến thiên trong dữ liệu
            - PC1 thường có % cao nhất (ví dụ: 80%) → quan trọng nhất
            - PC2, PC3, PC4... giảm dần

            #### 🎯 **Score Plot**
            - Mỗi điểm = 1 phổ của bạn
            - Khoảng cách giữa các điểm = mức độ khác biệt giữa các phổ
            - Điểm gần nhau = phổ tương tự
            - Điểm xa nhau = phổ khác biệt

            #### 📈 **Loading Plot**
            - Loading = "Phổ đặc trưng" của mỗi PC
            - Peak cao trong loading plot = wavenumber quan trọng
            - Cho biết **vùng phổ nào** đóng góp vào sự khác biệt giữa các mẫu
            - Ví dụ: Peak cao ở 1000 cm⁻¹ trong PC1 loading → vùng 1000 cm⁻¹ là đặc trưng chính phân biệt các phổ
            """)

        col1, col2 = st.columns(2)

        with col1:
            # Scree plot
            st.write("#### 📊 Explained Variance")
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
            ax1.bar(range(1, len(explained_variance)+1), explained_variance * 100)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot')
            plot_with_download(fig1, "pca_scree_plot.png", "📥 Tải Scree Plot")

        with col2:
            # Score plot với màu sắc và legend
            st.write("#### 🎯 Score Plot")

            # Dropdown để chọn PC nào để plot
            if n_components >= 2:
                col_x, col_y = st.columns(2)
                with col_x:
                    pc_x = st.selectbox("Trục X:", [f"PC{i+1}" for i in range(n_components)], index=0, key="pc_x_select")
                    pc_x_idx = int(pc_x.replace("PC", "")) - 1
                with col_y:
                    pc_y = st.selectbox("Trục Y:", [f"PC{i+1}" for i in range(n_components)], index=1, key="pc_y_select")
                    pc_y_idx = int(pc_y.replace("PC", "")) - 1

                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)

                # Số phổ
                n_spectra = len(scores)
                colors = plt.cm.tab10(np.linspace(0, 1, n_spectra))

                # Plot từng điểm với màu riêng
                for i in range(n_spectra):
                    label = spectrum_names[i] if i < len(spectrum_names) else f'Phổ {i+1}'
                    ax2.scatter(scores[i, pc_x_idx], scores[i, pc_y_idx],
                               color=colors[i], s=100, alpha=0.8,
                               edgecolors='black', linewidth=1,
                               label=label)

                ax2.set_xlabel(f'{pc_x} ({explained_variance[pc_x_idx]*100:.1f}%)')
                ax2.set_ylabel(f'{pc_y} ({explained_variance[pc_y_idx]*100:.1f}%)')
                ax2.set_title(f'Score Plot: {pc_x} vs {pc_y}')
                ax2.legend(loc='best', fontsize=9, framealpha=0.9)
                ax2.grid(True, alpha=0.3)
                plot_with_download(fig2, f"pca_score_{pc_x}_vs_{pc_y}.png", "📥 Tải Score Plot")
            else:
                st.warning("Cần ít nhất 2 components để plot Score Plot")

        # Score plot matrix (tất cả các cặp PC)
        if n_components >= 3:
            st.markdown("---")
            show_matrix = st.checkbox("📊 Hiển thị Score Plot Matrix (tất cả các cặp PC)", value=False)

            if show_matrix:
                st.write("### 🎯 Score Plot Matrix")
                st.info("Ma trận này hiển thị tất cả các cặp PC có thể. Mỗi ô = 1 score plot với 2 PC khác nhau.")

                # Tính số plots
                n_plots = min(4, n_components)  # Tối đa 4 PCs để không quá nhiều plots
                fig_matrix, axes_matrix = plt.subplots(n_plots-1, n_plots-1, figsize=(4*(n_plots-1), 4*(n_plots-1)), dpi=150)

                n_spectra = len(scores)
                colors = plt.cm.tab10(np.linspace(0, 1, n_spectra))

                for i in range(n_plots-1):
                    for j in range(n_plots-1):
                        if j > i:
                            # Upper triangle - hide
                            if n_plots > 2:
                                axes_matrix[i, j].set_visible(False)
                        else:
                            # Lower triangle - plot
                            ax = axes_matrix[i, j] if n_plots > 2 else axes_matrix[i] if n_plots == 2 and j == 0 else axes_matrix

                            # PC indices: x = j+1, y = i+2
                            pc_x_idx = j
                            pc_y_idx = i + 1

                            # Plot each spectrum
                            for k in range(n_spectra):
                                label = spectrum_names[k] if k < len(spectrum_names) else f'Phổ {k+1}'
                                ax.scatter(scores[k, pc_x_idx], scores[k, pc_y_idx],
                                          color=colors[k], s=60, alpha=0.8,
                                          edgecolors='black', linewidth=0.5,
                                          label=label if i == 0 and j == 0 else "")

                            ax.set_xlabel(f'PC{pc_x_idx+1} ({explained_variance[pc_x_idx]*100:.1f}%)', fontsize=9)
                            ax.set_ylabel(f'PC{pc_y_idx+1} ({explained_variance[pc_y_idx]*100:.1f}%)', fontsize=9)
                            ax.grid(True, alpha=0.3)

                            # Legend chỉ ở plot đầu tiên
                            if i == 0 and j == 0:
                                ax.legend(loc='best', fontsize=7, framealpha=0.9)

                plt.tight_layout()
                plot_with_download(fig_matrix, "pca_score_matrix.png", "📥 Tải Score Matrix")

        # Loading plot - hiển thị TẤT CẢ components
        st.markdown("---")
        st.write(f"#### 📈 Loading Plots (tất cả {n_components} components)")
        st.info("💡 Loading plot cho biết wavenumber nào quan trọng trong mỗi PC. Peak cao = vùng phổ đặc trưng.")

        # Tính số hàng và cột cho subplot
        n_cols = min(3, n_components)  # Tối đa 3 cột
        n_rows = (n_components + n_cols - 1) // n_cols  # Làm tròn lên

        fig3, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=150)

        # Flatten axes để dễ iterate
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_components):
            ax = axes[i]
            ax.plot(loadings[i], linewidth=1.5, color=plt.cm.tab10(i/10))
            ax.set_title(f'PC{i+1} Loading ({explained_variance[i]*100:.1f}%)', fontweight='bold')
            ax.set_xlabel('Wavenumber index')
            ax.set_ylabel('Loading')
            ax.grid(True, alpha=0.3)

        # Ẩn các subplot trống nếu có
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_with_download(fig3, "pca_loadings.png", "📥 Tải Loading Plots")

        # CSV Export cho PCA
        st.markdown("---")
        st.write("### 📥 Tải dữ liệu PCA")

        col_csv1, col_csv2, col_csv3 = st.columns(3)

        with col_csv1:
            st.write("**Scores**")
            # Tạo DataFrame cho scores
            score_columns = [f'PC{i+1}' for i in range(n_components)]
            score_labels = [spectrum_names[i] if i < len(spectrum_names) else f'Phổ {i+1}' for i in range(len(scores))]
            scores_df = pd.DataFrame(scores, columns=score_columns, index=score_labels)
            scores_df.insert(0, 'Spectrum', score_labels)
            create_csv_download(scores_df, "pca_scores.csv", "📥 Tải Scores CSV")

        with col_csv2:
            st.write("**Loadings**")
            # Tạo DataFrame cho loadings
            loading_columns = [f'Feature_{i+1}' for i in range(loadings.shape[1])]
            loading_labels = [f'PC{i+1}' for i in range(n_components)]
            loadings_df = pd.DataFrame(loadings, columns=loading_columns, index=loading_labels)
            loadings_df.insert(0, 'Component', loading_labels)
            create_csv_download(loadings_df, "pca_loadings.csv", "📥 Tải Loadings CSV")

        with col_csv3:
            st.write("**Explained Variance**")
            # Tạo DataFrame cho explained variance
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                'Explained Variance (%)': explained_variance * 100,
                'Cumulative (%)': np.cumsum(explained_variance) * 100
            })
            create_csv_download(var_df, "pca_explained_variance.csv", "📥 Tải Variance CSV")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>RamanSPy GUI v1.0 | Được xây dựng với Streamlit</p>
    <p>Tài liệu: <a href='https://ramanspy.readthedocs.io'>ramanspy.readthedocs.io</a></p>
</div>
""", unsafe_allow_html=True)
