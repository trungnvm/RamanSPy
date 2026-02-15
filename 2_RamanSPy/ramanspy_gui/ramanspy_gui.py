"""
RamanSPy GUI - Graphical User Interface for Raman Spectral Analysis
Run app: streamlit run ramanspy_gui.py
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
import json

# ==================== LOCALIZATION SETUP ====================
def load_locales():
    """Load localization files"""
    locales = {}
    locale_dir = Path(__file__).parent / "locales"

    # Default fallback (Vietnamese)
    try:
        with open(locale_dir / "vi.json", "r", encoding="utf-8") as f:
            locales["vi"] = json.load(f)
    except Exception as e:
        st.error(f"Error loading Vietnamese locale: {e}")
        return {"vi": {}}

    # English
    try:
        with open(locale_dir / "en.json", "r", encoding="utf-8") as f:
            locales["en"] = json.load(f)
    except Exception as e:
        # st.warning(f"Error loading English locale: {e}")
        pass

    return locales

# Load locales once
LOCALES = load_locales()

def t(key_path):
    """
    Get translated string by dot-notation key path.
    Example: t("upload_page.header")
    Uses st.session_state.language to determine language.
    """
    lang = st.session_state.get('language', 'vi')
    data = LOCALES.get(lang, LOCALES.get('vi', {}))

    keys = key_path.split('.')
    val = data

    try:
        for k in keys:
            val = val[k]
        return val
    except (KeyError, TypeError):
        # Fallback to Vietnamese if key missing in current lang
        if lang != 'vi':
            val = LOCALES.get('vi', {})
            try:
                for k in keys:
                    val = val[k]
                return val
            except:
                pass
        return key_path # Return key if translation missing

# ==================== HELPER FUNCTIONS ====================
def plot_with_download(fig, filename="plot.png", download_label_key="upload_page.upload_tab.download_plot"):
    """Display plot with download button"""
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # Display plot
    st.pyplot(fig)

    # Download button
    st.download_button(
        label=t(download_label_key) if "." in download_label_key else download_label_key, # Check if key or literal
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

    plt.close(fig)

def create_csv_download(dataframe, filename="data.csv", label_key="upload_page.upload_tab.download_csv"):
    """Create download button for CSV data"""
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)

    st.download_button(
        label=t(label_key) if "." in label_key else label_key,
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def get_contrasting_colors(n):
    """
    Return array of high-contrast colors avoiding yellow and similar hues.
    Colors chosen for maximum distinction and visibility.
    """
    # High contrast color palette: Red, Blue, Green, Purple, Orange, Magenta, Brown, DarkGreen
    base_colors = [
        '#E41A1C',  # Bright Red
        '#377EB8',  # Blue
        '#4DAF4A',  # Green
        '#984EA3',  # Purple
        '#FF7F00',  # Orange (not yellow)
        '#F781BF',  # Pink/Magenta
        '#A65628',  # Brown
        '#006400',  # Dark Green
    ]

    if n <= len(base_colors):
        colors_hex = base_colors[:n]
    else:
        # Cycle through if more colors needed
        colors_hex = (base_colors * ((n // len(base_colors)) + 1))[:n]

    # Convert hex to RGB array for matplotlib
    colors_rgb = []
    for hex_color in colors_hex:
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        colors_rgb.append(rgb)

    return np.array(colors_rgb)

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
if 'language' not in st.session_state:
    st.session_state.language = 'vi' # Default language

# Sidebar - Language Selector & Navigation
with st.sidebar:
    # Language Selector
    lang_col1, lang_col2 = st.columns([1, 2])
    with lang_col1:
        st.write("🌐 Language:")
    with lang_col2:
        selected_lang = st.selectbox(
            "Language",
            options=['vi', 'en'],
            format_func=lambda x: "Tiếng Việt" if x == 'vi' else "English",
            index=0 if st.session_state.language == 'vi' else 1,
            label_visibility="collapsed"
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()

    st.title(t("menu_title"))
    page = st.radio(
        t("menu_select_label"),
        [t("menu_options.upload"), t("menu_options.preprocessing"), t("menu_options.analysis"), t("menu_options.visualization")]
    )

    st.markdown("---")
    st.info(t("sidebar_info"))

# Header
st.markdown(f'<p class="main-header">{t("main_header")}</p>', unsafe_allow_html=True)

# ==================== TRANG TẢI DỮ LIỆU ====================
if page == t("menu_options.upload"):
    st.markdown(f'<p class="sub-header">{t("upload_page.header")}</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(t("upload_page.tabs"))

    # Tab 1: Tải từ file
    with tab1:
        st.write(t("upload_page.upload_tab.title"))

        col1, col2 = st.columns([2, 1])

        with col1:
            file_format = st.selectbox(
                t("upload_page.upload_tab.format_label"),
                t("upload_page.upload_tab.formats")
            )

        with col2:
            st.info(t("upload_page.upload_tab.format_hint"))

        # Hướng dẫn cho CSV/Text
        # Note: Index 0 is CSV/Text in both languages
        if file_format == t("upload_page.upload_tab.formats")[0]:
            with st.expander(t("upload_page.upload_tab.csv_help_title")):
                st.write(t("upload_page.upload_tab.csv_help_content"))

        uploaded_files = st.file_uploader(
            t("upload_page.upload_tab.file_uploader_label"),
            type=['txt', 'csv', 'wdf', 'npy', 'npz', 'dat'],
            help=t("upload_page.upload_tab.file_uploader_help"),
            accept_multiple_files=True
        )

        # Checkbox để tự động thêm vào collection
        st.markdown("---")
        auto_add_to_collection = st.checkbox(
            t("upload_page.upload_tab.auto_add_checkbox"),
            value=True,
            help=t("upload_page.upload_tab.auto_add_help")
        )
        if auto_add_to_collection:
            st.caption(t("upload_page.upload_tab.auto_add_info"))
        else:
            st.caption(t("upload_page.upload_tab.auto_add_warning"))

        # Hiển thị số files đã được xử lý
        if len(st.session_state.processed_file_ids) > 0:
            st.info(t("upload_page.upload_tab.processed_info").format(count=len(st.session_state.processed_file_ids)))

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

                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        # Load dữ liệu theo định dạng
                        # Use indices to match logic regardless of translation
                        formats = t("upload_page.upload_tab.formats")

                        if file_format == formats[1]: # WITec
                            loaded_data = rp.load.witec(tmp_path)
                        elif file_format == formats[2]: # Renishaw
                            loaded_data = rp.load.renishaw(tmp_path)
                        elif file_format == formats[3]: # NumPy
                            data_array = np.load(tmp_path)
                            loaded_data = rp.Spectrum(data_array)
                        else: # CSV/Text (formats[0])
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
                                elif '\\t' in line:
                                    parts = line.split('\\t')
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
                                raise ValueError("Cannot read data from file.")

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
                st.success(t("upload_page.upload_tab.success_msg").format(count=loaded_count))
                if auto_add_to_collection:
                    st.info(t("upload_page.upload_tab.collection_info").format(count=loaded_count))

            if failed_files:
                st.error(t("upload_page.upload_tab.error_msg").format(count=len(failed_files)))
                for fname, error in failed_files:
                    st.write(f"- {fname}: {error}")

            # Hiển thị thông tin phổ cuối cùng được load
            if st.session_state.data is not None and loaded_count > 0:
                st.write(t("upload_page.upload_tab.last_spectrum_info"))
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(t("upload_page.upload_tab.metric_type"), type(st.session_state.data).__name__)
                with col2:
                    st.metric(t("upload_page.upload_tab.metric_size"), str(st.session_state.data.shape))
                with col3:
                    if hasattr(st.session_state.data, 'spectral_axis'):
                        st.metric(t("upload_page.upload_tab.metric_points"), len(st.session_state.data.spectral_axis))

    # Tab 2: Dữ liệu mẫu
    with tab2:
        st.write(t("upload_page.sample_tab.title"))

        st.info(t("upload_page.sample_tab.download_info"))

        dataset_type = st.selectbox(
            t("upload_page.sample_tab.dataset_label"),
            t("upload_page.sample_tab.datasets")
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button(t("upload_page.sample_tab.download_btn"), type="primary"):
                try:
                    with st.spinner(t("upload_page.sample_tab.downloading_spinner")):
                        data_dir = "./data/kallepitis_data"

                        # Chọn cell type
                        # Use indices to match datasets
                        datasets = t("upload_page.sample_tab.datasets")

                        if dataset_type == datasets[0]: # THP-1
                            cell_type = 'THP-1'
                        elif dataset_type == datasets[1]: # MCF-7
                            cell_type = 'MCF-7'
                        else:
                            cell_type = 'THP-1'

                        volumes = rp.datasets.volumetric_cells(cell_type=cell_type, folder=data_dir)
                        st.session_state.data = volumes[0]

                        st.success(t("upload_page.sample_tab.success_msg").format(dataset=dataset_type))
                        st.rerun()

                except Exception as e:
                    st.warning(t("upload_page.sample_tab.error_msg").format(error=str(e)))
                    st.info(t("upload_page.sample_tab.fallback_info"))

        with col2:
            datasets = t("upload_page.sample_tab.datasets")
            if dataset_type == datasets[0]:
                st.write(t("upload_page.sample_tab.thp1_desc"))
            elif dataset_type == datasets[1]:
                st.write(t("upload_page.sample_tab.mcf7_desc"))

    # Tab 3: Dữ liệu tổng hợp
    with tab3:
        st.write(t("upload_page.synthetic_tab.title"))

        col1, col2 = st.columns(2)

        with col1:
            n_spectra = st.slider(t("upload_page.synthetic_tab.n_spectra"), 10, 1000, 100)
            n_points = st.slider(t("upload_page.synthetic_tab.n_points"), 100, 2000, 500)

        with col2:
            noise_level = st.slider(t("upload_page.synthetic_tab.noise_level"), 0.0, 0.5, 0.1, 0.05)
            n_peaks = st.slider(t("upload_page.synthetic_tab.n_peaks"), 1, 10, 3)

        if st.button(t("upload_page.synthetic_tab.generate_btn"), type="primary"):
            with st.spinner(t("upload_page.synthetic_tab.generating_spinner")):
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

                st.success(t("upload_page.synthetic_tab.success_msg").format(n_spectra=n_spectra, n_points=n_points))

    # ==================== QUẢN LÝ NHIỀU PHỔ ====================
    st.markdown("---")
    with st.expander(t("collection.title"), expanded=False):
        st.write(t("collection.add_current_title"))

        col_name, col_add = st.columns([3, 1])

        with col_name:
            spectrum_name = st.text_input(
                t("collection.name_input"),
                value=f"Spectrum_{len(st.session_state.spectra_collection)+1}",
                help=t("collection.name_help")
            )

        with col_add:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button(t("collection.add_btn")):
                if st.session_state.data is not None:
                    # Kiểm tra trùng tên
                    existing_names = [s['name'] for s in st.session_state.spectra_collection]
                    if spectrum_name in existing_names:
                        st.error(t("collection.error_duplicate").format(name=spectrum_name))
                    else:
                        st.session_state.spectra_collection.append({
                            'name': spectrum_name,
                            'data': st.session_state.data,
                            'preprocessed': st.session_state.preprocessed_data,
                            'selected': True
                        })
                        st.success(t("collection.success_add").format(name=spectrum_name))
                        st.rerun()
                else:
                    st.warning(t("collection.warning_no_data"))

        # Hiển thị collection
        if len(st.session_state.spectra_collection) > 0:
            # Thống kê collection
            total_count = len(st.session_state.spectra_collection)
            preprocessed_count = sum(1 for s in st.session_state.spectra_collection if s['preprocessed'] is not None)

            st.write(t("collection.list_title").format(count=total_count))

            # Progress bar cho preprocessing status
            if total_count > 0:
                progress = preprocessed_count / total_count
                st.progress(progress, text=t("collection.progress_text").format(
                    processed=preprocessed_count, total=total_count, percent=int(progress*100)))

            st.markdown("")  # Spacing

            # Selection mode
            col_mode1, col_mode2, col_mode3, col_mode4 = st.columns(4)
            with col_mode1:
                if st.button(t("collection.select_all")):
                    for spec in st.session_state.spectra_collection:
                        spec['selected'] = True
                    st.rerun()
            with col_mode2:
                if st.button(t("collection.deselect_all")):
                    for spec in st.session_state.spectra_collection:
                        spec['selected'] = False
                    st.rerun()
            with col_mode3:
                if st.button(t("collection.delete_all")):
                    st.session_state.spectra_collection = []
                    st.session_state.processed_file_ids = set()
                    st.rerun()
            with col_mode4:
                if st.button(t("collection.reset_cache")):
                    st.session_state.processed_file_ids = set()
                    st.info(t("collection.cache_cleared"))
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
                        st.write(t("collection.file_label").format(name=original_name))
                    else:
                        st.write(t("collection.spectrum_label").format(index=i+1))

                with col3:
                    # Editable name
                    new_name = st.text_input(
                        t("collection.name_input"),
                        value=spec['name'],
                        key=f"name_{i}",
                        label_visibility="collapsed",
                        placeholder=t("collection.name_placeholder")
                    )
                    if new_name != spec['name'] and new_name.strip():
                        # Check duplicate
                        existing = [s['name'] for idx, s in enumerate(st.session_state.spectra_collection) if idx != i]
                        if new_name not in existing:
                            spec['name'] = new_name

                    # Status với màu sắc rõ ràng
                    data_shape = spec['data'].shape if hasattr(spec['data'], 'shape') else "N/A"
                    if spec['preprocessed'] is not None:
                        st.caption(t("collection.status_processed").format(shape=data_shape))
                    else:
                        st.caption(t("collection.status_raw").format(shape=data_shape))

                with col4:
                    if st.button("🗑️", key=f"del_{i}", help=t("collection.btn_delete_help")):
                        st.session_state.spectra_collection.pop(i)
                        st.rerun()

                with col5:
                    if st.button("👁️", key=f"view_{i}", help=t("collection.btn_view_help")):
                        st.session_state.data = spec['data']
                        st.session_state.preprocessed_data = spec['preprocessed']
                        st.success(t("collection.load_success").format(name=spec['name']))
                        st.rerun()

            # Actions
            st.markdown("---")
            selected_count = sum(1 for s in st.session_state.spectra_collection if s['selected'])

            # Hiển thị thống kê chi tiết
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric(t("collection.stat_selected"), f"{selected_count}/{total_count}")
            with col_stat2:
                selected_preprocessed = sum(1 for s in st.session_state.spectra_collection if s['selected'] and s['preprocessed'] is not None)
                st.metric(t("collection.stat_processed"), f"{selected_preprocessed}/{selected_count}")
            with col_stat3:
                selected_raw = selected_count - selected_preprocessed
                if selected_raw > 0:
                    st.metric(t("collection.stat_unprocessed"), selected_raw, delta=t("collection.delta_needed"), delta_color="off")
                else:
                    st.metric(t("collection.stat_unprocessed"), "0", delta="✓", delta_color="normal")

            st.markdown("")

            if selected_count > 0:
                # Batch preprocessing section
                st.write(t("collection.batch_title"))
                st.info(t("collection.batch_hint_title"))
                st.markdown(t("collection.batch_hint_list"))

                col_quick = st.columns(1)[0]
                with col_quick:
                    # Quick access button
                    if st.button(t("collection.goto_preprocessing_btn"), use_container_width=True, type="primary"):
                        st.session_state['show_batch_hint'] = True
                        st.info(t("collection.goto_preprocessing_msg"))

            elif selected_count == 1:
                st.info(t("collection.single_select_hint"))
        else:
            st.info(t("collection.empty_hint"))

    # Preview dữ liệu nếu đã load
    if st.session_state.data is not None:
        st.markdown("---")
        st.write(t("preview.title"))

        try:
            # Lấy một vài phổ để hiển thị
            data_type = type(st.session_state.data).__name__

            if data_type == 'Spectrum':
                # Spectrum đơn lẻ - plot với màu đẹp
                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

                if hasattr(st.session_state.data, 'spectral_axis') and hasattr(st.session_state.data, 'spectral_data'):
                    ax.plot(st.session_state.data.spectral_axis, st.session_state.data.spectral_data,
                           color='#1f77b4', linewidth=1.5)
                    ax.set_xlabel(t("preview.plot_xlabel"))
                    ax.set_ylabel(t("preview.plot_ylabel"))
                else:
                    rp.plot.spectra(st.session_state.data, ax=ax, plot_type='single')

                ax.set_title(t("preview.plot_title"))
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
                    spectrum_labels = [f'{t("collection.spectrum_label").format(index=i+1)}' for i in range(n_spectra)]

                colors = get_contrasting_colors(n_spectra)

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
                    label = spectrum_labels[i] if i < len(spectrum_labels) else f'{t("collection.spectrum_label").format(index=i+1)}'
                    ax.plot(x_data, y_data, color=colors[i], linewidth=1.5, alpha=0.7, label=label)

                ax.set_title(f'{t("preview.plot_title")} ({n_spectra})')
                ax.set_xlabel(t("preview.plot_xlabel"))
                ax.set_ylabel(t("preview.plot_ylabel"))
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            elif hasattr(st.session_state.data, 'flat'):
                # Volumetric data - plot 5 phổ đầu với màu khác nhau
                sample_spectra = st.session_state.data.flat[0:5]
                n_samples = len(sample_spectra)
                colors = get_contrasting_colors(n_samples)

                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                for i in range(n_samples):
                    spec = sample_spectra[i]
                    if hasattr(spec, 'spectral_axis') and hasattr(spec, 'spectral_data'):
                        ax.plot(spec.spectral_axis, spec.spectral_data,
                               color=colors[i], linewidth=1.5, alpha=0.7, label=f'{t("collection.spectrum_label").format(index=i+1)}')

                ax.set_title(f'{t("preview.plot_title")} (5)')
                ax.set_xlabel(t("preview.plot_xlabel"))
                ax.set_ylabel(t("preview.plot_ylabel"))
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            else:
                # Fallback
                sample_spectra = st.session_state.data
                fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
                rp.plot.spectra(sample_spectra, ax=ax, plot_type='single')
                ax.set_title(t("preview.plot_title"))
                ax.set_xlabel(t("preview.plot_xlabel"))
                ax.set_ylabel(t("preview.plot_ylabel"))
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(t("preview.error_msg").format(error=str(e)))
            st.info(f"Debug: Data type = {type(st.session_state.data).__name__}")

# ==================== TRANG TIỀN XỬ LÝ ====================
elif page == t("menu_options.preprocessing"):
    st.markdown(f'<p class="sub-header">{t("preprocessing_page.header")}</p>', unsafe_allow_html=True)

    # Hiển thị thông tin collection nếu có
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 0:
            preprocessed_in_selected = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)
            raw_in_selected = len(selected_in_collection) - preprocessed_in_selected

            if raw_in_selected > 0:
                st.info(t("preprocessing_page.collection_info").format(total=len(selected_in_collection), processed=preprocessed_in_selected, raw=raw_in_selected))
            else:
                st.success(t("preprocessing_page.collection_success").format(total=len(selected_in_collection)))

    if st.session_state.data is None:
        st.warning(t("preprocessing_page.warning_no_data"))
        st.stop()

    st.write(t("preprocessing_page.pipeline_title"))

    # Sidebar cho việc chọn các bước preprocessing
    st.write(t("preprocessing_page.steps_label"))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write(t("preprocessing_page.step1"))
        use_cropping = st.checkbox(t("preprocessing_page.use_cropping"), value=True)
        if use_cropping:
            crop_min = st.number_input("Wavenumber min (cm⁻¹):", 0, 4000, 700, 50)
            crop_max = st.number_input("Wavenumber max (cm⁻¹):", 0, 4000, 1800, 50)

    with col2:
        st.write(t("preprocessing_page.step2"))
        use_despike = st.checkbox(t("preprocessing_page.use_despike"), value=True)
        if use_despike:
            st.write(t("preprocessing_page.despike_method"))
            with st.expander(t("preprocessing_page.despike_params")):
                despike_kernel = st.slider(t("preprocessing_page.kernel_size"), 1, 9, 3, 2, help=t("preprocessing_page.kernel_help"))
                despike_threshold = st.slider(t("preprocessing_page.threshold"), 1.0, 20.0, 8.0, 1.0, help=t("preprocessing_page.threshold_help"))

    col3, col4 = st.columns([1, 1])

    with col3:
        st.write(t("preprocessing_page.step3"))
        use_denoise = st.checkbox(t("preprocessing_page.use_denoise"), value=True)
        if use_denoise:
            denoise_method = st.selectbox(
                t("preprocessing_page.denoise_method_label"),
                ["SavGol", "Gaussian"]
            )

            if denoise_method == "SavGol":
                window_length = st.slider("Window length:", 3, 21, 9, 2)
                polyorder = st.slider("Polynomial order:", 1, 5, 3)
            elif denoise_method == "Gaussian":
                sigma = st.slider("Sigma:", 0.5, 5.0, 1.0, 0.5)

    with col4:
        st.write(t("preprocessing_page.step4"))
        use_baseline = st.checkbox(t("preprocessing_page.use_baseline"), value=True)
        if use_baseline:
            baseline_method = st.selectbox(
                t("preprocessing_page.baseline_method_label"),
                ["ASPLS", "ASLS", "Poly"]
            )

            if baseline_method == "Poly":
                poly_order = st.slider("Polynomial order:", 1, 5, 3)

    col5, col6 = st.columns([1, 1])

    with col5:
        st.write(t("preprocessing_page.step5"))
        use_normalize = st.checkbox(t("preprocessing_page.use_normalize"), value=True)
        if use_normalize:
            normalize_method = st.selectbox(
                t("preprocessing_page.normalize_method_label"),
                ["MinMax", "AUC", "Vector", "SNV"]
            )

    # Nút áp dụng pipeline
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        if st.button(t("preprocessing_page.apply_btn"), type="primary", use_container_width=True):
            try:
                with st.spinner(t("preprocessing_page.processing_spinner")):
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

                    st.success(t("preprocessing_page.success_msg").format(count=len(steps)))

            except Exception as e:
                st.error(t("preprocessing_page.error_msg").format(error=str(e)))

    with col_btn2:
        if st.button(t("preprocessing_page.reset_btn"), use_container_width=True):
            st.session_state.preprocessed_data = None
            st.session_state.pipeline_steps = []
            st.info(t("preprocessing_page.reset_msg"))

    with col_btn3:
        # Batch preprocessing for collection
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 0:
            button_label = t("preprocessing_page.batch_apply_btn").format(count=len(selected_in_collection))
            if st.button(button_label, use_container_width=True, type="primary"):
                try:
                    with st.spinner(t("preprocessing_page.batch_spinner").format(count=len(selected_in_collection))):
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
                        progress_bar = st.progress(0, text=t("preprocessing_page.batch_progress_start"))

                        for idx, item in enumerate(st.session_state.spectra_collection):
                            if item['selected']:
                                try:
                                    progress = (idx + 1) / len(selected_in_collection)
                                    progress_bar.progress(progress, text=t("preprocessing_page.batch_progress_item").format(name=item['name'], current=idx+1, total=len(selected_in_collection)))

                                    item['preprocessed'] = pipeline.apply(item['data'])
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"Error processing '{item['name']}': {str(e)}")

                        progress_bar.progress(1.0, text=t("preprocessing_page.batch_progress_done"))

                        st.success(t("preprocessing_page.batch_success").format(success=success_count, total=len(selected_in_collection), steps=len(steps)))

                except Exception as e:
                    st.error(t("preprocessing_page.error_msg").format(error=str(e)))
        else:
            st.info(t("preprocessing_page.batch_hint"))

    # Combine section - Kết hợp phổ để phân tích
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        if len(selected_in_collection) > 1:
            st.markdown("---")
            st.write(t("preprocessing_page.combine_title"))

            col_combine1, col_combine2 = st.columns([2, 1])

            # Calculate stats
            selected_preprocessed = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)
            selected_raw = len(selected_in_collection) - selected_preprocessed

            with col_combine1:
                # Hiển thị warning nếu có phổ chưa xử lý
                if selected_raw > 0:
                    st.warning(t("preprocessing_page.combine_warning").format(count=selected_raw))
                else:
                    st.success(t("preprocessing_page.combine_success_check").format(count=len(selected_in_collection)))
                    st.info(t("preprocessing_page.combine_info"))

            with col_combine2:
                # Combine spectra button
                if st.button(t("preprocessing_page.combine_btn"), type="primary", use_container_width=True, key="combine_preprocessing"):
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
                            st.success(t("preprocessing_page.combine_success_preprocessed").format(count=len(selected_in_collection)))
                        else:
                            st.success(t("preprocessing_page.combine_success_raw").format(count=len(selected_in_collection)))
                            st.warning(t("preprocessing_page.combine_warning_raw"))

                        st.info(t("preprocessing_page.combine_goto_analysis"))
                        st.rerun()

                    except Exception as e:
                        st.error(t("preprocessing_page.combine_error").format(error=str(e)))

    # So sánh trước/sau
    if st.session_state.preprocessed_data is not None:
        st.markdown("---")
        st.write(t("preprocessing_page.compare_title"))

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
                st.write(t("preprocessing_page.before_label"))
                fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(raw_spectrum, 'spectral_axis') and hasattr(raw_spectrum, 'spectral_data'):
                    ax1.plot(raw_spectrum.spectral_axis, raw_spectrum.spectral_data, linewidth=1.5)
                    ax1.set_xlabel(t("preview.plot_xlabel"))
                    ax1.set_ylabel(t("preview.plot_ylabel"))
                else:
                    rp.plot.spectra(raw_spectrum, ax=ax1)

                ax1.set_title(t("preprocessing_page.before_plot_title"))
                ax1.grid(True, alpha=0.3)
                plot_with_download(fig1, "raw_spectrum.png", "preprocessing_page.download_raw_plot")

            with col_after:
                st.write(t("preprocessing_page.after_label"))
                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)

                # Plot trực tiếp với matplotlib để tránh lỗi indexing
                if hasattr(processed_spectrum, 'spectral_axis') and hasattr(processed_spectrum, 'spectral_data'):
                    ax2.plot(processed_spectrum.spectral_axis, processed_spectrum.spectral_data, linewidth=1.5)
                    ax2.set_xlabel(t("preview.plot_xlabel"))
                    ax2.set_ylabel(t("preview.plot_ylabel"))
                else:
                    rp.plot.spectra(processed_spectrum, ax=ax2)

                ax2.set_title(t("preprocessing_page.after_plot_title"))
                ax2.grid(True, alpha=0.3)
                plot_with_download(fig2, "preprocessed_spectrum.png", "preprocessing_page.download_proc_plot")

        except Exception as e:
            st.error(t("preprocessing_page.compare_error").format(error=str(e)))

        # CSV Export cho preprocessing data
        st.markdown("---")
        st.write(t("preprocessing_page.export_title"))

        col_csv1, col_csv2 = st.columns(2)

        with col_csv1:
            st.write(t("preprocessing_page.export_raw_label"))
            if hasattr(raw_spectrum, 'spectral_axis') and hasattr(raw_spectrum, 'spectral_data'):
                raw_df = pd.DataFrame({
                    'Wavenumber (cm⁻¹)': raw_spectrum.spectral_axis,
                    'Intensity': raw_spectrum.spectral_data
                })
                create_csv_download(raw_df, "raw_spectrum.csv", "preprocessing_page.export_raw_csv")
            else:
                st.info(t("preprocessing_page.export_unavailable"))

        with col_csv2:
            st.write(t("preprocessing_page.export_proc_label"))
            if hasattr(processed_spectrum, 'spectral_axis') and hasattr(processed_spectrum, 'spectral_data'):
                processed_df = pd.DataFrame({
                    'Wavenumber (cm⁻¹)': processed_spectrum.spectral_axis,
                    'Intensity': processed_spectrum.spectral_data
                })
                create_csv_download(processed_df, "preprocessed_spectrum.csv", "preprocessing_page.export_proc_csv")
            else:
                st.info(t("preprocessing_page.export_unavailable"))

    # Overlay plots cho batch preprocessing
    if len(st.session_state.spectra_collection) > 0:
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        preprocessed_count = sum(1 for s in selected_in_collection if s['preprocessed'] is not None)

        if len(selected_in_collection) > 1 and preprocessed_count > 0:
            st.markdown("---")
            st.write(t("preprocessing_page.overlay_title"))
            st.info(t("preprocessing_page.overlay_info").format(total=len(selected_in_collection), processed=preprocessed_count))

            try:
                # Tạo figure với 2 subplots
                fig, (ax_raw, ax_processed) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

                # High contrast color palette
                colors = get_contrasting_colors(len(selected_in_collection))

                # Plot raw spectra (bên trái)
                ax_raw.set_title(t("preprocessing_page.overlay_raw_title"), fontsize=12, fontweight='bold')
                ax_raw.set_xlabel(t("preview.plot_xlabel"))
                ax_raw.set_ylabel(t("preview.plot_ylabel"))
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
                ax_processed.set_title(t("preprocessing_page.overlay_proc_title"), fontsize=12, fontweight='bold')
                ax_processed.set_xlabel(t("preview.plot_xlabel"))
                ax_processed.set_ylabel(t("preview.plot_ylabel"))
                ax_processed.grid(True, alpha=0.3)

                preprocessed_items = [item for item in selected_in_collection if item['preprocessed'] is not None]

                if len(preprocessed_items) > 0:
                    # High contrast colors for preprocessed items
                    proc_colors = get_contrasting_colors(len(preprocessed_items))

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
                    ax_processed.text(0.5, 0.5, t("preprocessing_page.overlay_empty"),
                                     ha='center', va='center', transform=ax_processed.transAxes,
                                     fontsize=12, style='italic')

                plt.tight_layout()
                plot_with_download(fig, "batch_preprocessing_overlay.png", "preprocessing_page.download_overlay")

                # CSV Export cho batch preprocessing
                st.markdown("---")
                st.write(t("preprocessing_page.batch_export_title"))

                col_batch1, col_batch2 = st.columns(2)

                with col_batch1:
                    st.write(t("preprocessing_page.batch_export_raw_label"))
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
                            create_csv_download(raw_batch_df, "batch_raw_spectra.csv", "preprocessing_page.batch_export_raw_csv")
                        else:
                            st.info(t("preprocessing_page.export_unavailable"))
                    except Exception as e:
                        st.warning(f"Error export: {str(e)}")

                with col_batch2:
                    st.write(t("preprocessing_page.batch_export_proc_label"))
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
                                create_csv_download(proc_batch_df, "batch_preprocessed_spectra.csv", "preprocessing_page.batch_export_proc_csv")
                            else:
                                st.info(t("preprocessing_page.export_unavailable"))
                        except Exception as e:
                            st.warning(f"Error export: {str(e)}")
                    else:
                        st.info(t("preprocessing_page.overlay_empty"))

            except Exception as e:
                st.error(t("preprocessing_page.compare_error").format(error=str(e)))

        # Stacked plots cho batch preprocessing
        if len(selected_in_collection) > 1:
            st.markdown("---")
            st.write(t("preprocessing_page.stacked_title"))
            st.info(t("preprocessing_page.stacked_info"))

            # Options
            col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1])

            with col_opt1:
                # Use indices to match modes
                modes = t("preprocessing_page.stacked_modes")
                stack_mode = st.radio(
                    t("preprocessing_page.stacked_mode_label"),
                    modes,
                    horizontal=True
                )

            with col_opt2:
                offset_multiplier = st.slider(
                    t("preprocessing_page.stacked_offset_label"),
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help=t("preprocessing_page.stacked_offset_help")
                )

            with col_opt3:
                reverse_order = st.checkbox(t("preprocessing_page.stacked_reverse"), value=False)

            try:
                # Compare both (modes[2])
                if stack_mode == modes[2]:
                    fig, (ax_raw_stack, ax_proc_stack) = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
                    axes_list = [ax_raw_stack, ax_proc_stack]
                    titles = [t("preprocessing_page.stacked_raw_title"), t("preprocessing_page.stacked_proc_title")]
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
                    axes_list = [ax]
                    # Title based on mode
                    titles = [f"{stack_mode} (Stacked)"]

                colors = get_contrasting_colors(len(selected_in_collection))

                # Determine order
                items_to_plot = list(reversed(selected_in_collection)) if reverse_order else selected_in_collection

                for ax_idx, ax_current in enumerate(axes_list):
                    # Determine which data to use
                    # Raw (modes[0]) or Compare both and first ax
                    if stack_mode == modes[0] or (stack_mode == modes[2] and ax_idx == 0):
                        use_raw = True
                    else:
                        use_raw = False

                    # Filter items based on data availability
                    if use_raw:
                        items = items_to_plot
                    else:
                        items = [item for item in items_to_plot if item['preprocessed'] is not None]

                    if len(items) == 0:
                        ax_current.text(0.5, 0.5, t("preprocessing_page.stacked_no_data"),
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
                    ax_current.set_xlabel(t("preview.plot_xlabel"))
                    ax_current.set_ylabel("Intensity (offset)")
                    ax_current.grid(True, alpha=0.2, axis='x')
                    ax_current.legend(loc='upper right', fontsize=8, framealpha=0.9)

                plt.tight_layout()

                # Download button
                if stack_mode == modes[2]:
                    plot_with_download(fig, "batch_stacked_comparison.png", "preprocessing_page.download_stacked_both")
                elif stack_mode == modes[0]:
                    plot_with_download(fig, "batch_stacked_raw.png", "preprocessing_page.download_stacked_single")
                else:
                    plot_with_download(fig, "batch_stacked_preprocessed.png", "preprocessing_page.download_stacked_single")

            except Exception as e:
                st.error(t("preprocessing_page.compare_error").format(error=str(e)))

# ==================== TRANG PHÂN TÍCH ====================
elif page == t("menu_options.analysis"):
    st.markdown(f'<p class="sub-header">{t("analysis_page.header")}</p>', unsafe_allow_html=True)

    # Kiểm tra dữ liệu
    if st.session_state.data is None:
        st.warning(t("analysis_page.warning_no_data"))
        st.stop()

    # Sử dụng dữ liệu đã xử lý nếu có, không thì dùng dữ liệu gốc
    data_to_analyze = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.data

    if st.session_state.preprocessed_data is None:
        st.info(t("analysis_page.info_using_raw"))

    # Debug info
    st.write(t("analysis_page.data_type_info").format(type=type(data_to_analyze).__name__))

    st.write(t("analysis_page.method_title"))

    # Warning về N-FINDR compatibility
    with st.expander(t("analysis_page.nfindr_warning_title"), expanded=False):
        st.warning(t("analysis_page.nfindr_warning_content"))

    analysis_method = st.selectbox(
        t("analysis_page.method_label"),
        t("analysis_page.methods")
    )

    methods = t("analysis_page.methods")

    # Spectral Unmixing
    if analysis_method == methods[2]: # N-FINDR
        st.write(t("analysis_page.nfindr.title"))
        st.write(t("analysis_page.nfindr.desc"))

        col1, col2 = st.columns([2, 1])

        with col1:
            n_endmembers = st.slider(t("analysis_page.nfindr.n_endmembers"), 2, 10, 5)

        with col2:
            st.info(t("analysis_page.nfindr.info"))

        if st.button(t("analysis_page.nfindr.run_btn"), type="primary"):
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
                        st.error(t("analysis_page.nfindr.error_single_spectrum"))
                        st.info(t("analysis_page.nfindr.hint_single_spectrum"))
                        st.stop()

                with st.spinner(t("analysis_page.nfindr.spinner")):
                    try:
                        unmixer = rp.analysis.unmix.NFINDR(n_endmembers=n_endmembers)
                        abundance_maps, endmembers = unmixer.apply(data_to_analyze)
                    except AttributeError as ae:
                        if "_flinalg" in str(ae) or "scipy.linalg" in str(ae):
                            st.error(t("analysis_page.nfindr.error_scipy"))
                            st.warning(t("analysis_page.nfindr_warning_content"))
                            st.stop()
                        else:
                            raise ae

                    st.session_state.analysis_results = {
                        'type': 'unmixing',
                        'abundance_maps': abundance_maps,
                        'endmembers': endmembers,
                        'data': data_to_analyze
                    }

                    st.success(t("analysis_page.nfindr.success").format(n=n_endmembers))

            except Exception as e:
                st.error(t("analysis_page.nfindr.error").format(error=str(e)))

    # Peak Detection
    elif analysis_method == methods[1]: # Peak Detection
        st.write(t("analysis_page.peak_detection.title"))
        st.write(t("analysis_page.peak_detection.desc"))

        # Check if we have Collection with multiple spectra
        selected_in_collection = [s for s in st.session_state.spectra_collection if s['selected']]
        has_collection = len(selected_in_collection) > 0

        # Options for multi-spectrum analysis
        if has_collection and len(selected_in_collection) > 1:
            st.info(t("analysis_page.peak_detection.collection_info").format(count=len(selected_in_collection)))

            col_mode1, col_mode2 = st.columns([2, 1])
            with col_mode1:
                modes = t("analysis_page.peak_detection.modes")
                peak_mode = st.radio(
                    t("analysis_page.peak_detection.mode_label"),
                    modes,
                    horizontal=True,
                    help=t("analysis_page.peak_detection.mode_help")
                )

            if peak_mode == modes[0]: # Single
                spectrum_options = [item['name'] for item in selected_in_collection]
                selected_spectrum_name = st.selectbox(
                    t("analysis_page.peak_detection.select_spectrum"),
                    spectrum_options
                )
                selected_spectrum_idx = spectrum_options.index(selected_spectrum_name)
            else:
                selected_spectrum_idx = None  # Analyze all
        else:
            # Match default single mode string
            peak_mode = t("analysis_page.peak_detection.modes")[0]
            selected_spectrum_idx = 0
            if has_collection:
                st.info(t("analysis_page.peak_detection.single_info").format(name=selected_in_collection[0]['name']))

        col1, col2 = st.columns(2)

        with col1:
            prominence = st.slider(t("analysis_page.peak_detection.prominence"), 0.01, 1.0, 0.1, 0.01)

        with col2:
            distance = st.slider(t("analysis_page.peak_detection.distance"), 5, 100, 20)

        if st.button(t("analysis_page.peak_detection.run_btn"), type="primary"):
            try:
                from scipy.signal import find_peaks

                with st.spinner(t("analysis_page.peak_detection.spinner")):
                    # Prepare results storage
                    all_peaks_results = []

                    # Determine which spectra to analyze
                    # Get translated modes again for comparison
                    modes = t("analysis_page.peak_detection.modes")

                    if has_collection:
                        if peak_mode == modes[1]: # Compare All
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

                        spectra_to_analyze = [{'name': 'Spectrum', 'data': spectrum}]

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
                    if peak_mode == modes[1]: # Compare All
                        st.success(t("analysis_page.peak_detection.success_all").format(count=total_peaks, spectra=len(all_peaks_results)))
                    else:
                        st.success(t("analysis_page.peak_detection.success_single").format(count=total_peaks))

            except Exception as e:
                st.error(t("analysis_page.peak_detection.error").format(error=str(e)))

    # PCA
    elif analysis_method == methods[0]: # PCA
        st.write(t("analysis_page.pca.title"))
        st.write(t("analysis_page.pca.desc"))

        n_components = st.slider(t("analysis_page.pca.n_components"), 2, 10, 3)

        if st.button(t("analysis_page.pca.run_btn"), type="primary"):
            try:
                from sklearn.decomposition import PCA

                with st.spinner(t("analysis_page.pca.spinner")):
                    # Chuẩn bị dữ liệu
                    data_type = type(data_to_analyze).__name__

                    if data_type == 'Spectrum':
                        # Spectrum đơn lẻ - PCA cần ít nhất 2 mẫu
                        st.error(t("analysis_page.pca.error_single"))
                        st.info(t("analysis_page.pca.hint_single"))
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
                        st.error(t("analysis_page.pca.error_less_than_2"))
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

                    st.success(t("analysis_page.pca.success").format(n=n_components))

            except Exception as e:
                st.error(t("analysis_page.pca.error").format(error=str(e)))

# ==================== TRANG TRỰC QUAN HÓA ====================
elif page == t("menu_options.visualization"):
    st.markdown(f'<p class="sub-header">{t("visualization_page.header")}</p>', unsafe_allow_html=True)

    if st.session_state.analysis_results is None:
        st.warning(t("visualization_page.warning_no_result"))
        st.stop()

    results = st.session_state.analysis_results
    result_type = results['type']

    # Hiển thị theo loại phân tích
    if result_type == 'unmixing':
        st.write(t("visualization_page.unmixing.title"))

        endmembers = results['endmembers']
        abundance_maps = results['abundance_maps']
        data = results['data']

        # Plot endmembers
        st.write(t("visualization_page.unmixing.endmembers_title"))
        fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

        if hasattr(data, 'spectral_axis'):
            rp.plot.spectra(endmembers, wavenumber_axis=data.spectral_axis, ax=ax1, plot_type='single stacked')
        else:
            rp.plot.spectra(endmembers, ax=ax1, plot_type='single stacked')

        ax1.set_title("Endmember Spectra")
        plot_with_download(fig1, "unmixing_endmembers.png", "visualization_page.unmixing.download_endmembers")

        # Plot abundance maps
        st.write(t("visualization_page.unmixing.maps_title"))

        try:
            # Nếu là volumetric data, lấy một layer
            if len(abundance_maps[0].shape) == 3:
                layer_idx = st.slider(t("visualization_page.unmixing.layer_slider"), 0, abundance_maps[0].shape[2]-1, abundance_maps[0].shape[2]//2)

                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4), dpi=150)
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap[:, :, layer_idx], cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                plot_with_download(fig2, "unmixing_abundance_maps.png", "visualization_page.unmixing.download_maps")

            else:
                # 2D data
                fig2, axes = plt.subplots(1, len(abundance_maps), figsize=(4*len(abundance_maps), 4), dpi=150)
                if len(abundance_maps) == 1:
                    axes = [axes]

                for i, (amap, ax) in enumerate(zip(abundance_maps, axes)):
                    im = ax.imshow(amap, cmap='viridis')
                    ax.set_title(f"Endmember {i+1}")
                    plt.colorbar(im, ax=ax)

                plot_with_download(fig2, "unmixing_abundance_maps.png", "visualization_page.unmixing.download_maps")

        except Exception as e:
            st.warning(t("visualization_page.unmixing.error_maps").format(error=str(e)))

    elif result_type == 'peaks':
        st.write(t("visualization_page.peaks.title"))

        all_peaks = results['all_peaks']
        mode = results.get('mode', '')

        # Use modes from translation to compare
        modes = t("analysis_page.peak_detection.modes")

        # Single spectrum mode
        if mode == modes[0] or len(all_peaks) == 1:
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

            ax.set_xlabel(t("preview.plot_xlabel"))
            ax.set_ylabel(t("preview.plot_ylabel"))
            ax.set_title(f'Peak Detection: {spectrum_name} - {len(peaks)} peaks')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_with_download(fig, f"peak_detection_{spectrum_name}.png", "visualization_page.peaks.download_plot")

            # Bảng thông tin peaks
            st.write(t("visualization_page.peaks.list_title"))
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
            create_csv_download(df, f"peak_detection_{spectrum_name}.csv", "visualization_page.peaks.download_csv")

        # Compare all spectra mode
        else:
            st.info(t("visualization_page.peaks.compare_info").format(count=len(all_peaks)))

            # Plot all spectra with peaks
            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

            # High contrast color palette
            colors = get_contrasting_colors(len(all_peaks))

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

            ax.set_xlabel(t("preview.plot_xlabel"))
            ax.set_ylabel(t("preview.plot_ylabel"))
            ax.set_title(t("visualization_page.peaks.compare_title").format(count=len(all_peaks)))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plot_with_download(fig, "peak_detection_comparison.png", "visualization_page.peaks.download_compare_plot")

            # Bảng thông tin tổng hợp
            st.write(t("visualization_page.peaks.summary_title"))

            # Create combined table
            all_peak_data = []
            for result in all_peaks:
                spectrum_name = result['name']
                peaks = result['peaks']
                intensities = result['intensities']
                spectral_axis = result['spectral_axis']

                for i, peak in enumerate(peaks):
                    all_peak_data.append({
                        'Sample': spectrum_name,
                        'Peak #': i+1,
                        'Wavenumber (cm⁻¹)': f"{spectral_axis[peak]:.2f}",
                        'Intensity': f"{intensities[peak]:.4f}"
                    })

            import pandas as pd
            df_all = pd.DataFrame(all_peak_data)
            st.dataframe(df_all, use_container_width=True)

            # CSV Export
            st.markdown("---")
            create_csv_download(df_all, "peak_detection_all_spectra.csv", "visualization_page.peaks.download_all_csv")

            # Summary statistics
            st.write(t("visualization_page.peaks.stats_title"))
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                total_peaks = sum(len(r['peaks']) for r in all_peaks)
                st.metric(t("visualization_page.peaks.total_peaks"), total_peaks)

            with col_stat2:
                avg_peaks = total_peaks / len(all_peaks)
                st.metric(t("visualization_page.peaks.avg_peaks"), f"{avg_peaks:.1f}")

            with col_stat3:
                st.metric(t("visualization_page.peaks.n_spectra"), len(all_peaks))

    elif result_type == 'pca':
        st.write(t("visualization_page.pca.title"))

        scores = results['scores']
        loadings = results['loadings']
        explained_variance = results['explained_variance']
        spectrum_names = results.get('spectrum_names', [])
        n_components = results.get('n_components', len(loadings))

        # Giải thích PCA
        with st.expander(t("visualization_page.pca.info_title"), expanded=False):
            st.markdown(t("visualization_page.pca.info_content"))

        col1, col2 = st.columns(2)

        with col1:
            # Scree plot
            st.write(t("visualization_page.pca.scree_title"))
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
            ax1.bar(range(1, len(explained_variance)+1), explained_variance * 100)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot')
            plot_with_download(fig1, "pca_scree_plot.png", "visualization_page.pca.download_scree")

        with col2:
            # Score plot với màu sắc và legend
            st.write(t("visualization_page.pca.score_title"))

            # Dropdown để chọn PC nào để plot
            if n_components >= 2:
                col_x, col_y = st.columns(2)
                with col_x:
                    pc_x = st.selectbox("X Axis:", [f"PC{i+1}" for i in range(n_components)], index=0, key="pc_x_select")
                    pc_x_idx = int(pc_x.replace("PC", "")) - 1
                with col_y:
                    pc_y = st.selectbox("Y Axis:", [f"PC{i+1}" for i in range(n_components)], index=1, key="pc_y_select")
                    pc_y_idx = int(pc_y.replace("PC", "")) - 1

                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)

                # High contrast colors for spectra
                n_spectra = len(scores)
                colors = get_contrasting_colors(n_spectra)

                # Plot từng điểm với màu riêng
                for i in range(n_spectra):
                    label = spectrum_names[i] if i < len(spectrum_names) else f'{t("collection.spectrum_label").format(index=i+1)}'
                    ax2.scatter(scores[i, pc_x_idx], scores[i, pc_y_idx],
                               color=colors[i], s=100, alpha=0.8,
                               edgecolors='black', linewidth=1,
                               label=label)

                ax2.set_xlabel(f'{pc_x} ({explained_variance[pc_x_idx]*100:.1f}%)')
                ax2.set_ylabel(f'{pc_y} ({explained_variance[pc_y_idx]*100:.1f}%)')
                ax2.set_title(f'Score Plot: {pc_x} vs {pc_y}')
                ax2.legend(loc='best', fontsize=9, framealpha=0.9)
                ax2.grid(True, alpha=0.3)
                plot_with_download(fig2, f"pca_score_{pc_x}_vs_{pc_y}.png", "visualization_page.pca.download_score")
            else:
                st.warning(t("visualization_page.pca.warning_components"))

        # Score plot matrix (tất cả các cặp PC)
        if n_components >= 3:
            st.markdown("---")
            show_matrix = st.checkbox(t("visualization_page.pca.matrix_checkbox"), value=False)

            if show_matrix:
                st.write(t("visualization_page.pca.matrix_title"))
                st.info(t("visualization_page.pca.matrix_info"))

                # Tính số plots
                n_plots = min(4, n_components)  # Tối đa 4 PCs để không quá nhiều plots
                fig_matrix, axes_matrix = plt.subplots(n_plots-1, n_plots-1, figsize=(4*(n_plots-1), 4*(n_plots-1)), dpi=150)

                n_spectra = len(scores)
                colors = get_contrasting_colors(n_spectra)

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
                                label = spectrum_names[k] if k < len(spectrum_names) else f'{t("collection.spectrum_label").format(index=k+1)}'
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
                plot_with_download(fig_matrix, "pca_score_matrix.png", "visualization_page.pca.download_matrix")

        # Loading plot - hiển thị TẤT CẢ components
        st.markdown("---")
        st.write(t("visualization_page.pca.loadings_title").format(n=n_components))
        st.info(t("visualization_page.pca.loadings_info"))

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

        # Get colors for loadings
        loading_colors = get_contrasting_colors(n_components)

        for i in range(n_components):
            ax = axes[i]
            ax.plot(loadings[i], linewidth=1.5, color=loading_colors[i])
            ax.set_title(f'PC{i+1} Loading ({explained_variance[i]*100:.1f}%)', fontweight='bold')
            ax.set_xlabel('Wavenumber index')
            ax.set_ylabel('Loading')
            ax.grid(True, alpha=0.3)

        # Ẩn các subplot trống nếu có
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_with_download(fig3, "pca_loadings.png", "visualization_page.pca.download_loadings")

        # CSV Export cho PCA
        st.markdown("---")
        st.write(t("visualization_page.pca.export_title"))

        col_csv1, col_csv2, col_csv3 = st.columns(3)

        with col_csv1:
            st.write(t("visualization_page.pca.export_scores"))
            # Tạo DataFrame cho scores
            score_columns = [f'PC{i+1}' for i in range(n_components)]
            score_labels = [spectrum_names[i] if i < len(spectrum_names) else f'{t("collection.spectrum_label").format(index=i+1)}' for i in range(len(scores))]
            scores_df = pd.DataFrame(scores, columns=score_columns, index=score_labels)
            scores_df.insert(0, 'Spectrum', score_labels)
            create_csv_download(scores_df, "pca_scores.csv", "visualization_page.pca.download_scores_csv")

        with col_csv2:
            st.write(t("visualization_page.pca.export_loadings"))
            # Tạo DataFrame cho loadings
            loading_columns = [f'Feature_{i+1}' for i in range(loadings.shape[1])]
            loading_labels = [f'PC{i+1}' for i in range(n_components)]
            loadings_df = pd.DataFrame(loadings, columns=loading_columns, index=loading_labels)
            loadings_df.insert(0, 'Component', loading_labels)
            create_csv_download(loadings_df, "pca_loadings.csv", "visualization_page.pca.download_loadings_csv")

        with col_csv3:
            st.write(t("visualization_page.pca.export_variance"))
            # Tạo DataFrame cho explained variance
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                'Explained Variance (%)': explained_variance * 100,
                'Cumulative (%)': np.cumsum(explained_variance) * 100
            })
            create_csv_download(var_df, "pca_explained_variance.csv", "visualization_page.pca.download_variance_csv")

# Footer
st.markdown("---")
st.markdown(t("footer"), unsafe_allow_html=True)
