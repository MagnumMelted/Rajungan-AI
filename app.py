import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detection import init_models, process_frame, export_to_excel

st.set_page_config(page_title="Deteksi Rajungan", layout="wide")
st.title("📷 Deteksi Panjang, Lebar, & Berat Rajungan")

# Inisialisasi model sekali saja
if "models_initialized" not in st.session_state:
    with st.spinner("Memuat model YOLO & EasyOCR..."):
        init_models()
    st.session_state["models_initialized"] = True

# Placeholder untuk info berat, lebar, panjang
info_placeholder = st.empty()

# Kelas VideoProcessor untuk memproses frame video
class RajunganVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.berat = None
        self.ukuran = None
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed_frame, berat, ukuran = process_frame(img)

        self.berat = berat
        self.ukuran = ukuran
        self.last_frame = processed_frame.copy() if processed_frame is not None else None

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# Jalankan webrtc streamer
ctx = webrtc_streamer(key="rajungan", video_processor_factory=RajunganVideoProcessor)

# Setelah stream aktif, tampilkan info berat, lebar, panjang
if ctx.video_processor:
    berat = ctx.video_processor.berat
    ukuran = ctx.video_processor.ukuran

    if berat is not None and ukuran is not None:
        info_placeholder.markdown(
            f"**Berat:** {berat:.1f} gram  \n"
            f"**Lebar:** {ukuran['lebar_cm']:.2f} cm  \n"
            f"**Panjang:** {ukuran['panjang_cm']:.2f} cm"
        )
    else:
        info_placeholder.text("Menunggu data deteksi...")

# Tombol snapshot: ambil frame terakhir dan download
if st.button("📸 Ambil Snapshot"):
    if ctx.video_processor and ctx.video_processor.last_frame is not None:
        ret, buf = cv2.imencode('.png', ctx.video_processor.last_frame)
        if ret:
            st.success("Snapshot berhasil diambil!")
            st.download_button(
                label="Unduh Snapshot PNG",
                data=buf.tobytes(),
                file_name="snapshot_terakhir.png",
                mime="image/png"
            )
        else:
            st.error("Gagal mengubah frame ke gambar.")
    else:
        st.warning("Tidak ada frame untuk di-snapshot.")

# Tombol download hasil deteksi Excel
if st.button("💾 Download Hasil ke Excel"):
    filepath = export_to_excel()
    if filepath:
        st.success(f"File siap diunduh: {filepath}")
        with open(filepath, "rb") as f:
            st.download_button(
                label="Unduh File Excel",
                data=f,
                file_name="hasil_deteksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Belum ada data untuk diunduh.")
