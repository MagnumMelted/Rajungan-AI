import streamlit as st
import cv2
import tempfile
import openpyxl
import os
from detection import init_models, process_frame, export_to_excel

# Set judul halaman
st.set_page_config(page_title="Deteksi Rajungan", layout="wide")
st.title("📷 Deteksi Panjang, Lebar, & Berat Rajungan")

# Inisialisasi model saat pertama kali
if "models_initialized" not in st.session_state:
    with st.spinner("Memuat model YOLO & EasyOCR..."):
        init_models()
    st.session_state["models_initialized"] = True

# Tempat untuk video
video_placeholder = st.empty()

# Placeholder untuk info berat, lebar, panjang agar tidak menumpuk
info_placeholder = st.empty()

# Variabel simpan frame terakhir untuk snapshot
if "last_frame" not in st.session_state:
    st.session_state["last_frame"] = None

# Fungsi convert frame OpenCV BGR ke bytes PNG untuk download
def convert_frame_to_bytes(frame):
    ret, buf = cv2.imencode('.png', frame)
    if not ret:
        return None
    return buf.tobytes()

# Tombol kontrol
col1, col2, col3 = st.columns([1,1,1])
start_btn = col1.button("▶ Mulai Kamera")
stop_btn = col2.button("⏹ Hentikan Kamera")
snapshot_btn = col3.button("📸 Ambil Snapshot")

# Jalankan webcam
if start_btn:
    cap = cv2.VideoCapture(0)
    st.session_state["camera_running"] = True
    while st.session_state.get("camera_running", False):
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera!")
            break

        processed_frame, berat, ukuran = process_frame(frame)

        # Simpan frame terakhir hasil proses ke session_state (BGR)
        st.session_state["last_frame"] = processed_frame.copy()

        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(processed_frame_rgb, channels="RGB")

        if berat is not None and ukuran is not None:
            info_placeholder.markdown(
                f"**Berat:** {berat:.1f} gram  \n"
                f"**Lebar:** {ukuran['lebar_cm']:.2f} cm  \n"
                f"**Panjang:** {ukuran['panjang_cm']:.2f} cm"
            )
        else:
            info_placeholder.text("Menunggu data deteksi...")

        if stop_btn:
            st.session_state["camera_running"] = False

    cap.release()

# Tombol snapshot: simpan dan siap diunduh
if snapshot_btn:
    frame = st.session_state.get("last_frame", None)
    if frame is not None:
        img_bytes = convert_frame_to_bytes(frame)
        if img_bytes:
            st.success("Snapshot berhasil diambil!")
            st.download_button(
                label="Unduh Snapshot PNG",
                data=img_bytes,
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
