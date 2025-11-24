import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Сегментация спутниковых снимков", page_icon="🛰️", layout="wide")

BACKEND_URL = "http://localhost:8000"


def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'


def call_segmentation_api(file_bytes: bytes) -> Image.Image:
    files = {"file": ("image.png", file_bytes, "image/png")}
    resp = requests.post(f"{BACKEND_URL}/predict/", files=files)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


with st.sidebar:
    with st.expander("📋 Легенда классов", expanded=True):
        classes = {
            "Дороги": "#FFFFFF",
            "Здания": "#0000FF",
            "Низкая растительность": "#00FFFF",
            "Деревья": "#00FF00",
            "Машины": "#FFFF00",
            "Прочее": "#FF0000",
        }
        for name, color in classes.items():
            st.markdown(
                f'<span style="display:inline-block;width:1em;height:1em;background:{color};'
                f'margin-right:0.5em;border:1px solid #000;"></span>{name}',
                unsafe_allow_html=True
            )

st.title("🛰️ Сегментация спутниковых снимков")

uploaded = st.file_uploader("Выберите снимок", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(img, use_container_width=True)
    
    if st.button("▶️ Сегментировать"):
        with st.spinner("Идёт сегментация..."):
            try:
                result = call_segmentation_api(uploaded.getvalue())
                with col2:
                    st.subheader("Результат сегментации")
                    st.image(result, use_container_width=True)
                    link = get_image_download_link(result, "segmentation.png", "⬇️ Скачать")
                    st.markdown(link, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Ошибка: {e}")
else:
    st.info("Загрузите изображение")
