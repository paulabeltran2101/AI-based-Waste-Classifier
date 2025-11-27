import streamlit as st
import cv2
import time

st.title("🔴 Test cámara Logitech StreamCam")
st.write("Prueba de cámara en directo usando OpenCV + Streamlit (sin WebRTC)")

# Placeholder donde se mostrará el vídeo
frame_window = st.image([])

# Abrir la cámara externa (Logitech StreamCam)
# Prueba 0, si no funciona prueba 1, 2...
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    st.error("❌ No se ha podido abrir la cámara. Prueba con ID 1 o 2.")
else:
    st.success("🎥 Cámara conectada correctamente.")

# Checkbox para iniciar/detener la cámara
run = st.checkbox("Iniciar cámara")

frame_count = 0

while run:
    ret, frame = cam.read()
    if not ret:
        st.warning("⚠️ No se puede leer frame de la cámara.")
        print("No se pudo leer frame de la cámara")
        break

    frame_count += 1
    print(f"Frame {frame_count} leído")

    # Convertir BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mostrar frame en Streamlit
    frame_window.image(frame_rgb)

    time.sleep(0.03)  # Aproximadamente 30 FPS

cam.release()
print("Cámara liberada")
