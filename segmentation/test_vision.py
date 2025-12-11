# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import os
import numpy as np

# --- RUTAS ---
# Ruta a la carpeta de UN video especifico que sabemos que existe
ruta_video = "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1"

# Nombre de UNA imagen que vimos en tu 'dir'
nombre_imagen = "01April_2010_Thursday_heute.avi_pid0_fn000000-0.png"

ruta_completa = os.path.join(ruta_video, nombre_imagen)

print("-" * 40)
print("PRUEBA DE VISION ARTIFICIAL")
print("Intentando leer imagen:")
print(ruta_completa)

# 1. Verificar si el archivo existe
if os.path.exists(ruta_completa):
    print("[OK] El archivo existe en el disco.")
else:
    print("[ERROR FATAL] Python no encuentra el archivo. Revisa la ruta.")
    exit()

# 2. Intentar leer con OpenCV
try:
    img = cv2.imread(ruta_completa)
except Exception as e:
    print(f"[ERROR EXCEPCION] Fallo al intentar leer: {e}")
    exit()

if img is None:
    print("[FALLO] OpenCV devolvio 'None'. No pudo leer la imagen.")
    print("Posibles causas: Ruta muy larga o caracteres extraños.")
    exit()
else:
    print(f"[OK] Imagen leida correctamente. Dimensiones: {img.shape}")

# 3. Intentar procesar con MediaPipe
print("Cargando MediaPipe...")
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
    # Convertir color
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Procesar
    results = holistic.process(img_rgb)
    
    if results.segmentation_mask is not None:
        print("[EXITO TOTAL] MediaPipe genero la mascara.")
        print("El sistema de vision funciona bien.")
    else:
        print("[ADVERTENCIA] MediaPipe corrio pero no encontro nada.")

print("-" * 40)
