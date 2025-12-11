import cv2
import mediapipe as mp
import numpy as np
import os
import gzip
import sys
import argparse

# --- RUTAS FIJAS (Las que sabemos que funcionan) ---
dataset_path = "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
des_path = "D:/Tesis_CSLR/Repos/CSLR-IIGA/segmentation/train_segmentation"

# Configuracion de MediaPipe
mp_holistic = mp.solutions.holistic

print(f"Ruta Origen: {dataset_path}")
print(f"Ruta Destino: {des_path}")

if not os.path.exists(dataset_path):
    print("ERROR FATAL: No encuentro la carpeta de origen.")
    sys.exit()

video_list = os.listdir(dataset_path)
total_videos = len(video_list)
print(f"--- COMENZANDO PROCESAMIENTO MASIVO DE {total_videos} VIDEOS ---")
print("Esto tomara varias horas. Puedes minimizar la ventana.")

# Iniciamos MediaPipe
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    enable_segmentation=True) as holistic:
    
    count = 0
    for i, video in enumerate(video_list):
        # Barra de progreso simple estilo texto
        count += 1
        if count % 10 == 0 or count == 1:
            print(f"Procesando video {count}/{total_videos}: {video}")

        # Ruta interna a la carpeta '1'
        video_path = os.path.join(dataset_path, video, '1')

        if not os.path.exists(video_path):
            continue

        # Crear carpeta de destino
        save_folder = os.path.join(des_path, video)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        frames = os.listdir(video_path)
        
        for img_name in frames:
            if not img_name.endswith('.png') and not img_name.endswith('.jpg'):
                continue
            
            # Verificar si ya existe (para poder pausar y reanudar otro dia)
            save_file = os.path.join(save_folder, '{}.npy.gz'.format(img_name[:-4]))
            if os.path.exists(save_file):
                continue

            # Leer imagen
            image = cv2.imread(os.path.join(video_path, img_name))
            if image is None: continue

            # Procesar
            image = cv2.resize(image , (224,224))
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.segmentation_mask is None: continue
            
            # Guardar resultado
            array = (results.segmentation_mask > 0.5).astype(int)
            
            f = gzip.GzipFile(save_file, "wb")
            np.save(file=f, arr=array)
            f.close()

print("\n--- ¡PROCESO COMPLETADO! ---")