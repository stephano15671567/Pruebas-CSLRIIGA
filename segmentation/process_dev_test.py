import cv2
import mediapipe as mp
import numpy as np
import os
import gzip
import sys

# --- TU CONFIGURACION DE RUTAS (WINDOWS) ---
# Ruta base donde estan las carpetas train, dev, test
base_dataset = "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px"
# Ruta base donde guardaremos las carpetas de salida
base_dest = "D:/Tesis_CSLR/Repos/CSLR-IIGA/segmentation"

# Lista de lo que falta procesar
subsets = ['dev', 'test'] 

mp_holistic = mp.solutions.holistic

print("=" * 50)
print("   SCRIPT DE PROCESAMIENTO: DEV y TEST")
print("=" * 50)

# Verificacion inicial
if not os.path.exists(base_dataset):
    print("ERROR FATAL: No encuentro la carpeta base del dataset.")
    sys.exit()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    enable_segmentation=True) as holistic:

    for subset in subsets:
        # Construir rutas automaticamente
        dataset_path = os.path.join(base_dataset, subset)
        des_path = os.path.join(base_dest, f"{subset}_segmentation")
        
        print(f"\n>>> INICIANDO SET: {subset.upper()}")
        print(f"    Origen:  {dataset_path}")
        print(f"    Destino: {des_path}")

        if not os.path.exists(dataset_path):
            print(f"    [ERROR] No encuentro la carpeta '{subset}'. Verificalo.")
            continue

        video_list = os.listdir(dataset_path)
        total_videos = len(video_list)
        print(f"    Total de videos a procesar: {total_videos}")
        
        count = 0
        for i, video in enumerate(video_list):
            count += 1
            # Log simple cada 20 videos
            if count % 20 == 0 or count == 1:
                print(f"    [{subset.upper()}] Progreso: {count}/{total_videos} videos...")

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
                
                save_file = os.path.join(save_folder, '{}.npy.gz'.format(img_name[:-4]))
                
                # Si ya existe, saltamos (eficiencia)
                if os.path.exists(save_file):
                    continue

                image = cv2.imread(os.path.join(video_path, img_name))
                if image is None: continue

                image = cv2.resize(image , (224,224))
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results.segmentation_mask is None: continue
                
                array = (results.segmentation_mask > 0.5).astype(int)
                
                f = gzip.GzipFile(save_file, "wb")
                np.save(file=f, arr=array)
                f.close()
        
        print(f"    >>> Set {subset.upper()} finalizado.")

print("\n" + "="*50)
print("¡TODO EL DATASET (DEV + TEST) HA SIDO PROCESADO!")
print("="*50)
