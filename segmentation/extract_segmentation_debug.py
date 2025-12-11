import cv2
import mediapipe as mp
import numpy as np
import os
import gzip
import sys

# --- RUTAS FIJAS ---
dataset_path = "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
des_path = "D:/Tesis_CSLR/Repos/CSLR-IIGA/segmentation/train_segmentation"

print(f"1. Buscando videos en: {dataset_path}")

if not os.path.exists(dataset_path):
    print("ERROR FATAL: La carpeta de origen no existe.")
    sys.exit()

video_list = os.listdir(dataset_path)
print(f"2. Se encontraron {len(video_list)} carpetas de video.")

if len(video_list) == 0:
    print("ERROR: La carpeta esta vacia.")
    sys.exit()

print("3. Cargando MediaPipe...")
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    enable_segmentation=True) as holistic:

    print("4. Iniciando bucle de procesamiento (MODO TEXTO)...")
    
    # Procesamos solo el PRIMER video para probar
    video = video_list[0] 
    print(f"\n--- PROCESANDO VIDEO 1: {video} ---")

    video_path = os.path.join(dataset_path, video, '1')
    print(f"   Buscando ruta interna: {video_path}")

    if not os.path.exists(video_path):
        print("   [ERROR] No encuentro la carpeta '1' dentro de este video.")
    else:
        frames = os.listdir(video_path)
        print(f"   [OK] Carpeta '1' encontrada. Contiene {len(frames)} archivos.")
        
        # Crear carpeta destino
        save_folder = os.path.join(des_path, video)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"   [INFO] Carpeta destino creada: {save_folder}")
        
        count = 0
        for img_name in frames:
            if not img_name.endswith('.png'):
                continue
                
            # Intento de procesar UNA imagen
            print(f"   -> Procesando imagen: {img_name} ...", end="")
            
            # Leer
            img_full = os.path.join(video_path, img_name)
            image = cv2.imread(img_full)
            
            if image is None:
                print(" [FALLO AL LEER]")
                continue

            # Procesar
            image = cv2.resize(image , (224,224))
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.segmentation_mask is None:
                print(" [SIN MASCARA - NORMAL]")
                continue
            
            # Guardar
            array = (results.segmentation_mask > 0.5).astype(int)
            save_file = os.path.join(save_folder, '{}.npy.gz'.format(img_name[:-4]))
            
            f = gzip.GzipFile(save_file, "wb")
            np.save(file=f, arr=array)
            f.close()
            
            print(" [GUARDADO OK]")
            count += 1
            if count >= 3: # Solo probamos 3 frames para no llenar la pantalla
                print("\n   [STOP] Prueba finalizada exitosamente tras 3 frames.")
                break

print("\n--- FIN DEL DIAGNOSTICO ---")

