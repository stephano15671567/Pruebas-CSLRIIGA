import os

# Usamos barras normales / para evitar errores de Windows
ruta = "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"

print("\n--- INICIO DEL DIAGNOSTICO ---")
print(f"Analizando ruta: {ruta}")

if not os.path.exists(ruta):
    print("[ERROR] Python NO encuentra la carpeta train.")
else:
    print("[OK] Carpeta train encontrada.")
    lista = os.listdir(ruta)
    print(f"[INFO] Videos encontrados: {len(lista)}")
    
    if len(lista) > 0:
        video = lista[0]
        ruta_video = os.path.join(ruta, video)
        ruta_uno = os.path.join(ruta_video, "1")
        
        print(f"--- Inspeccionando video: {video} ---")
        if os.path.exists(ruta_uno):
            contenido = os.listdir(ruta_uno)
            print(f"[EXITO] Carpeta '1' encontrada con {len(contenido)} archivos.")
        else:
            print(f"[FALLO] No existe la carpeta '1'. Contenido real: {os.listdir(ruta_video)}")

print("--- FIN ---\n")

