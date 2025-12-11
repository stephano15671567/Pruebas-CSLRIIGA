import torch
from dataloader import loader
import os
import sys

# --- RUTAS ---
csv_path = r"D:\Tesis_CSLR\Datasets\phoenix-2014.v3\phoenix2014-release\phoenix-2014-multisigner\annotations\manual\train.corpus.csv"
root_dir = r"D:\Tesis_CSLR\Datasets\phoenix-2014.v3\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\train"
seg_path = r"D:\Tesis_CSLR\Repos\CSLR-IIGA\segmentation\train_segmentation"
lookup_table = r"D:\Tesis_CSLR\Repos\CSLR-IIGA\IIGA\tools\data\SLR_lookup_pickle.txt"

print("--- INICIANDO TEST FINAL ---")

if not os.path.exists(csv_path): sys.exit(f"[ERROR] Falta CSV: {csv_path}")
if not os.path.exists(root_dir): sys.exit(f"[ERROR] Falta Imagenes: {root_dir}")
if not os.path.exists(seg_path): sys.exit(f"[ERROR] Falta Segmentacion: {seg_path}")
if not os.path.exists(lookup_table): sys.exit(f"[ERROR] Falta Diccionario: {lookup_table}")

try:
    print("1. Cargando Loader...")
    train_loader, train_size = loader(
        csv_file=csv_path,
        root_dir=root_dir,
        segment_path=seg_path,
        lookup=lookup_table,
        rescale=224,
        batch_size=2,
        num_workers=0, 
        random_drop=True,
        uniform_drop=False,
        show_sample=False,
        istrain=True
    )
    print(f"2. Loader listo ({train_size} videos). Leyendo batch...")

    # --- ESTA ES LA PARTE QUE DEBE CAMBIAR ---
    # NO USAMOS batch['images']. USAMOS ESTO:
    for i, batch in enumerate(train_loader):
        images, src_lengths, text, trg_lengths, hands, hand_lengths = batch
        
        print("\n" + "="*40)
        print("   ✅ ¡EXITO! TODO FUNCIONA   ")
        print("="*40)
        print(f"Shape Imagenes: {images.shape}")
        print("-" * 40)
        break 

except Exception as e:
    print(f"\n❌ ERROR:\n{e}")
    import traceback
    traceback.print_exc()