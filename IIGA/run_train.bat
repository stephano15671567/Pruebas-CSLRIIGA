@echo off
echo ===================================================
echo INICIANDO ENTRENAMIENTO DE TESIS (CSLR - PHOENIX)
echo ===================================================
echo.

python train.py ^
 --data "D:/Tesis_CSLR/Datasets/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner" ^
 --train_segment_root "D:/Tesis_CSLR/Repos/CSLR-IIGA/segmentation/train_segmentation" ^
 --val_segment_root "D:/Tesis_CSLR/Repos/CSLR-IIGA/segmentation/dev_segmentation" ^
 --lookup_table "D:/Tesis_CSLR/Repos/CSLR-IIGA/IIGA/tools/data/SLR_lookup_pickle.txt" ^
 --data_stats "D:/Tesis_CSLR/Repos/CSLR-IIGA/IIGA/tools/data/data_stats.pt" ^
 --hand_stats "D:/Tesis_CSLR/Repos/CSLR-IIGA/IIGA/tools/data/data_stats.pt" ^
 --save_dir "D:/Tesis_CSLR/Repos/CSLR-IIGA/trained_model" ^
 --num_workers 0 ^
 --batch_size 2 ^
 --data_type "features" ^
 --weight_decay 1e-5 ^
 --dp_keep_prob 0.9 ^
 --pretrained True ^
 --scheduler "multi-step" ^
 --milestones "10,30" ^
 --num_epochs 40

echo.
echo ===================================================
echo ENTRENAMIENTO FINALIZADO O INTERRUMPIDO
echo ===================================================
pause