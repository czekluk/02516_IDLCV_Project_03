#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516

### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ----------------
#BSUB -J project_3

### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o /zhome/c4/e/203768/02516_IDLCV_Project_03/hpc_logs/OUTPUT_FILE%J.out
#BSUB -e /zhome/c4/e/203768/02516_IDLCV_Project_03/hpc_logs/OUTPUT_FILE%J.err

module load python3/3.11.9

source /zhome/c4/e/203768/IDLCV/bin/activate
python /zhome/c4/e/203768/02516_IDLCV_Project_03/src/main.py