#!/bin/bash

### -- set the job Name -- 
#BSUB -J yourjob

### -- specify queue -- 
#BSUB -q c02613

### -- ask for number of cores -- 
#BSUB -n 6

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"

### -- set walltime limit: hh:mm or mm--
#BSUB -W 15

### -- set the email address --
#BSUB -u s204165@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

#module load cuda/11.8 
#module load matplotlib/3.10.0-numpy-2.2.2-python-3.10.16
#module load numpy/2.2.2-python-3.10.16-openblas-0.3.29
#module load pandas/2.2.3-python-3.10.16
#module load scipy/1.15.1-python-3.10.16
module load nccl/2.21.5-1-cuda-12.5 

# Setup env
source /dtu/blackhole/00/best/env_vlm_bk/bin/activate

# the file to run 
python milestone_hackathon.py

# run the following in terminal to submit the job
#bsub < yourjob.sh
