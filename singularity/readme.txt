to build singularity image from dockerfile:

1. go on hpc cluster
2. log into some compute node, e.g. with "srun --pty -p int -N 1 -t 30 -c 16 /bin/bash"
3. module load apptainer
4. go into directory where dockerfile resides and run: TMPDIR=/usr/users/henrich1/temp apptainer build /usr/users/henrich1/repos/exercises_summer_school/exenv.sif /usr/users/henrich1/repos/exercises_summer_school/singularity/Dockerfile

if problems occur during building, check if tempdir is full and clean files if necessary

# start jupyter notebook with gpu support:
https://jupyter-hpc.gwdg.de/