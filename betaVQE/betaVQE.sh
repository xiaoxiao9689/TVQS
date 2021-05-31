#!/bin/bash
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

beta=0.5
h=0.0
delta=1.0
plot=1
for ti in `seq 0 1 4`
do
    python betaVQE.py $beta $h $delta $ti $plot&
done
wait 