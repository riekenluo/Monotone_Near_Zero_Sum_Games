#!/bin/bash

# Define the values for rho and seed
rho=0.0003
seed=0
tol=1e-7
alg=ICL

python main.py -rho "$rho" -tol "$tol" -seed "$seed" -alg "$alg"

