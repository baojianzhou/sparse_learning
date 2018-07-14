#!/bin/bash
# --------------------- test_loss_func ------------------------------------
bin_dir=../bin/test_loss_func
src_loss="../test_loss_func.c ../loss_func.c ../loss_func.h"
blas_include=/network/rit/lab/ceashpc/bz383376/opt/openblas-0.3.1/include/
blas_lib=/network/rit/lab/ceashpc/bz383376/opt/openblas-0.3.1/lib/
gcc -o ${bin_dir} ${src_loss} -I${blas_include} -L${blas_lib} -lm -lopenblas