/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMABLAS_H
#define MAGMABLAS_H

#include "magma_copy.h"
#include "magmablas_z.h"
#include "magmablas_c.h"
#include "magmablas_d.h"
#include "magmablas_s.h"
#include "magmablas_zc.h"
#include "magmablas_ds.h"
#include "magmablas_h.h"

extern "C"
void
magma_dgemm_cuimma(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    const double* dA, magma_int_t ldda,
    const double* dB, magma_int_t lddb,
    double beta,
    double      * dC, magma_int_t lddc,
    magma_queue_t queue );

#endif // MAGMABLAS_H
