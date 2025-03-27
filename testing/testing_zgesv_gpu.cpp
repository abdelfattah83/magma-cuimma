/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COND_THRESHOLD (1)

#define PRECISION_z

////////////////////////////////////////////////////////////////////////////////
static void
magma_zgesv_diagonal_scaling_A(
    magma_int_t N, magmaDoubleComplex* hA, magma_int_t lda, double cond )
{
    /* perform diagonal scaling */
    #ifdef PRECISION_d
    magma_int_t M = N;
    magma_int_t K = N;
    magma_int_t ione     = 1;

    if(cond > COND_THRESHOLD) {
        bool notransA = true;

		// make A between [1,2]
		#pragma omp parallel for
		for(magma_int_t j = 0; j < N; j++) {
		    for(magma_int_t i = 0; i < N; i++) {
			    hA[j*lda + i] += 1.;
		    }
		}


        double cond_sqrt = sqrt(cond);
        double* hD = NULL, *hVA = NULL;
        TESTING_CHECK( magma_dmalloc_cpu( &hD,  K ));
        TESTING_CHECK( magma_dmalloc_cpu( &hVA, M ));
        double scalar = pow( cond, 1/double(K-1) );
        hD[0] = 1 / cond_sqrt;
        for(magma_int_t iD = 1; iD < K; iD++) {
            hD[iD] = hD[iD-1] * scalar;
        }

        if(N == 8) {
            magma_dprint(N, N, hA, lda);
        }

        // scale columns/row of A for N/T
        if( 0 ) {
            for(magma_int_t ik = 0; ik < K; ik++) {
                double* hAt      = ( notransA ) ? hA + lda * ik : hA + ik;
                magma_int_t incA = ( notransA ) ?             1 : lda;
                blasf77_dscal(&K, &hD[ik], hAt, &incA);
            }
        }

        // scale rows/cols of B for N/T
        if( 1 ) {
            for(magma_int_t ik = 0; ik < K; ik++) {
                double* hAt      = ( notransA ) ? hA + ik : hA + lda * ik ;
                magma_int_t incA = ( notransA ) ?     lda : 1;
                double scal      = hD[ik]; //1 / hD[ik];
                blasf77_dscal(&K, &scal, hAt, &incA);
            }
        }


        if( 0 ) {
            // rotate rows/cols right/down of A for N/T
            for(magma_int_t i = 0; i < N; i++) {
                magma_int_t Vm   = ( notransA ) ? 1 : N;
                magma_int_t Vn   = ( notransA ) ? N : 1;
                magma_int_t vlda = Vm;

                double*     hA0 = ( notransA ) ? hA + i : hA + lda * i;
                double*     hA1 = hA0;
                magma_int_t Sm1 = ( notransA ) ? 1    : N-i;
                magma_int_t Sn1 = ( notransA ) ? N-i  : 1;

                double*     hA2 = ( notransA ) ? hA0 + (N-i) * lda : hA0 + (N-i);
                magma_int_t Sm2 = ( notransA ) ? 1 : i;
                magma_int_t Sn2 = ( notransA ) ? i : 1;

                lapackf77_dlacpy( "F", &Sm1, &Sn1, hA1, &lda,  hVA + i, &ione );
                lapackf77_dlacpy( "F", &Sm2, &Sn2, hA2, &lda,  hVA + 0, &ione );
                lapackf77_dlacpy( "F", &Vm,  &Vn,  hVA, &vlda, hA0,     &lda );
            }
        }

        if(M == 8 && N == 8 && K == 8) {
            magma_dprint(N, N, hA, lda);
        }

        magma_free_cpu( hD );
        magma_free_cpu( hVA );
    }
    #endif // end of diagonal scaling of A
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zgesv_scale_A(
    magma_int_t N, magmaDoubleComplex* hA, magma_int_t lda, double factor )
{
    // Upon entry, A entries are assumed to be between 0 & 1

    #ifdef PRECISION_d

    #if 0
    ////////////////////// SCALE ONLY UPPER PART ///////////////////////////
    // Diagonal is set to 1
    #pragma omp parallel for
    for(magma_int_t i = 0 ; i < N; i++) hA[i * lda + i] = 1.;

    // scale upper part by 'factor'
    #pragma omp parallel for schedule(dynamic)
    for(magma_int_t j = 1; j < N; j++) {
        for(magma_int_t i = 0; i < j; i++) {
            hA[j * lda + i] *= factor;
        }
    }

    #else
    ////////////////////// SCALE ENTIRE MATRIX ///////////////////////////
    // scale upper part by 'factor'
    #pragma omp parallel for schedule(dynamic)
    for(magma_int_t j = 0; j < N; j++) {
        for(magma_int_t i = 0; i < N; i++) {
            hA[j * lda + i] *= factor;
        }
    }

    #endif


    if(N == 8) magma_dprint(N, N, hA, lda);
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zgesv_generate_A(
    magma_int_t N, magmaDoubleComplex* hA, magma_int_t lda,
    magma_int_t nb, double cond, magma_int_t *ISEED)
{
    // generate a special matrix that results in a badly scaled
    // schur complement during the first iteration in a blocked factorization
    // requires knowledge of the used blocking size (nb)

    if(nb <= 0) return;

    if(cond < 1) cond = 1.;

    // constants
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t ione     = 1;
    magma_int_t sizeA    = lda * N;
    magma_int_t width    = N-nb;

    // generate a random A
    lapackf77_zlarnv( &ione, ISEED, &sizeA, hA );

    if(N == 16 && nb == 4) {
        printf("org\n");
        magma_zprint(N, N, hA, lda);
    }

    // TODO: maybe make A between 1:2 ?

    // prepare a diagonal matrix with large dynamic range based on cond
    double cond_sqrt = sqrt(cond);
    magmaDoubleComplex* hD = NULL;
    TESTING_CHECK( magma_zmalloc_cpu( &hD,  nb ));
    double scalar = pow( cond, 1/double(nb-1) );
    hD[0] = MAGMA_Z_MAKE(cond_sqrt, MAGMA_D_ZERO);
    for(magma_int_t iD = 1; iD < nb; iD++) {
        hD[iD] = hD[iD-1] / scalar;
    }

    if(N == 16 && nb == 4) {
        printf("hD\n");
        magma_zprint(1, nb, hD, 1);
    }

    // copy the panel into a workspace
    magmaDoubleComplex *hPanel = NULL;
    magma_zmalloc_cpu(&hPanel, N * nb);
    lapackf77_zlacpy("F", &N, &nb, hA, &lda, hPanel, &N);

    if(N == 16 && nb == 4) {
        printf("hPanel\n");
        magma_zprint(N, nb, hPanel, N);
    }

    // perform LU factorization on the panel
    magma_int_t *ipiv = NULL;
    magma_int_t info;
    magma_imalloc_cpu(&ipiv, nb);
    lapackf77_zgetrf(&N, &nb, hPanel, &N, ipiv, &info);

    if(N == 16 && nb == 4) {
        printf("LU on hPanel\n");
        magma_zprint(N, nb, hPanel, N);
    }

    // generate a permutation vector based on ipiv
    magma_int_t *ipermute = NULL;
    magma_imalloc_cpu(&ipermute, N);
    #pragma omp parallel for
    for(magma_int_t i = 0; i < N; i++) ipermute[i] = i;

    for(magma_int_t i = 0; i < nb; i++) {
        magma_int_t pivot = ipiv[i] - 1; // undo fortran indexing

        // swap
        magma_int_t tmp = ipermute[i];
        ipermute[i]     = ipermute[ pivot ];
        ipermute[pivot] = tmp;
    }

    if(N == 16 && nb == 4) {
        printf("iPermute\n");
        magma_iprint(1, N, ipermute, 1);
    }


    // generate the "inverse permutation matrix"
    // the permutation matrix has exactly one entry per row set to '1', decided by permutation vector
    // its matrix is equal to its transpose (one entry per column set to '1')
    magmaDoubleComplex *hPermute = NULL;
    magma_zmalloc_cpu(&hPermute, N*N);
    memset(hPermute, 0, N*N*sizeof(magmaDoubleComplex));
    for(magma_int_t j = 0; j < N; j++) {
        hPermute[j * N + ipermute[j]] = MAGMA_Z_ONE;
    }

    if(N == 16 && nb == 4) {
        printf("hPermute\n");
        magma_zprint(N, N, hPermute, N);
    }

    // scale A[1:nb, nb+1:N] using hD
    magma_int_t scal_length = width;
    for(magma_int_t j = 0; j < nb; j++) {
        magmaDoubleComplex* hAt = hA + nb * lda + j;
        blasf77_zscal(&scal_length, &hD[j], hAt, &lda);
    }

    if(N == 16 && nb == 4) {
        printf("After diagonal scaling\n");
        magma_zprint(N, N, hA, lda);
    }


    // undo trsm, multiply the square part of the 'L' factor with A[1:nb, nb+1:N]
    blasf77_ztrmm( lapack_side_const(MagmaLeft), lapack_uplo_const(MagmaLower),
                   lapack_trans_const(MagmaNoTrans), lapack_diag_const(MagmaUnit),
                   &nb, &width, &c_one,
                   hPanel, &N, hA + nb * lda, &lda);

    if(N == 16 && nb == 4) {
        printf("After trmm\n");
        magma_zprint(N, N, hA, lda);
    }

    // undo pivoting, pre-multiply A[1:N, nb+1 : N] with the inverse permutation matrix
    // copy A[1:N, nb+1 : N] into hTmp then: hPermute x hTmp ==> A[1:N, nb+1 : N]
    magmaDoubleComplex *hTmp = NULL;
    magma_zmalloc_cpu(&hTmp, N * (N-nb));
    lapackf77_zlacpy ("F", &N, &width, hA + nb*lda, &lda, hTmp, &N);
    blasf77_zgemm( lapack_trans_const(MagmaNoTrans), lapack_trans_const(MagmaNoTrans),
                     &N, &width, &N,
                     &c_one,  hPermute,    &N,
                              hTmp,        &N,
                     &c_zero, hA + nb*lda, &lda);

    if(N == 16 && nb == 4) {
        printf("After gemm with permutation matrix\n");
        magma_zprint(N, N, hA, lda);
    }

    #if 1
    // For testing only
    lapackf77_zlaswp(&width, hA + nb * lda, &lda, &ione, &nb, ipiv, &ione);
    blasf77_ztrsm( lapack_side_const(MagmaLeft), lapack_uplo_const(MagmaLower),
                   lapack_trans_const(MagmaNoTrans), lapack_diag_const(MagmaUnit),
                   &nb, &width, &c_one,
                   hPanel, &N, hA + nb * lda, &lda);


    if(N == 16 && nb == 4) {
        printf("before 1st schur complement\n");
        magma_zprint(N, N, hA, lda);
    }
    #endif

    magma_free_cpu( hD );
    magma_free_cpu( hPanel );
    magma_free_cpu( ipiv );
    magma_free_cpu( ipermute );
    magma_free_cpu( hPermute );
    magma_free_cpu( hTmp );

}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesv_gpu
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X, *h_Xlapack;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t *ipiv;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    nrhs = opts.nrhs;

    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   Anorm      Xnorm      ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb   = ldda;
            gflops = ( FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, nrhs ) ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Xlapack, ldb*nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));

            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs ));

            /* Initialize the matrices */
            magma_int_t sizeA = lda*N;
            sizeB = ldb*nrhs;

            // check nb
            if(opts.nb <= 4) opts.nb = 4;

            // generate A
            #if 1
            magma_zgesv_generate_A(N, h_A, lda, opts.nb, opts.cond, ISEED);
            //magma_generate_matrix( opts, N, N, h_A, lda );
            #else
            // generate randomly & scale full or upper part to large values
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            //magma_zgesv_scale_A(N, h_A, lda, opts.cond );
            magma_zgesv_diagonal_scaling_A(N, h_A, lda, opts.cond );
            #endif

            // generate B
            #if 0
            /////////////////////////////////////////////////////////////////////////////
            // set the solution to be all one’s and generate b = A*x
            magmaDoubleComplex c_zero     = MAGMA_Z_ZERO;
            #pragma omp parallel for
            for(magma_int_t j = 0; j < nrhs; j++) {
                for(magma_int_t i = 0; i < N; i++) {
                    h_Xlapack[j * ldb + i] = MAGMA_Z_ONE;
                }
            }
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                               &c_one,  h_A,       &lda,
                                        h_Xlapack, &ldb,
                               &c_zero, h_B,       &ldb);

            // copy h_B to h_Xlapack
            memcpy(h_Xlapack, h_B, sizeB * sizeof(magmaDoubleComplex));

            #else
            /////////////////////////////////////////////////////////////////////////////
            // let B be close to 1
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B ); // [0:1]
            #ifdef PRECISION_d
            for(magma_int_t j = 0; j < nrhs; j++) {
                for(magma_int_t i = 0; i < N; i++) {
                    h_B[j * ldb + i] *= 0.1; // [0.0 : 0.1]
                    h_B[j * ldb + i] += 0.9; // [0.9 : 1.0]
                    h_B[j * ldb + i] /= opts.cond; // [0.9 : 1.0]
                }
            }
            #endif // PRECISION_d
            // copy h_B to h_Xlapack
            memcpy(h_Xlapack, h_B, sizeB * sizeof(magmaDoubleComplex));

            #endif

            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_zgesv_gpu( N, nrhs, d_A, ldda, ipiv, d_B, lddb, &info );
            }
            else if (opts.version == 2){
                magma_zgesv_native_oz(N, nrhs, d_A, ldda, ipiv, d_B, lddb, &info, opts.oz_nsplits);
            }
            else if(opts.version == 3) {
                magma_zgesv_native_oz_nb(N, nrhs, opts.nb, d_A, ldda, ipiv, d_B, lddb, &info, opts.oz_nsplits);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgesv_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual & run LAPACK
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );

            //if(N == 4096 && nrhs == 1) {
            //   magma_zprint(100, nrhs, h_X, ldb);
            //}

            if(opts.check == 1) {
                Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X, &ldb, work);

                blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                               &c_one,     h_A, &lda,
                                           h_X, &ldb,
                               &c_neg_one, h_B, &ldb);

                Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, work);
                error = Rnorm/(N*Anorm*Xnorm);
            }

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgesv( &N, &nrhs, h_A, &lda, ipiv, h_Xlapack, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgesv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }

            status += ! (error < tol);

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        Anorm, Xnorm, error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, gpu_perf, gpu_time,
                        Anorm, Xnorm, error, (error < tol ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Xlapack );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );

            magma_free( d_A );
            magma_free( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
