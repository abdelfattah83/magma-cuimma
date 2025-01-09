machine=blackwell
magma_cu=.
outdir=cuimma_results

mkdir -p $outdir

##################################################################################################################
## GEMM on random matrices, three shapes (square, rank-k updates, left-looking cholesky updates)
nstart=1024
nstop=20480
nstep=1024
niter=5

# square
for s in 3 6 7 8
do
   fname=$outdir/dgemm_sq_cuimma_s${s}_$machine.txt
   echo "generating " $fname
   ${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep --oz $s --niter $niter -l > $fname
done

# rank-k updates (k = 512)
for s in 3 6 7 8
do
   for k in 512
   do
       fname=$outdir/dgemm_k${k}_cuimma_s${s}_$machine.txt
       echo "generating " $fname
       ${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep,$nstart:$nstop:$nstep,${k} --oz $s --niter $niter -l > $fname
   done
done

# left-looking chol. updates (N = 40960, k = 512)
N=40960
K=512
Nstop=$((N-K-K))
for s in 3 6 7 8
do
    fname=$outdir/dgemm_chol_40k_k${K}_cuimma_s${s}_${machine}.txt
    echo "generating " $fname
    ${magma_cu}/testing_dgemm -N $Nstop:$K:-$K,$K,$K:$Nstop:$K -NT --oz $s --niter $niter -l > $fname
done

##################################################################################################################
# GEMM on badly scaled matrices
nstart=1024
nstop=20480
nstep=1024
niter=1

for s in 8 12 16 18
do
   for cond in 1e10 1e20 1e30 1e40
   do
	fname=$outdir/dgemm_sq_badly_scaled_Dcond_${cond}_cuimma_s${s}_$machine.txt
	echo "generating " $fname
	${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep --oz $s --niter $niter -l -c2 --cond $cond > $fname
    done
done


##################################################################################################################
# DSYEVD
nstart=1024
nstop=10240
nstep=1024
niter=1

for splits in 3 6 7 8
do
    for cond in 1e5 1e10
    do
        for mtx in poev_cluster1
        do
            fname=$outdir/dsyevd_${mtx}_${cond}_cuimma_s${splits}_${machine}.txt
            echo "generating " $fname
            ${magma_cu}/testing_dsyevd_gpu -JV -N $nstart:$nstop:$nstep -c --niter $niter --matrix $mtx --cond $cond --oz $splits > $fname
        done
    done
done

