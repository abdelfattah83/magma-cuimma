machine=blackwell
magma_cu=.
outdir=cuimma_results
logfile=$outdir/cuimma_log.txt

mkdir -p $outdir
touch    $logfile

alldone=1
##################################################################################################################
## GEMM on random matrices, three shapes (square, rank-k updates, left-looking cholesky updates)
nstart=1024
nstop=20480
nstep=1024
niter=4

# square
for s in 3 6 7 8
do
    fname=$outdir/dgemm_sq_cuimma_s${s}_$machine.txt
    status=done_dgemm_sq_cuimma_s${s}_$machine.txt
    if grep -Fxq "$status" $logfile; then
        : # nothing to do
    else
        # run the test
        alldone=0
        echo "generating " $fname
        ${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep --oz $s --niter $niter -l > $fname
        echo $status >> $logfile
    fi
done

# rank-k updates (k = 512)
for s in 3 6 7 8
do
    for k in 512
    do
        fname=$outdir/dgemm_k${k}_cuimma_s${s}_$machine.txt
        status=done_dgemm_k${k}_cuimma_s${s}_$machine.txt
        if grep -Fxq "$status" $logfile; then
            : # nothing to do
        else
            # run the test
            alldone=0
            echo "generating " $fname
            ${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep,$nstart:$nstop:$nstep,${k} --oz $s --niter $niter -l > $fname
            echo $status >> $logfile
        fi
    done
done

# left-looking chol. updates (N = 40960, k = 512)
N=40960
K=512
Nstop=$((N-K-K))
for s in 3 6 7 8
do
    fname=$outdir/dgemm_chol_40k_k${K}_cuimma_s${s}_${machine}.txt
    status=done_dgemm_chol_40k_k${K}_cuimma_s${s}_${machine}.txt
    if grep -Fxq "$status" $logfile; then
        : # nothing to do
    else
        # run the test
        alldone=0
        echo "generating " $fname
        ${magma_cu}/testing_dgemm -N $Nstop:$K:-$K,$K,$K:$Nstop:$K -NT --oz $s --niter $niter -l > $fname
        echo $status >> $logfile
    fi
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
        status=done_dgemm_sq_badly_scaled_Dcond_${cond}_cuimma_s${s}_$machine.txt
        if grep -Fxq "$status" $logfile; then
            : # nothing to do
        else
            # run the test
            alldone=0
            echo "generating " $fname
            ${magma_cu}/testing_dgemm -N $nstart:$nstop:$nstep --oz $s --niter $niter -l -c2 --cond $cond > $fname
            echo $status >> $logfile
        fi
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
            status=done_dsyevd_${mtx}_${cond}_cuimma_s${splits}_${machine}.txt
            if grep -Fxq "$status" $logfile; then
                : # nothing to do
            else
                # run the test
                alldone=0
                echo "generating " $fname
                ${magma_cu}/testing_dsyevd_gpu -JV -N $nstart:$nstop:$nstep -c --niter $niter --matrix $mtx --cond $cond --oz $splits > $fname
                echo $status >> $logfile
            fi
        done
    done
done

if [ "$alldone" = "1" ]; then
  echo "All tests already complete"
fi
