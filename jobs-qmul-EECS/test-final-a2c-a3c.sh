for i in {0,2}
    do
    for j in {1,5,6}
        do
            /homes/sg324/.conda/envs/central/bin/python \
            /homes/sg324/mobile-kube/experiments/analysis/test_rls.py \
            --train-series $1 --test-series $1 --network-id $j --experiment-id 0 \
            --trace-id 2 --trace-id-test $i
        done
    done