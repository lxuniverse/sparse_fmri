for lr in $3 # 1e-5 1e-4 1e-6
do
  for gm in $2 # 1 2 3
  do
    for fold in 0 1 2
    do
    CUDA_VISIBLE_DEVICES=$1 python fine_tune.py --exp fine_tune_ft --data_fold $fold --group_method $method --lr $lr
    done
  done
done

