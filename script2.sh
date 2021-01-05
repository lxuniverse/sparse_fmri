for lr in 1e-5 5e-5 5e-6 # 1e-5 1e-4 1e-6
do
  for method in $2 # 1 3
  do
    for fold in $3 # 0 1 2
    do
      for gnum in 5
      do
      CUDA_VISIBLE_DEVICES=$1 python fine_tuning.py --group_select_num $gnum --exp fine_tune_ft --data_fold $fold --group_method $method --lr $lr
      done
    done
  done
done

