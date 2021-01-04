for method in 1 2 3
do
  for fold in 0 1 2
  do
    for l in $2 # 6 8 10 15
    do
    CUDA_VISIBLE_DEVICES=$1 python select_region.py --exp select_region_seed --data_fold $fold --group_method $method --L $l
    done
  done
done

