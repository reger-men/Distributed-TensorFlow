CUDA_VISIBLE_DEVICES=-1 python3 task.py --job_name=ps --task_index=0 &
CUDA_VISIBLE_DEVICES=1 python3 task.py --job_name=worker --task_index=1 &
CUDA_VISIBLE_DEVICES=2 python3 task.py --job_name=worker --task_index=2

