python train_search_hfai.py --device=0 --num_obj=2 --num_states=20 --predictor_lambda=1 --lfs_lambda=1 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=1 --num_obj=2 --num_states=20 --predictor_lambda=2 --lfs_lambda=2 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=2 --num_obj=2 --num_states=20 --predictor_lambda=5 --lfs_lambda=5 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=3 --num_obj=2 --num_states=20 --predictor_lambda=10 --lfs_lambda=10 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=4 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=5 --num_obj=2 --num_states=20 --predictor_lambda=50 --lfs_lambda=50 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=6 --num_obj=2 --num_states=20 --predictor_lambda=100 --lfs_lambda=100 --predictor_warm_up=2000 --data=hfai &
python train_search_hfai.py --device=7 --num_obj=2 --num_states=20 --predictor_lambda=200 --lfs_lambda=200 --predictor_warm_up=2000 --data=hfai