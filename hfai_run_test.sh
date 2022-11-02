python train_search_hfai.py --device=0 --search_epochs=1 --retrain_epochs=1 --num_obj=2 --num_states=14 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=100 --warm_up_population=1


# hfai stop hfai_run_test.sh
# hfai bash hfai_run_test.sh -- --force --no_diff -n 1
