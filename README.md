Please download SurvivalEVAL from https://github.com/shi-ang/CSD/blob/main/README.md

You may download datasets by following the explanation of the appendix D.

Below is the example to run the code.

# DeepSurv
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cox --lam 0 --lr 1e-4 --dropout_rate 0.1 --batch_size 64 --num_epochs 1000
# MTLR
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model MTLRNN --model_dist mtlr --lam 0 --lr 1e-2 --num_cat_bins 20 --batch_size 64 --num_epochs 1000
# Parametric model
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist lognormal --lam 0 --lr 1e-3 --dropout_rate 0.1 --batch_size 64 --num_epochs 1000
# CRPS
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist lognormal --loss_fn crps --lam 0 --lr 1e-2 --dropout_rate 0.1 --batch_size 128 --num_epochs 1000
# DeepHit
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cat --lam 0 --lr 1e-3 --dropout_rate 0.1 --batch_size 64 --num_epochs 1000
# AFT
python train_ks.py --name ks --dataset whas --data_dir data/whas/ --model AFTNN --model_dist weibull --lam 0 --lr 1e-2 --dropout_rate 0.1 --batch_size 64 --num_epochs 1000


**After training, do the post-processing by using checkpoints.**
# Baseline
python test.py --name ks --phase test --dropout_rate 0.0 --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cox --batch_size 982 --ckpt_path ckpts/whas/DeepSurv/cox_ks_ds_whas_lam0.0_dr0.1_bs64_lr0.0001_optimadam_epoch1000_seed1/best.pth.tar
# KSP
python KSP.py --name ks --phase test --dropout_rate 0.0 --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cox --batch_size 982 --ckpt_path ckpts/whas/DeepSurv/cox_ks_ds_whas_lam0.0_dr0.1_bs64_lr0.0001_optimadam_epoch1000_seed1/best.pth.tar
# CSD
python CSD.py --name ks --phase test --dropout_rate 0.0 --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cox --batch_size 982 --ckpt_path ckpts/whas/DeepSurv/cox_ks_ds_whas_lam0.0_dr0.1_bs64_lr0.0001_optimadam_epoch1000_seed1/best.pth.tar
# CSD-iPOT
python CSD-iPOT.py --name ks --phase test --dropout_rate 0.0 --dataset whas --data_dir data/whas/ --model SyntheticNN --model_dist cox --batch_size 982 --ckpt_path ckpts/whas/DeepSurv/cox_ks_ds_whas_lam0.0_dr0.1_bs64_lr0.0001_optimadam_epoch1000_seed1/best.pth.tar
