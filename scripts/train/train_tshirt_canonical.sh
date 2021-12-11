tcp_port=$((12000 + $RANDOM % 20000))
ngpu=2

srun -p dsta \
     --mpi=pmi2 \
     --gres=gpu:${ngpu} \
     -n${ngpu} \
     --ntasks-per-node=${ngpu} \
     --job-name=pointsmpl \
     --kill-on-bad-exit=1 \
     -w SG-IDC1-10-51-2-74 \
     python -u train_temporal.py \
          --config cfgs/tshirt.yaml \
          --num_workers 8 \
          --batch_size 20 \
          --epoch_num 100 \
          --lr 0.001 \
          --lr_sche \
          --npoints 6890 \
          --output_dir ./output/seg_pca_shirt \
          --ckpt_name model.ckpt \
          --launcher slurm \
          --tcp_port ${tcp_port} \
          --local_rank 0 \
          --syncbn 1 \
          --T 10 \
          --GarmentPCA 1 \
          --only_seg 0 \
          --pretrained_model ./output/seg_pca_shirt/pretrain.ckpt
