python -m torch.distributed.launch \
  --nproc_per_node=3 \
  train/train_demesh.py \
  --arch Uformer_B \
  --mode demesh \
  --dataset X-EUV \
  --env _0706 \
  --batch_size 4 \
  --gpu 0,1,2 \
  --patch_size 4 \
  --embed_dim 96 \
  --depths 2 2 4 8 \
  --num_heads 2 4 6 8 \
  --win_size 8 \
  --token_mlp "g_ffn" \
  --attention_type "HiLo_attention" \
  --final_upsample "dual_upsample" \
  --upsample_style "dual_upsample" \
  --downsample_style "patch_merging" \
  --train_ps 256 \
  --val_ps 256 \
  --nepoch 3000 \
  --checkpoint 100 \
  --warmup \
  --distribute \
  --train_dir /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/demesh/X-EUV/train \
  --val_dir /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/demesh/X-EUV/val \
  --pretrain_weights /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/logs/demesh/X-EUV/Uformer_B_0706/models/model_latest.pth  \
  --save_dir /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/logs/


