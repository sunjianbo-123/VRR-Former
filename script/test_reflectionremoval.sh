### test on ReflectionRemovalDatasets ###
# note：train_ps: the size of the test patch is 1280
python3 test_reflectionremoval.py \
        --arch Uformer_B \
        --env _L1loss+0.1Percep+Edge+dualpath+test \
        --dataset my_syn \
        --gpu 0,1,2 \
        --patch_size 4 \
        --embed_dim 32 \
        --depths 2 2 4 6 \
        --depths_FFTblock 2 2 2 2 \
        --num_heads 2 4 4 8 \
        --win_size 8 \
        --token_mlp "g_ffn" \
        --attention_type "HiLo_attention" \
        --upsample_style "dual_upsample" \
        --downsample_style "conv_downsample" \
        --input_dir /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/reflectionremoval/my_syn/test/ \
        --result_dir /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/results/ \
        --weights /home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/logs/reflectionremoval/my_syn/Uformer_B_L1loss+0.1Percep+Edge+dualpath+test/models/model_best.pth