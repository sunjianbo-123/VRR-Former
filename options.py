import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default = 'my_data')
        parser.add_argument('--pretrain_weights', type=str, default='./pre-trained_weights/Uformer_B_GoPro.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1,2', help='GPUs')

        # args for test
        parser.add_argument('--input_dir', default='/data1/wangzd/datasets/deblurring/GoPro/test/',
                            type=str, help='Directory of test images')
        parser.add_argument('--result_dir',
                            default='/data1/wangzd/uformer_cvpr/results_release/deblurring/GoPro/Uformer_B/',
                            type=str, help='Directory for results')
        parser.add_argument('--weights',
                            default='/data1/wangzd/uformer_cvpr/logs/motiondeblur/GoPro/Uformer_B_1129/models/model_best.pth',
                            type=str, help='Path to weights')
        parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
        parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')


        # 修改模型训练之前需要修改对应的模型名称
        parser.add_argument('--arch', type=str, default ='Uformer_B',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='deblurring',  help='image restoration mode')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
        parser.add_argument('--env', type=str, default ='',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
        parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')
        parser.add_argument('--patch_size', type=int, default=4, help='patch size of transformer')
        parser.add_argument('--embed_dim', type=int, default=96, help='dim of emdeding features')
        parser.add_argument('--depths', type=int, metavar='N', nargs='+', default=[2, 2, 4, 8],    help='depths of swin-transformer encoder')
        parser.add_argument('--depths_FFTblock', type=int, metavar='N', nargs='+', default=[2, 2, 2, 2],help='depths of spectral block')
        parser.add_argument('--num_heads', type=int, metavar='N', nargs='+', default=[2, 4, 6, 8], help='num_heads of MSA')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--mlp_ratio', type=float, default=4., help='mlp ratio of transformer ffn')
        parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='g_ffn', help='ffn/leff/g_ffn')
        parser.add_argument('--attention_type', type=str,default='HiLo_attention', help='HiLo_attention/w-msa')
        parser.add_argument('--final_upsample', type=str, default='dual_upsample',help='dual_upsample/transpose_conv_upsample, upsample method of final output')
        parser.add_argument('--upsample_style', type=str, default='dual_upsample',help='dual_upsample/transpose_conv_upsample')
        parser.add_argument('--downsample_style', type=str, default='patch_merging',help='patch_merging/conv_downsample')




        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=256, help='patch size of validation sample')


        # action='store_true': 告诉解析器如果命令行中存在--resume参数，则将此参数的值设置为True，否则为False
        # 这是处理存在或不存在的布尔标志的常见方式
        # default = False: 默认值False。因此，如果命令行中没有提供 - -resume参数，那么 - -resume的值将是False
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default ='/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/deblurring/GoPro/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/deblurring/GoPro/test',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        # DDP
        parser.add_argument("--local_rank", type=int, default=-1, help='DDP parameter, do not modify') #不需要赋值，启动命令 torch.distributed.launch会自动赋值
        parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
        parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
        return parser
