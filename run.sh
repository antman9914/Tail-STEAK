DATASET='deezer'    # ['deezer', 'lastfm']
CHECKPOINT_DIR='./checkpoint_'$DATASET
OUTPUT_DIR='./output_'$DATASET
DATASET_DIR='./'$DATASET
if [ ! -d $CHECKPOINT_DIR ]
then
    mkdir $CHECKPOINT_DIR
fi
if [ ! -d $OUTPUT_DIR ]
then
    mkdir $OUTPUT_DIR
fi

NODE_FEAT_INIT=$DATASET_DIR'/init_emb.npy'
GRAPH_PATH=$DATASET_DIR'/graph.txt'

mode='train'    # ['train', 'test']

gpu_id=0
layer_num=2
input_dim=32
hidden_channel=32
sample_bound=50
batch_size=1024   
neg_sample_num=19
neg_sample_num_test=99
lr=0.001
weight_decay=0.0001

eval_per_n=50   # 400 for lastfm, 50 for deezer
epoch_num=25    # 5 for lastfm, 25 for deezer
ft_epoch=10     # 3 for lastfm, 10 for deezer

ssl_temp=0.5
ssl_reg=1      # 1 for deezer, 0.5 for lastfm
deg_t=3     # 7 for LastFM, 3 for Deezer
gamma=3     # 3 for Deezer, 4 for LastFM
U=1000      # 1000 for Deezer, 2000 for LastFM

model_type='HGN'    # ['HGN', 'LightGCN']
debias_on=1         # 1 for using Tail-STEAK, otherwise set 0

python -u main.py \
    --mode=$mode \
    --model=$model_type \
    --debias_on=$debias_on \
    --dataset=$DATASET \
    --init_node_feat=$NODE_FEAT_INIT \
    --graph_path=$GRAPH_PATH \
    --dataset_path=$DATASET_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --output_dir=$OUTPUT_DIR \
    --gpu_id=$gpu_id \
    --layer_num=$layer_num \
    --sample_bound=$sample_bound \
    --input_dim=$input_dim \
    --hidden_channel=$hidden_channel \
    --gamma=$gamma \
    --U=$U \
    --weight_decay=$weight_decay \
    --lr=$lr \
    --batch_size=$batch_size \
    --neg_sample_num=$neg_sample_num \
    --neg_sample_num_test=$neg_sample_num_test \
    --eval_per_n=$eval_per_n \
    --epoch_num=$epoch_num \
    --ft_epoch=$ft_epoch \
    --ssl_temp=$ssl_temp \
    --ssl_reg=$ssl_reg \
    --deg_t_low=$deg_t \
    
