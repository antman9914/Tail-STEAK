# Tail-STEAK: Improve Friend Recommendation for Tail Users via Self-Training Enhanced Knowledge Distillation

This is the code for our paper [*"Tail-STEAK: Improve Friend Recommendation for Tail Users via Self-Training Enhanced Knowledge Distillation"*](https://ojs.aaai.org/index.php/AAAI/article/view/28737) accepted by AAAI 2024. Our code is implemented in PyTorch 1.10.2 and PyTorch-Geometric 2.2.0. If you would like to use your own datasets, it's suggested to install numba, gensim and scipy as well.

## Datasets

We use Deezer and Last.FM for evaluation. We have uploaded preprocessed data in both [Quark Drive](https://pan.quark.cn/s/d97dcea27d07) and Google Drive([Deezer](https://drive.google.com/drive/folders/1xn8jrKm-cdEQxBsxKV2v5LCQsym7QtsW?usp=sharing) and [Last.FM](https://drive.google.com/drive/folders/1ayc5rT0awYfhXLo88X7WkLk0WfZRoKKR?usp=sharing)). Please keep the provided file structure unchanged. Each dataset is composed of 2 files, where `graph.txt` provides graph data, `init_emb.npy` is Node2Vec-based user ID embedding initialization. If you would like to use your own datasets, please refer to the data format in `graph.txt`. We have also provided Node2Vec-based ID embedding initialization code in `preprocess/dense_feat_extract.py`.

## Model Training

To train our model with default hyper-parameters, you can directly run `run.sh`. Hyperparameter settings for both adopted datasets have been listed in the comments within the provided shell script. To skip the training procedure and perform model inference only, you can change the value of `mode` into `test`. 

## Citation

If you found the codes are useful, please cite our paper:

```
@article{ma2024tailsteak,
 title={Tail-STEAK: Improve Friend Recommendation for Tail Users via Self-Training Enhanced Knowledge Distillation},
 volume={38},
 number={8},
 journal={Proceedings of the AAAI Conference on Artificial Intelligence},
 author={Ma, Yijun and Li, Chaozhuo and Zhou, Xiao},
 year={2024},
 pages={8895-8903}
}
```
