from templates import *
from templates_latent import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-o', '--output', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/diffae/test')
    args = parser.parse_args()
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0,1]
    conf = ffhq128_test(args)
    train(conf, gpus=gpus)

    # # infer the latents for training the latent DPM
    # # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [0, 1, 2, 3]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # # train the latent DPM
    # # NOTE: only need a single gpu
    # gpus = [0]
    # conf = ffhq128_autoenc_latent()
    # train(conf, gpus=gpus)

    # # unconditional sampling score
    # # NOTE: a lot of gpus can speed up this process
    # gpus = [0, 1, 2, 3]
    # conf.eval_programs = ['fid(10,10)']
    # train(conf, gpus=gpus, mode='eval')