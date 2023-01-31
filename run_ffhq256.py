from templates import *
from templates_latent import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--ngpus', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-o', '--output', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/diffae/ae256')
    args = parser.parse_args()
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    # batch size=32 for 2x A100 80GB GPU
    gpus = list(range(args.ngpus))
    nodes = 1
    conf = ffhq256_test(args)
    train(conf, gpus=gpus)