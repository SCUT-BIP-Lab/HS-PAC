# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import argparse
import os.path as osp

# Parse GPU argument before importing torch to ensure CUDA_VISIBLE_DEVICES takes effect
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=None, help="GPU IDs (e.g., '0' or '0,1,2,3')")
args, remaining = parser.parse_known_args()

# Set visible GPUs before importing torch
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# print("CUDA_VISIBLE_DEVICES set to:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

# Now import torch after CUDA_VISIBLE_DEVICES is set
import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
import torch.distributed as dist
import tensorboardX

DIR = osp.realpath(osp.dirname(__file__))
sys.path.insert(0, DIR)
sys.path.insert(0, osp.join(DIR, '../'))
sys.path.insert(0, osp.join(DIR, '../../'))

from trainer import Trainer
from src.utils.util import parse_conf, set_random_seed
from src.utils.logger import config_logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, default='none', help="Path to configuration file")
    parser.add_argument("--mode", type=str, default='train', help="Mode: train, eval, or inference")
    parser.add_argument("--verbose", action='store_true', default=False, help="Enable debug logging")
    parser.add_argument("--gpu", type=str, default=None, help="GPU IDs (e.g., '0' or '0,1,2,3')")
    args = parser.parse_args()
    return args

# python src/main/train.py --conf_file path/to/config.conf
#
# DDP
# torchrun --nproc_per_node=4 src/main/train.py --conf_file path/to/config.conf


def setup_distributed():
    """
    Setup for torchrun distributed training.
    Returns: (rank, local_rank, world_size, is_distributed)
    torchrun can automatically set the following environment variables:
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        is_distributed = world_size > 1

        if is_distributed:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')

        return rank, local_rank, world_size, is_distributed
    else:
        # Single GPU or CPU mode
        return 0, 0, 1, False


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed()

    # Parse arguments
    args = get_args()

    # Setup distributed training
    rank, local_rank, world_size, is_distributed = setup_distributed()
    print("Distributed training:", is_distributed, "Rank:", rank, "Local rank:", local_rank, "World size:", world_size)

    # Load configuration file
    if args.conf_file == 'none':
        root_dir = r'/home/wxl/codeForLYX/dhg224/'
        conf_relative_path = r'conf/lyxconf/SPNet_all_br.conf'
        conf_file = osp.join(root_dir, conf_relative_path)
    else:
        conf_file = args.conf_file

    # Parse configuration
    conf = parse_conf(conf_file, True)
    conf["verbose"] = args.verbose
    conf["mode"] = args.mode
    conf["local_rank"] = local_rank
    conf["rank"] = rank
    conf["world_size"] = world_size
    conf["distributed"] = is_distributed

    # Print training info on main process
    if rank == 0:
        print("#" * 40)
        print("%s is running..." % conf["task_name"])
        print("#" * 40)
        print(f'''
        Mode:            {args.mode}
        Distributed:     {is_distributed}
        World size:      {world_size}
        Local rank:      {local_rank}
        Epochs:          {conf["num_epochs"]}
        Batch size:      {conf["batchsize_train"]}
        Learning rate:   {conf["learning_rate"]}
        Resume:          {"load_from" in conf}
    ''')

    # Backup source code (only on main process)
    if rank == 0 and args.mode == "train" :

        # check dec
        if os.path.exists(conf["log_file_dir"]) and args.mode == 'train' :
            print("log file dir exists, remove it and create a new one?")
            if input("y/n") == "n":
                exit()
            os.system("rm -rf {}".format(conf["log_file_dir"]))
            os.system("rm -rf {}".format(conf["tb_file_dir"]))
            os.system("rm -rf {}".format(conf["tb_file_dir"].replace('tb', 'backup')))

        src_dir = os.path.join(conf["home_self"], "src")
        backup_dir = conf["tb_file_dir"].replace('tb', 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        os.system("cp -r %s %s" % (src_dir, backup_dir))
        os.system("cp %s %s" % (conf_file, backup_dir))

    # Setup logger
    logger = config_logging(conf["log_file_dir"], "DEBUG", False)
    logger.info("task: " + str(conf))
    conf["logger"] = logger

    # Setup tensorboard
    tb_writer = tensorboardX.SummaryWriter(conf["tb_file_dir"])
    conf["tb_writer"] = tb_writer

    # Setup device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    # Train or evaluate
    if args.mode == "train":
        trainer = Trainer(conf)
        trainer.train()

    # load_from direct to a checkpoint for evaluation
    elif args.mode == "eval" or args.mode == "evaluate":
        args.mode="evaluate"
        pretrained_parms_path = conf.get('load_from')
        if pretrained_parms_path :
            trainer = Trainer(conf)
            trainer.evaluate(epoch=0)

    elif args.mode == "inference":
        pretrained_parms_path = conf.get('load_from')
        if pretrained_parms_path :
            trainer = Trainer(conf)
            trainer.inference(data_save_dir=conf.get('data_save_dir', None),img_folder=conf.get('img_folder', None))

    elif args.mode == "gradcam":
        pretrained_parms_path = conf.get('load_from')
        if pretrained_parms_path:
            trainer = Trainer(conf)
            trainer.visualize_grad_cam(
                save_dir=conf.get('data_save_dir', './gradcam_output'),
                target_layer_name=conf.get('gradcam_target_layer', 'backbone'),
                max_samples=int(conf.get('gradcam_max_samples', 200))
            )

    elif args.mode == "model_analysis" or args.mode == "analysis":
        # 模型分析模式：分析参数量、GFLOPs和推理速度
        print("\n" + "="*70)
        print("🔍 模型分析模式")
        print("="*70 + "\n")

        # 启用模型分析
        conf["model_analysis"] = True

        # 创建Trainer（会自动触发模型分析）
        trainer = Trainer(conf)

        print("\n" + "="*70)
        print("✅ 模型分析完成！")
        print("="*70 + "\n")

    else:
        raise NotImplementedError

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

