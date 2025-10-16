import numpy as np
import os
import sys
def setup_gpu_before_torch():
    """在导入 torch 之前设置 GPU 环境变量"""
    import argparse
    
    # 简单解析 GPU 参数
    if '--gpu' in sys.argv:
        try:
            gpu_idx = sys.argv[sys.argv.index('--gpu') + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
            print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_idx}")
        except (IndexError, ValueError):
            print("Warning: Invalid --gpu argument")

setup_gpu_before_torch()

import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)



