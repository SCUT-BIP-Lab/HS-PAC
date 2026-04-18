# -*- coding: utf-8 -*-
import random
import re
import os

import torch
import numpy as np




def parse_conf(conf_path, variable=True):
    conf_dict = {}
    with open(conf_path, 'r', encoding='utf-8') as filein:
        for line in filein:
            line = line.strip()
            if len(line) == 0 or line[0] == "#" or line[0] == "[":
                continue
            line = line.split("#")[0].strip()  # 切除#后注释信息
            data = line.split("=")
            assert len(data) == 2  # =左右应该是键和值
            key = data[0].strip()
            values = data[1].strip()
            conf_dict[key] = values

    if variable:  # change env names to explicit dirs,将以${}形式表示的地址描述替换成真实的地址
        regex = re.compile(r"\$\{.+?\}")
        for key in conf_dict:
            value = conf_dict[key]
            s = regex.findall(value)
            if s:
                for match_str in s:
                    to_replce = match_str
                    var_key = to_replce[2:-1]
                    if var_key in conf_dict:
                        assert regex.search(conf_dict[var_key]) is None
                        value = value.replace(to_replce, conf_dict[var_key].rstrip('/'))
                        conf_dict[key] = value

    # bool
    for key, value in conf_dict.items():
        if not key == 'description':
            if ',' in value:
                value = [v.strip() for v in value.split(',')]
                try:
                    value.remove('')
                except ValueError:
                    pass
            else:
                value = value.strip()
            if value == "True":
                value = True
            if value == "False":
                value = False
            conf_dict[key] = value
    try:
        conf_dict['modality'] = [conf_dict['modality']] if isinstance(conf_dict['modality'], str) else conf_dict['modality']
        conf_dict["frames_root"] = [conf_dict["frames_root"]] if isinstance(conf_dict["frames_root"], str) else conf_dict["frames_root"]
        conf_dict["eval_label_files"] = [conf_dict["eval_label_files"]] if isinstance(conf_dict["eval_label_files"], str) else conf_dict[
            "eval_label_files"]
    except Exception as e:
        print("Error in parsing the configuration file: ", e)

    return conf_dict


def set_random_seed(seed=7):
    # pass
    # 固定随机参数
    random.seed(seed)
    np.random.seed(seed)

    # os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    #
    #
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #
    #
    # torch.backends.cudnn.benchmark = False  # 卷积算法机制
    # torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    # # torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
