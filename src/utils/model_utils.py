'''
@Project ：SPNet 
@File    ：model_utils.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2025/11/25 16:44
'''
import importlib.util


# make model from configuration
# from src import model.<model_path> import <model_name>
def make_model(conf):
    model_path = conf['model_path']
    model_name = conf['model_name']
    # module = importlib.import_module(f'..model.{model_path}', 'src.utils')
    module = importlib.import_module(f'.model.{model_path}', 'src')
    MyModel = getattr(module, model_name)
    model = MyModel(conf)
    return model


# load pretrained model parameters to current model，compare the parameter names and sizes to ensure correct loading
def load_pretrained_params(state_load, state_cur, is_distributed=False):
    if is_distributed:
        state_dict = {k.replace('module.', ''): v for k, v in
                      state_load.items()}  # if the loaded model is trained by multi-GPU, need to remove 'module.' for single GPU loading and inference
    else:
        state_dict = state_load
    pretrained_dict = {k: v for k, v in state_dict.items() if
                       k in state_cur and v.size() == state_cur[k].size()}  # 同时存在于当前模型（实例）和预训练模型中的参数

    wasted_module = [(k, v.size()) for k, v in state_dict.items() if
                     k not in state_cur or v.size() != state_cur[k].size()]  # 找出重载模型中有，而当前模型中没有的，或大小不同的模块
    print("wasted_module: {}".format(wasted_module))  # 打印出来看哪些参数没有load进去

    missing_modlue = [(k, v.size()) for k, v in state_cur.items() if
                      k not in state_dict or v.size() != state_dict[k].size()]  # 找出当前模型中存在,重载模型不存在的模块
    print("missing_modlue: {}".format(missing_modlue))
    state_cur.update(pretrained_dict)  # 更新要读取的参数 为 模型(实例) 和 模型参数能对应上的 参数
    # self.model.load_state_dict(state_cur)  # 读取模型
    # if self.mode == "train":
    #     print("load:{}".format([(k, v.size()) for k, v in state_cur.items()]))
    return state_cur
