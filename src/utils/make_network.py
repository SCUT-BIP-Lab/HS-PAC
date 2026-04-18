import importlib.util
import torch


def make_model(conf):
    model_path = conf['model_path']
    model_name = conf['model_name']
    # module = importlib.import_module(f'..model.{model_path}', 'src.utils')
    module = importlib.import_module(f'.model.{model_path}', 'src')
    MyModel = getattr(module, model_name)
    model = MyModel(conf)
    return model
