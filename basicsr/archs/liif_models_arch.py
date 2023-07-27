import copy
from basicsr.utils.registry import ARCH_REGISTRY
models = {}


def register(name):

    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    # 从ARCH_REGISTRY中获取模型
    model_cls = ARCH_REGISTRY.get(model_spec['name'])
    model = model_cls(**model_args)

    if load_sd:
        model.load_state_dict(model_spec['sd'])

    return model
