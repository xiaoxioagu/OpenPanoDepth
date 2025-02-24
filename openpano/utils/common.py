import yaml

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    # with open("./configs/default.yaml", 'r') as stream:
    #     dst_cfgs = yaml.safe_load(stream)
    # MergeCfgsDict(src_cfgs, dst_cfgs)
    return src_cfgs