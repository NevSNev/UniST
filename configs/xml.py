import yaml
import os
def read_args(cfg):
    with open(cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_cfg

