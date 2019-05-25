import sys
import os
import yaml


def load_config():
    with open("configuration.yaml", "rb") as ff:
        config = yaml.load(ff)
    print(config)
    return config

