import os
from pathlib import Path
import shutil


class FileFinder:
    def __init__(self):
        self.project_path = Path.cwd()
        self.model_dir = self.project_path / 'models'
        self.exp_dir = self.project_path / 'exp'
        self.data_dir = self.project_path / 'data'
        self.dlgo_dir = self.project_path / 'dlgo'

    def find_model(self, model_name):
        full_path = self.get_model_full_path(model_name)
        if not os.path.exists(full_path):
            raise ModuleNotFoundError
        else:
            return full_path

    def get_model_full_path(self, model_name):
        return self.model_dir.joinpath(model_name)

    def find_exp(self, exp_name):
        full_path = self.get_exp_full_path(exp_name)
        if not os.path.exists(full_path):
            raise ModuleNotFoundError
        else:
            return full_path

    def get_exp_full_path(self, exp_name):
        full_path = self.exp_dir.joinpath(exp_name)
        return full_path

    def copy_model_and_get_path(self, model_name, prefix='copy_'):
        if not os.path.exists(self.model_dir.joinpath(model_name)):
            raise FileNotFoundError
        model_path = self.model_dir.joinpath(model_name)
        new_name = self.get_new_prefix_name_from_model(model_name, prefix)
        copy_path = self.model_dir.joinpath(new_name)
        shutil.copy(str(model_path), str(copy_path))
        if not os.path.exists(copy_path):
            raise FileNotFoundError
        return copy_path

    @staticmethod
    def get_new_prefix_name_from_model(model_name, prefix='exp_'):
        split_name = model_name.split('_')
        core = split_name[1:]
        exp_name = prefix + '_'.join(core)
        return exp_name
