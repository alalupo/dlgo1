import os
from pathlib import Path

import h5py


def get_model_path(dir_name, file_name):
    path = Path(__file__)
    project_lvl_path = path.parent
    model_dir_full_path = project_lvl_path.joinpath(dir_name)
    model_path = str(model_dir_full_path.joinpath(file_name))
    if not os.path.exists(model_path):
        raise FileNotFoundError
    return model_path


def check_model():
    model_dir = 'checkpoints'
    model_name = 'model_simple_small_250_50_epoch50_37proc.h5'
    model_path = get_model_path(model_dir, model_name)
    with h5py.File(model_path, 'r') as model:
        checker = H5ModelChecker(model)
        checker.show()


def check_exp():
    exp_h5file = 'exp10.h5'
    with h5py.File(exp_h5file, "r") as exp:
        checker = H5ExperienceChecker(exp)
        checker.show()
        checker.short_check()


def main():
    check_exp()


class H5ModelChecker:
    def __init__(self, h5file):
        self.file = h5file
        self.grp_model_weights_name = 'model_weights'
        self.grp_optimizer_weights_name = 'optimizer_weights'
        if self.grp_model_weights_name in h5file.keys():
            self.grp_model_weights = h5file[self.grp_model_weights_name]
        if self.grp_optimizer_weights_name in h5file.keys():
            self.grp_optimizer_weights = h5file[self.grp_optimizer_weights_name]

    def printname(self, name):
        print(f'> {name}')

    def show(self):
        print(f'********************************************')
        print(f'Keys: {list(self.file.keys())}')
        print(f'>>>Visiting keys...')
        print(f'Model weights:')
        self.grp_model_weights.visit(self.printname)
        print(f'Optimizer weights:')
        self.grp_optimizer_weights.visit(self.printname)
        print(f'********************************************')

    def get_size(self):
        pass


class H5ExperienceChecker:
    def __init__(self, h5file):
        self.file = h5file
        self.group_name = 'experience'
        if self.group_name in h5file.keys():
            self.grp = h5file[self.group_name]

    def printname(self, name):
        print(name)

    def show(self):
        print(f'********************************************')
        print(f'Keys: {list(self.file.keys())}')
        print(f'>>>Visiting keys...')
        self.grp.visit(self.printname)
        print(f'********************************************')

    def check(self):
        dset1 = self.file['experience/states']
        print(f'states shape: {dset1.shape}')
        print(f'states dtype: {dset1.dtype}')
        dset2 = self.file['experience/actions']
        print(f'actions shape: {dset2.shape}')
        print(f'actions dtype: {dset2.dtype}')
        dset3 = self.file['experience/rewards']
        print(f'rewards shape: {dset3.shape}')
        print(f'rewards dtype: {dset3.dtype}')
        dset4 = self.file['experience/advantages']
        print(f'advantages shape: {dset4.shape}')
        print(f'advantages dtype: {dset4.dtype}')

    def short_check(self):
        dset = self.file['experience/states']
        print(f'Size: {dset.shape[0]}')


if __name__ == '__main__':
    main()
