import os
from pathlib import Path
import h5py
from dlgo.tools.file_finder import FileFinder


def get_file_path(dir_name, file_name):
    path = Path(__file__)
    project_lvl_path = path.parents[2]
    print(project_lvl_path)
    dir_full_path = project_lvl_path.joinpath(dir_name)
    print(dir_full_path)
    file_full_path = str(dir_full_path.joinpath(file_name))
    print(file_full_path)
    if not os.path.exists(file_full_path):
        raise FileNotFoundError
    return file_full_path


def get_model_name():
    return 'model_simple_small_1000_20_epoch12_10proc.h5'


def get_exp_name(model_name):
    split_name = model_name.split('_')
    part_name = split_name[1:]
    part_name = 'exp_' + '_'.join(part_name)
    return part_name


def check_model():
    finder = FileFinder()
    model_path = finder.find_model(get_model_name())
    with h5py.File(model_path, 'r') as model:
        checker = H5ModelChecker(model)
        checker.show()


def check_exp():
    finder = FileFinder()
    exp_name = finder.get_new_prefix_name_from_model(get_model_name())
    exp_path = finder.get_exp_full_path(exp_name)
    with h5py.File(exp_path, "r") as exp:
        checker = H5ExperienceChecker(exp)
        checker.show()
        checker.short_check()


def main():
    print(f'Path.cwd(): {Path.cwd()}')
    check_exp()
    check_model()


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
        print(f'Shape[0]: {dset.shape[0]}')

    def get_max(self):
        return self.file['experience/states'].shape[0]


if __name__ == '__main__':
    main()
