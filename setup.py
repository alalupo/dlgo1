from setuptools import setup, find_packages

setup(
    name='go_bot',
    version='0.1',
    description='Deep Learning for the Game of Go',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'flask',
        'h5py',
        'keras',
        'matplotlib',
        'dlgo',
    ]
)

