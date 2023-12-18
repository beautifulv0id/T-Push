from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='Tshape',
    version='1.0.0',
    packages=['.'],
    include_package_data=True,
)

