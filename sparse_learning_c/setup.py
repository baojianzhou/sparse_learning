# -*- coding: utf-8 -*-
"""
Some documents will be added later.
"""
import numpy
from os import path
from setuptools import setup
from distutils.core import Extension

here = path.abspath(path.dirname(__file__))

# calling the setup function
setup(
    # sparse_learning package.
    name='sparse_learning',
    # current version is 0.2.0
    version='0.2.0',
    # this is a wrapper of head and tail projection.
    description='A wrapper for sparse learning algorithms.',
    # a long description should be here.
    long_description='This package collects sparse learning algorithms.',
    # url of github projection.
    url='https://github.com/baojianzhou/sparse_learning.git',
    # number of authors.
    author='Baojian Zhou',
    # my email.
    author_email='bzhou6@albany.edu',
    include_dirs=[numpy.get_include()],
    license='MIT',
    packages=['sparse_learning'],
    classifiers=("Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    # specify requirements of your package here
    # will add openblas in later version.
    install_requires=['numpy'],
    headers=['cpp/head_tail.h', 'cpp/pcst_fast.h'],
    # define the extension module
    ext_modules=[Extension('sparse_learning_module',
                           sources=['src/python_wrapper.c'],
                           language="c",
                           extra_compile_args=['-std=c11', '-lpython3.6'],
                           include_dirs=[numpy.get_include(), '/src'], )],
    keywords='sparse learning, structure sparsity, head/tail projection')
