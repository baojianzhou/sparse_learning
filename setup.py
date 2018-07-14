# -*- coding: utf-8 -*-
"""
Some documents will be added later.
"""
import os
import numpy
from os import path
from setuptools import setup
from distutils.core import Extension

here = path.abspath(path.dirname(__file__))


def get_c_sources(folder, include_headers=False):
    """Find all C/C++ source files in the `folder` directory."""
    allowed_extensions = [".c", ".C", ".cc", ".cpp", ".cxx", ".c++"]
    if include_headers:
        allowed_extensions.extend([".h", ".hpp"])
    sources = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1]
            if ext in allowed_extensions:
                sources.append(os.path.join(root, name))
    return sources


# calling the setup function
setup(
    # sparse_learning package.
    name='sparse_learning',
    # current version is 0.2.1
    version='0.2.1',
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
    classifiers=("Programming Language :: Python :: 2",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    # specify requirements of your package here
    # will add openblas in later version.
    install_requires=['numpy'],
    headers=['cpp/head_tail.h', 'cpp/pcst_fast.h'],
    # define the extension module
    ext_modules=[Extension('proj_module',
                           sources=['cpp/main_wrapper.cpp'],
                           language="c++",
                           extra_compile_args=['-std=c++11', '-lpython2.7'],
                           include_dirs=[numpy.get_include(),
                                         '/usr/include/', 'cpp/include/'],
                           )],
    keywords='sparse learning, structure sparsity, head/tail projection')
