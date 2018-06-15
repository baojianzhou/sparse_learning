#!/usr/bin/env bash
rm -rf dist
rm -rf build
rm -rf sparse_learning.egg-info
python setup.py sdist bdist_wheel
python setup.py sdist bdist_wheel
twine upload dist/*.tar.gz
