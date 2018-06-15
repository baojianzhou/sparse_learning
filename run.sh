#!/usr/bin/env bash
rm -rf dist
rm -rf build
rm -rf sparse_learning.egg-info
python3 setup.py sdist bdist_wheel
python3 setup.py sdist bdist_wheel
twine upload dist/*
