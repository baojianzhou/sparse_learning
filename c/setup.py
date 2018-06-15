from distutils.core import setup, Extension

setup(name='proj_head', version='0.0.2',
      ext_modules=[Extension('proj_head', ['main.c'])])
