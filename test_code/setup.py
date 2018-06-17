from distutils.core import setup, Extension

import numpy

# define the extension module
lasso_module = Extension('test', sources=['lasso.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=['cblas'],
                         extra_compile_args=["-Wall", "-lcblas"])
print(numpy.get_include())
online_graph_ghtp_module = Extension('online_graph_ghtp', sources=['online_graph_ghtp.c'],
                                     include_dirs=[numpy.get_include(),
                                                   '/usr/local/openblas/include'],
                                     libraries=['cblas', '/usr/local/openblas/lib'],
                                     extra_compile_args=["-Wall", "-lcblas"])
# run the setup
setup(ext_modules=[lasso_module, online_graph_ghtp_module])

# run the following command
# python setup.py build_ext --inplace
