from setuptools import setup

# specify requirements of your package here
requirements = ['numpy']

# some more details
classifiers = (
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",)

# calling the setup function
setup(name='sparse_learning',
      version='0.0.2',
      description='A wrapper for sparse learning algorithms.',
      long_description='This package collects sparse learning algorithms.',
      url='https://github.com/baojianzhou/sparse_learning.git',
      author='Baojian Zhou',
      author_email='bzhou6@albany.edu',
      license='MIT',
      packages=['sparse_learning'],
      classifiers=classifiers,
      install_requires=requirements,
      keywords='sparse learning, structure sparsity, head/tail projection')
