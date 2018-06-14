from setuptools import setup

# reading long description from file
with open('DESCRIPTION.txt') as f:
    long_description = f.read()

# specify requirements of your package here
REQUIREMENTS = ['numpy']

# some more details
CLASSIFIERS = classifiers = (
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",)

# calling the setup function
setup(name='sparse_learning',
      version='0.0.1',
      description='A wrapper for sparse learning algorithms.',
      long_description=long_description,
      url='https://github.com/baojianzhou/sparse_learning.git',
      author='Baojian Zhou',
      author_email='bzhou6@albany.edu',
      license='MIT',
      packages=['sparse_learning'],
      classifiers=CLASSIFIERS,
      install_requires=REQUIREMENTS,
      keywords='sparse learning, structure sparsity, head/tail projection')
