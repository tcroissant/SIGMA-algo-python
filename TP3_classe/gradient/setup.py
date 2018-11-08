#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup

# load the version from the file
with open("VERSION", 'r') as fic:
    version = fic.read()

# set the parameter of the setup
setup(name='gradient', # define the name of the package
      version=version,
      description='Package to compute coefficients with a gradient descent technique',
      author='Antoine Dumas',
      author_email='dumas@phimeca.com',
      packages=['gradient', 'gradient.example'], # namespace of the package
      # define where the package "gradient" is located
      # and define subpackage example
      package_dir={'gradient':'src',
                   'gradient.example':'src/example'}, 
      test_suite='test', # subclass for unittest
      # some additional data included in the package
      data_files = [('.', ['VERSION']), 
                    ('data', ['data/Sigma_features.pkl',
                              'data/Sigma_labels.pkl'])],

      # List of dependancies
      install_requires= ['numpy']
      )
