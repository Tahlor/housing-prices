#!/usr/bin/env python
#import os
#del os.link
try:
    from setuptools import setup
    args = {}
except ImportError:
    from distutils.core import setup
    print("""\
*** WARNING: setuptools is not found.  Using distutils...
""")

from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

from os import path
setup(name='housing_prices',
      version='0.0.1',
      description='Model AMES Iowa housing sale prices',
      long_description= "" if not path.isfile("README.md") else read_md('README.md'),
      author='Taylor Archibald',
      author_email='tahlor@gmail.com',
      url='https://github.com/tahlor/housing-prices',
      setup_requires=['pytest-runner',],
      tests_require=['pytest','python-coveralls'],
      install_requires=[
          "numpy",
      ],
      license=['MIT'],
      packages=['housing_prices'],
      scripts=[],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: Windows',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
      ],
     )
