#!/usr/bin/env python

from distutils.core import setup

setup(
    name='ddd',
    version='0.1.0',
    author='Josh Gardner',
    author_email='jpgard@cs.washington.edu',
    packages=['ddd'],
    url='https://github.com/jpgard/driving-with-data-detroit',
    license='LICENSE',
    description='Driving with Data in Detroit',
    long_description=open('README.md').read(),
    install_requires=['pandas', 'scipy', 'numpy'],
)
