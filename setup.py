"""
@author scantleb
@created 30/09/2020

@brief setup.py script for pip installation of gnina_tensorflow including python
C++ bindings.

This script is a modified version of the pybind11 cmake example found at
https://github.com/pybind/cmake_example.
"""

import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\" -O3 -Wall'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())#
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


setup(
    name='gnina_tensorflow',
    version='0.1',
    author='Jack Scantlebury',
    author_email='jack.scantlebury@gmail.com',
    description='Tensorflow implementation of gnina and other tools',
    long_description='',
    packages=['autoencoder', 'calculate_distances', 'classifier'],
    package_dir={'calculate_distances': './cpp/src/calculate_distances'},
    ext_modules=[CMakeExtension('calculate_distances')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
