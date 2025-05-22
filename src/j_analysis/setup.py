from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# OpenMP flags for different compilers
extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']

# Check for NumPy version
numpy_version = numpy.__version__
if numpy_version >= '2.0.0':
    print(f"Warning: You're using NumPy version {numpy_version}, which may not be compatible with older modules.")

# Define the extension module
ext_modules = [
    Extension(
        "pet_calc",
        sources=["pet_calc.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(ext_modules, language_level="3"),
)