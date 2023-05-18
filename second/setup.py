from setuptools import setup, Extension
import pybind11

cpp_args =  ['/std:c++17']

sfc_module = Extension(
    'opencv_job',
    sources=['fun.cpp'],
    include_dirs=[pybind11.get_include(),
                  r"C:\Users\kopyl\source\repos\Dll1"],
    library_dirs=[r"C:\Users\kopyl\source\repos\Dll1\x64\Release"],
    libraries=["Dll1"], 
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='opencv_job',
    version='1.0',
    description='Python package with superfastcode2 C++ extension (PyBind11)',
    ext_modules=[sfc_module]
)

