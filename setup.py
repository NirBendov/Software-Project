from setuptools import setup, Extension

module = Extension('symnmfmodule',sources=['symnmfmodule.c', 'symnmf.c'])
setup(name='symnmfmodule',
    version='1.0',
    description='SymNMF Python C API',
    ext_modules=[module])