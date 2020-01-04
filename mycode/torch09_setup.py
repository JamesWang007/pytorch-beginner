'''
├── setup.py
├── src
│   └── namespace
│       └── mypackage
│           ├── __init__.py
│           └── mod1.py
└── tests
    └── test_mod1.py
'''


from setuptools import setup, find_namespace_packages

setup(
     name = "namespace.mypackage",
     version = "0.1",
     package_dir = {'': 'src'},
     #packages = find_namespace_packages(include=['namespace.*'])
     packages = find_namespace_packages(where='src')
)
