from distutils.core import setup
from setuptools import find_packages


setup(
    name="fast-keywords",
    version="0.0.1",
    description="Fast keyword identification with n-gram vector string matching.",
    url="",
    packages=find_packages(),
    install_requires=[
        "pandas==0.25.1"
    ]
)
