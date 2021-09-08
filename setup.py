from setuptools import setup, find_packages


setup(
    name="fast_keywords",
    version="0.0.1",
    description="Fast keyword identification with n-gram vector string matching.",
    license="unlicensed",
    url="",
    package_dir={"fast_keywords": "fast_keywords"},
    packages=find_packages(),
    install_requires=[
        "pandas==1.2.3",
        "scikit-learn==0.23.0",
        "sparse-dot-topn==0.2.9",
        "germansentiment==1.0.6",
        "dill==0.3.1.1",
        "nltk==3.4.5",
        "tqdm==4.47.0",
    ],
)
