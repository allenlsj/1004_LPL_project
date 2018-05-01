# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="spark_lean",
    packages = ['spark_lean'],
    version="0.3.2",
    description="an interactive PySpark-based Data Cleaning Library",
    license="MIT",
    author="Shuaiji Li, Yicheng Pu, Qingtai Liu",
    author_email="sl6486@nyu.edu, yp653@nyu.edu, ql819@nyu.edu",
    url = 'https://github.com/allenlsj/Spark-lean',
    download_url = 'https://github.com/allenlsj/Spark_lean/archive/0.3.tar.gz',
    install_requires=['pyspark'],
    long_description=long_description,
    keywords = ['PySpark', 'Data Cleaning', 'Interactive'],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ]
)
