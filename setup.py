#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Monte Carlo Contracts setup script.
"""

from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["numpy>=1.24", "pandas>=1.5"]

test_requirements = []

setup(
    author="luphord",
    author_email="luphord@protonmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="""Composable financial contracts with Monte Carlo valuation """,
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    data_files=[(".", ["LICENSE", "HISTORY.md"])],
    keywords="composable financial contracts Monte Carlo method",
    name="monte-carlo-contracts",
    packages=["mcc", "mcc.pricing_models"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/luphord/monte-carlo-contracts",
    version="0.9.0",
    zip_safe=True,
)
