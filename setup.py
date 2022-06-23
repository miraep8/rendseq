# -*- coding: utf-8 -*-
"""set package information for rendseq pip package."""
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rendseq",
    version="0.1.5",
    description="Package for RendSeq Data Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miraep8/rendseq",
    author="Mirae Parker",
    author_email="rendseq@mit.edu",
    license="MIT",
    packages=["rendseq"],
    python_requires=">=3.6",
)
