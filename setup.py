from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="visualizer",
    version="0.0.1",
    description="Automate the process of visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MosaabMuhammed/visualizer",
    author="Mosaab Muhammad",
    author_email="mosaabmuhammed@outlook.com",
    py_modules=["visualizer"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)