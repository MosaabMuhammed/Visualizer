from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="visualizer",
    version="0.0.2",
    description="Automate the process of visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MosaabMuhammed/visualizer",
    author="Mosaab Muhammad",
    author_email="mosaabmuhammed@outlook.com",
    license="MIT",
    py_modules=["visualizer"],
    package_dir={'': 'src'},
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=["matplotlib",
                      "seaborn",
                      "wordcloud",
                      "pandas",
                      "math",
                      "sklearn",
                      "termcolor"]
    python_requires='>=3.6'
)