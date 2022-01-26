import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    version=0.1,
    name="simpleSVGD",
    author="Lars Gebraad",
    author_email="lars.gebraad@erdw.ethz.ch",
    description="A very basic implementation of SVGD, based on https://github.com/dilinwang820/Stein-Variational-Gradient-Descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/simpleSVGD",
    project_urls={
        "Bug Tracker": "https://github.com/larsgeb/simpleSVGD/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
        ]
    },
)