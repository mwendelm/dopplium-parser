from setuptools import setup, find_packages

setup(
    name="dopplium-parser",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    python_requires=">=3.7",
)

