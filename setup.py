from setuptools import setup, find_packages

setup(
    name="extreme-evaporation-events",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
    ],
    author="Your Name",
    description="Analysis toolkit for extreme evaporation events",
    python_requires=">=3.8",
)