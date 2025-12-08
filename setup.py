from setuptools import setup, find_packages

setup(
    name="elo-prediction",
    version="0.1.0",
    description="Machine learning model for predicting chess player Elo ratings from game sequences",
    author="Logan Druley, Alberto Garcia, Roberto Palacios",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.3",
        "pandas>=2.1.4",
        "python-chess>=1.10.0",
        "scikit-learn>=1.3.2",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "tqdm>=4.66.1",
        "zstandard>=0.22.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
        ]
    },
)
