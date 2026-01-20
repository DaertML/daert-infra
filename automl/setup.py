from setuptools import setup, find_packages

setup(
    name="automl",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1.7",
        "pyyaml>=6.0.1",
        "rich>=13.6.0",
        "docker>=6.1.3",
        "paramiko>=3.3.1",
        "mlflow>=2.9.2",
        "scikit-learn>=1.3.2",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "automl=src.cli.cli:main",
        ],
    },
    include_package_data=True,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
)
