from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Function to parse requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line and not line.startswith("#")]

setup(
    name='polars_expr_transformer',  # Package name
    version='0.1.3',  # Package version
    description='Transform string-based expressions into Polars DataFrame operations',  # Short description
    long_description=long_description,
    long_description_content_type='text/markdown',  # Use markdown format for long description
    author='Edward van Eehoud',  # Replace with your name
    author_email='evaneechoudl@gmail.com',  # Replace with your email
    url='https://github.com/edwardvaneechoud/polars_expr_transformer',  # Replace with the URL to your package repository
    packages=find_packages(include=['polars_expr_transformer', 'polars_expr_transformer.*']),  # Automatically find package directories
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
    install_requires=parse_requirements('requirements.txt'),  # Parse requirements from requirements.txt
    extras_require={
        'dev': [
            'check-manifest',
            'flake8',
        ],
        'test': [
            'coverage',
            'pytest',
        ],
    },
    include_package_data=True,  # Include package data specified in MANIFEST.in
)