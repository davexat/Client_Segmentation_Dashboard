import os
from setuptools import setup, find_packages

def readme() -> str:
    """Utility function to read the README.md.

    Used for the `long_description`. It's nice, because now
    1) we have a top level README file and
    2) it's easier to type in the README file than to put a raw string in below.

    Returns:
        String of the README.md content.
    """
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        return f.read()

setup(
    name='customer_segmentation',
    version='0.1.0',
    author='David Sandoval, Jose Toapanta',
    author_email='d4vidsandoval@gmail.com',
    description=(
        'Customer segmentation project for e-commerce using data science. '
        'Includes exploratory analysis, clustering, and an interactive '
        'Streamlit dashboard to visualize key metrics and compare customer segments.'
    ),
    python_requires='>=3.10', 
    url='https://github.com/davexat/Client_Segmentation_Dashboard',
    packages=find_packages(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
