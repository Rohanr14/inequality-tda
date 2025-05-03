# setup.py
# Setup script for the inequality_tda package

from setuptools import setup, find_packages

setup(
    name='inequality_tda',
    version='0.1.0',
    author='Rohan Ramavajjala',
    author_email='rovajjala@gmail.com',
    description='Topological Data Analysis of Income Distribution Gaps',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Rohanr14/inequality-tda', # Placeholder - update later
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Core dependencies will be listed in requirements.txt
        # We read them from there to avoid duplication
        line.strip() for line in open('requirements.txt') if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'inequality_tda=inequality_tda.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.10',
    include_package_data=True, # To include non-code files specified in MANIFEST.in if needed
)
