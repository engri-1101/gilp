import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gilp",
    version="2.0.0",
    author="Henry Robbins",
    author_email="hw.robbins@gmail.com",
    description="A Python package for visualizing the geometry of linear programs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/engri-1101/gilp.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19',
        'networkx>=2.0',
        'typing>=3.7',
        'scipy>=1.3',
        'plotly>=5'
    ],
    extras_require= {
        "dev": ['pytest>=5',
                'mock>=3',
                'coverage>=4.5',
                'tox>=3']
    },
    python_requires='>=3.5',
)