import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gilp",
    version="0.0.1-rc-1",
    author="Henry Robbins",
    author_email="hwr26@cornell.edu",
    description="A Python package for visualizing the geometry of linear programs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henryrobbins/gilp.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'typing', 'scipy', 'plotly'
    ],
    python_requires='>=3.6',
)