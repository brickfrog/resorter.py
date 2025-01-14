from setuptools import setup, find_packages

setup(
    name="resorter-py",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "numpy",
        "pandas",
    ],
    entry_points={
        'console_scripts': [
            'resorter=resorter_py.main:main',
        ],
    },
    author="Justin Malloy",
    description="A Python implementation of Gwern's resorter(-ish)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
) 