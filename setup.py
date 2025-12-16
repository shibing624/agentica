# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = ""
exec(open('agentica/version.py').read())

if sys.version_info < (3, 10):
    sys.exit('Sorry, Python >= 3.10 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='agentica',
    version=__version__,
    description='AI Agent SDK',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/agentica',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.10.0",
    entry_points={"console_scripts": ["agentica = agentica.cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='Agentica,Agent Tool,action,agent,agentica',
    install_requires=[
        "httpx",
        "loguru",
        "beautifulsoup4",
        "openai",
        "python-dotenv",
        "pydantic",
        "requests",
        "sqlalchemy",
        "scikit-learn",
        "markdownify",
        "tqdm",
        "rich",
        "pyyaml",
        "mcp",
        "puremagic",
        "qdrant-client",
        "langfuse",
        "python-frontmatter",
    ],
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*', 'docs']),
    package_dir={'agentica': 'agentica'},
    package_data={'agentica': ['**/*.py', '**/*.md', '**/*.js']},
)
