#!/usr/bin/env python

import os

from setuptools import find_packages, setup

# Prepare long description using existing docs
long_description = ""
this_dir = os.path.abspath(os.path.dirname(__file__))
doc_files = ["README.md"]
for doc in doc_files:
    with open(os.path.join(this_dir, doc), "r") as f:
        long_description = "\n".join([long_description, f.read()])

setup(
    name="target_assign_rl",
    version="0.2.0",
    description="Target assignment using reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Morphlng/target_assign_rl",
    author="Morphlng",
    author_email="morphlng@proton.me",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy<2.0.0",
        "matplotlib",
        "pettingzoo",
        "pandas",
        "pygame",
        "seaborn",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "torch": ["torch", "torchvision", "torchaudio"],
        "ray": ["ray[rllib]==2.8.1"],
        "sb3": ["stable-baselines3", "sb3-contrib"],
    },
    keywords="gym, reinforcement learning, multi-agent, target assignment",
    project_urls={
        "Source": "https://github.com/Morphlng/target_assign_rl",
        "Report bug": "https://github.com/Morphlng/target_assign_rl/issues",
        "Author website": "https://github.com/Morphlng",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
