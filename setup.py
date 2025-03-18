"""
Setup configuration for the enhanced ensemble package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-ensemble",
    version="0.1.0",
    author="WITHIN AI Team",
    author_email="ai@within.ai",
    description="Advanced ensemble methods for machine learning with comprehensive performance monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/within-ai/enhanced-ensemble",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "pylint>=2.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-ensemble=app.models.ml.prediction.enhanced_ensemble:main",
        ],
    },
) 