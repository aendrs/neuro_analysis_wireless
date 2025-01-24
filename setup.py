from setuptools import setup, find_packages

setup(
    name="neuro_analysis_wireless",
    version="0.1.0",
    description="A toolkit for neural data analysis and visualization.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "umap-learn",
        "seaborn",
        "matplotlib",
        "python-pptx",
        "networkx",
        "cylouvain",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)