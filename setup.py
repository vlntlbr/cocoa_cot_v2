"""
Setup configuration for CoCoA-CoT: Extending Confidence and Consistency-Based
Uncertainty Quantification to Reasoning Language Models.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cocoa_cot",
    version="0.1.0",
    description=(
        "CoCoA-CoT: Extending Confidence and Consistency-Based Uncertainty "
        "Quantification to Reasoning Language Models"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CoCoA-CoT Authors",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cocoa-run-main=cocoa_cot.experiments.run_main:app",
            "cocoa-run-ablation=cocoa_cot.experiments.run_ablation:app",
            "cocoa-run-alpha=cocoa_cot.experiments.run_alpha:app",
            "cocoa-run-calibration=cocoa_cot.experiments.run_calibration:app",
            "cocoa-run-blackbox=cocoa_cot.experiments.run_blackbox:app",
            "cocoa-run-light=cocoa_cot.experiments.run_light:app",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
