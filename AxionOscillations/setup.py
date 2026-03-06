from setuptools import setup, find_packages

setup(
    name="BSM-with-TMS",
    version="0.1.0",
    description="Beyond Standard Model searches with TMS sensitivity",
    author="Jorge",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "notebook>=6.4.0",
    ],
    python_requires=">=3.8",
)