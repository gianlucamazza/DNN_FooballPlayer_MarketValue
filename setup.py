from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="football-market-value-prediction",
    version="0.1.0",
    author="Gianluca Mazza",
    author_email="info@gianlucamazza.it",
    description="A project for predicting football player market values using deep learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gianlucamazza/LSTM_FooballPlayer_MarketValue",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "fmvp-cli=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "football_player_valuation": ["data/raw/*.csv"],
    },
)
