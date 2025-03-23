import setuptools
# with open("README.md", "r", encoding="utf-8") as fh:
    # long_description = fh.read()
setuptools.setup(
    name="genetic_algorithm", # Ovviamente, metti il tuo nome ;)
    version="0.0.1",
    package_dir={"": "genetic_algorithm"},
    author="Your Name",
    author_email="your.email@example.com",
    description="Genetic Algorithm implementation",
    url="https://github.com/FlavioFoxes/Genetic_Algortithm_kmc.git",
    packages=setuptools.find_packages(
        where="genetic_algorithm",
        include=[
            "algorithm",        
        ],
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)