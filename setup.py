import setuptools
setuptools.setup(     
     name="PruneKit",     
     version="0.0.1",
     author="saipy",
     python_requires=">=3.11",   
     packages=setuptools.find_packages(),
     install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)