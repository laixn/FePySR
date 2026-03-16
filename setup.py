from setuptools import setup, find_packages

setup(
    name="fepysr",                          
    version="0.1.0",                        
    author="Zhiming Yu",                    
    description="A Two-Stage Symbolic Regression Framework via Feature Engineering",
    packages=find_packages(),               
    install_requires=[                      
        "torch",
        "pysr",
        "hydra-core",
        "numpy",
        "omegaconf"
    ],
    python_requires=">=3.9",
)