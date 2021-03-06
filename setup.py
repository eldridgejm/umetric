from setuptools import setup

setup(
    name="umetric",
    version="1.0",
    description="Utilities for working with ultrametrics.",

    author="Justin Eldridge",
    author_email="eldridge@cse.ohio-state.edu",

    packages=["umetric", "umetric.fit", "umetric.data"],
    include_package_data=True,

    install_requires=["networkx", "scipy", "numpy"],
)

