import os

import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
      name="DarwiNN",
      version="0.1.0-alpha",
      description="Distributed Deep Neuroevolution",
      long_description=read('README.md'),
      author="Lucian Petrica",
      author_email="lucianp@xilinx.com",
      url="https://github.com/Xilinx/DarwiNN",
      python_requires=">=3.6",
      install_requires=["torch>=1.1.0","deap"],
      packages=setuptools.find_packages()
)
