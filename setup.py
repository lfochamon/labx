from setuptools import setup

setup(
    name="labx",
    version="0.1",
    description="Material for labX",
    author="Luiz Chamon",
    url="https://github.com/lfochamon/labx",
    packages="labx",
    include_package_data=True,
    install_requires=["torch"],
)
