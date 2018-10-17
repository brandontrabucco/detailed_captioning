"""Setup script for detailed_captioning."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['object_detection', 'tensorflow-gpu']

setup(
    name='detailed_captioning',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('detailed_captioning')],
    description='Berkeley-CMU Detailed Captioning Workspace',
)