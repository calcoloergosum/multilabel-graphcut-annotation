"""Setup file"""
from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

REQUIREMENTS = [str(req.requirement) for req in parse_requirements(
    'requirements.txt', session=None)]

if __name__ == "__main__":
    setup(
        name="multilabel-graphcut-annotation",
        use_scm_version=True,
        package_data={"multilabel_graphcut_annotation": ["py.typed"]},
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        setup_requires=['setuptools_scm'],
        url="https://github.com/studentofkyoto/multilabel-graphcut-annotation/",
        python_requires=">= 3.8",
    )
