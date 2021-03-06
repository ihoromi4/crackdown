from setuptools import setup, find_packages

setup(
    name='Crackdown',
    version='0.1.0',
    description='Reinforcement Learning Utilities for Gaming',
    author='Ihor Omelchenko',
    author_email='counter3d@gmail.com',
    license='MIT',
    packages=find_packages(exclude=["docs", "tests", "examples"]),
    install_requires=[
        'numpy',
        'torch>=1.4',
        'gym',
        'tqdm>=4.41',
        'albumentations>=0.4.3',
    ]
)
