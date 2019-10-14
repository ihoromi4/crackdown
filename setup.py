from distutils.core import setup

setup(
    name='Crackdown',
    version='0.1.0',
    description='Reinforcement Learning Utilities for Gaming',
    author='Ihor Omelchenko',
    author_email='counter3d@gmail.com',
    license='MIT',
    packages=['crackdown'],
    install_requires=[
        'numpy',
        'torch',
        'gym',
    ]
)
