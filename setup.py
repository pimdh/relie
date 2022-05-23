from setuptools import setup, find_packages

setup(
    name='relie',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch3d',
        'numpy',
        'Pillow',
        'tensorboardX',
        'scikit-learn',
        'matplotlib'
    ]
)
