from setuptools import setup, find_packages

setup(
    name='LaTorch',
    version='0.1',
    packages=find_packages(),
    description='A translator from LaTeX tensor expressions to PyTorch models.',
    author='Leonardo Barazza',
    author_email='leonardo.barazza@gmail.com',
    url='https://github.com/lbarazza/LaTorch',  # Optional
    install_requires=[
        # List your library's dependencies here
        # e.g., 'numpy>=1.10.0',
        'torch',
    ],
    classifiers=[
        # See https://pypi.org/classifiers/ for all the values you can use
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
