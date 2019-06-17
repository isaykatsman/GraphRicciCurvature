from setuptools import setup

setup(
        name='GraphRicciCurvature',
        version='0.1',
        description='Compute Discrete Ricci curvature and Ricci flow on NetworkX '
        'graph',
        author='Chien-Chun Ni',
        setup_requires=[
                'setuptools>=18.0',
                'cython',
                'pytest-runner',
        ],
        install_requires=[
                'cvxpy',
                'networkx',
                'numpy',
        ],
        extras_requires={
                'faster_apsp': ['networkit'],
        },
        packages=['GraphRicciCurvature'],
        license='LICENSE.txt',
)
