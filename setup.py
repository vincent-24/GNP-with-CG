from setuptools import setup, find_packages

setup(
    name='GNP',
    version='1.0.0',
    author='Jie Chen',
    author_email='chenjie@us.ibm.com',
    description='Graph neural preconditioner',
    packages=find_packages(include=['GNP', 'GNP.*', 'scripts', 'scripts.*']),
    install_requires=[
        'mat73',
        'tqdm',
        'numpy',
        'scipy',
        'torch',
        'ssgetpy',
        'matplotlib',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'gnp-run=scripts.run_exp:main',
        ],
    },
)
