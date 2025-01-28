from setuptools import setup, find_packages

setup(
    name='dau-active-inference',
    version='1.0.3',
    author='LearnableLoopAI.com / UltimIT.us',
    author_email='kobus.esterhuysen.ai@gmail.com',
    description='Multi-agent active inference with PyMDP',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LearnableLoopAI/Designer-Artifact-User',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
    ],
    python_requires='>=3.12',
    install_requires=[
        'pymdp',
        'numpy',
        'matplotlib',
        'inferactively-pymdp',
        # add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'dau-active-inference=dau_active_inference.dau_active_inference:main',
        ],
    },
)

