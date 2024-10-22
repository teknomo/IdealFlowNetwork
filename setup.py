from setuptools import setup, find_packages
# from pathlib import Path

# # Read the contents of your README file
# this_directory = Path(__file__).parent
# long_description = (this_directory / 'README.md').read_text()

setup(name='IdealFlowNetwork',
    version='1.15.3',
    description='A Python package for Ideal Flow Network computations',
    # long_description=long_description,
    # long_description_content_type='text/markdown', 
    url='https://github.com/teknomo/IdealFlowNetwork',  
    author='Kardi Teknomo',
    author_email='kardi.teknomo@petra.ac.id',
    license='GNU General Public License v3.0',
    packages=find_packages(),  # Automatically find all packages in your source directory
    install_requires=[
        'scipy>=1.7',
        'networkx>=3.1',
        'numpy>=1.24.4',
        'matplotlib>=3.7.5'
    ],
    python_requires='>=3.10',  # Specify the minimum Python version
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    keywords=["ideal flow network", "graph theory","network analysis"],
    project_urls={
        'Documentation': 'https://people.revoledu.com/kardi/research/trajectory/ifn/doc/html',
        'Source': 'https://github.com/teknomo/IdealFlowNetwork',
        'Tracker': 'https://github.com/teknomo/IdealFlowNetwork/issues',
    },
    zip_safe=False
)