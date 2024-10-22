from setuptools import setup
setup(name='IdealFlowNetwork',
version='1.15.1',
description='Ideal Flow Network',
url='https://people.revoledu.com/kardi/',
author='Kardi Teknomo',
author_email='kardi.teknomo@petra.ac.id',
license='GNU Public License 3.0',
packages=['IdealFlow'],
install_requires=['scipy>=1.7','networkx>=3.1','numpy>=1.24.4','matplotlib>=3.7.5'],  # package dependencies
zip_safe=False)