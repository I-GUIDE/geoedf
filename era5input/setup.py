from setuptools import setup, find_packages

setup(name='era5',
      version='0.1',
      description='ERA 5',
      url='http://github.com/geoedf/',
      author='Gaurav',
      author_email='sachdevg@purdue.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['cdsapi'],
      zip_safe=False)
