from setuptools import setup, find_packages

setup(name='cuahsisubsetterinput',
      version='0.1',
      description='Connector for running the CUAHSI subsetter for a given HUC12 ID',
      url='http://github.com/geoedf/cuahsisubsetterinput',
      author='Rajesh Kalyanam',
      author_email='rkalyanapurdue@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['requests'],
      zip_safe=False)
