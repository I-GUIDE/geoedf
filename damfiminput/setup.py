from setuptools import setup, find_packages

setup(name='damfiminput',
      version='0.2',
      description='Connector for accessing Dam Flood Inundation Map datasets',
      url='http://github.com/geoedf/connectors',
      author='Rajesh Kalyanam',
      author_email='rkalyanapurdue@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['requests'],
      zip_safe=False)
