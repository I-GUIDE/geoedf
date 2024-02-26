from setuptools import setup, find_packages

setup(name='deminput',
      version='0.1',
      description='Connector for fetching DEM rasters for a given siteID and resolution',
      url='http://github.com/geoedf/deminput',
      author='Rajesh Kalyanam',
      author_email='rkalyanapurdue@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['rasterio'],
      zip_safe=False)
