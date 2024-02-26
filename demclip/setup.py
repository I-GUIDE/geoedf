from setuptools import setup, find_packages

setup(name='demclip',
      version='0.1',
      description='Processor for DEM Clip given the shapefile path and raster path',
      url='http://github.com/geoedf/clip',
      author='Noah S. Oller Smith',
      author_email='nollersm@purdue.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['rasterio', 'shapely', 'geopandas'],
      zip_safe=False)
