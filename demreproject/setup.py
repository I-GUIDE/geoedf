from setuptools import setup, find_packages

setup(name='demreproject',
      version='0.1',
      description='Processor for DEM Reproject given a site ID, raster path, and a shapefile path',
      url='http://github.com/geoedf/demreproject',
      author='Noah S. Oller Smith',
      author_email='nollersm@purdue.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['geopandas', 'pynhd', 'rasterio', 'pyproj'],
      zip_safe=False)
