#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import os

import rasterio
from pynhd import NLDI
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
import geopandas

""" Module for implementing the DEMReproject processor. This processor accepts a path for
    a shaplefile, a path for the raster, and a site id.
"""

class DEMReproject(GeoEDFPlugin):
    __optional_params = []
    __required_params = ['site_id', 'shapefile_path', 'raster_path']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for DEMProject not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # Correct paths
        self.shapefile_path = self.shapefile_path + ".shp"

        # class super class init
        super().__init__()

    # each Processor plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def process(self):

        # get the watershed using USGS station number using pynhd module
        try:
            watershed = NLDI().get_basins(self.site_id)
        except:
            raise GeoEDFError("Error occurred when obtaining watershed")
        
        # estimate crs using geopandas
        try:
            target_crs = watershed.estimate_utm_crs(datum_name='WGS 84')
        except:
            raise GeoEDFError("Error occurred when estimating crs")

        # reproject to shapefile
        try:
            watershed_file=geopandas.read_file(self.shapefile_path)
            watershed_proj=watershed_file.to_crs(target_crs)
            watershed_proj.to_file(self.target_path, driver='ESRI Shapefile', mode='w')
        except:
            raise GeoEDFError("Error occurred when reprojecting to shapefile")

        # obtain target crs projection
        try:
            # load the DEM file
            src = rasterio.open(self.raster_path)
            # define the target CRS
            target_crs_proj = pyproj.CRS.from_string(target_crs)
        except:
            raise GeoEDFError("Error occurred when obtaining crs projection")

        # calculate the transformation and new dimensions
        transform, width, height = calculate_default_transform(
            src.crs, target_crs_proj, src.width, src.height, *src.bounds)

        # update metadata for the new dataset
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs_proj,
            'transform': transform,
            'width': width,
            'height': height
        })

        # reproject to target_path
        try:
            # create the output dataset and perform the reprojection
            with rasterio.open(self.target_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs_proj,
                        resampling=Resampling.nearest
                    )
                src.close()
                dst.close()
        except:
            raise GeoEDFError("Error occurred when reprojecting to target path")
