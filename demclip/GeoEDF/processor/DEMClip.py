#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import os

import rasterio
import geopandas
from rasterio.mask import mask
from shapely.geometry import mapping

""" Module for implementing the DEMClip processor. This processor accepts a path for
    a shaplefile and a path for the raster.
"""

class DEMReproject(GeoEDFPlugin):
    __optional_params = []
    __required_params = ['shapefile_path', 'raster_path']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for DEMClip not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()

    # each Processor plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def process(self):

        # open the shapefile with geopandas
        try:
            shapefile_gdf = geopandas.read_file(self.shapefile)
        except:
            raise GeoEDFError("Error occurred when opening the shapefile")
        
        # open raster file
        try:
            src = rasterio.open(self.raster_path)
        except:
            raise GeoEDFError("Error occurred when opening raster file")

        # convert the shapefile geometry to the same CRS as the raster
        try:
            shapefile_gdf = shapefile_gdf.to_crs(src.crs)
        except:
            raise GeoEDFError("Error occurred when converting shapefile to same CRS")

        # convert the shapefile geometry to GeoJSON-like format
        try:
            geoms = [mapping(geom) for geom in shapefile_gdf.geometry]
        except:
            raise GeoEDFError("Error occurred when converting shapefile to GeoJSON-like format")

        # clip raster with shapefile geometry
        try:
            clipped, out_transform = mask(src, geoms, crop=True)
        except:
            raise GeoEDFError("Error occurred when clipping raster with shapefile geometry")

        # update metadata for the new dataset
        out_meta = src.meta.copy()
        out_meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': out_transform
        })
        src.close()

        # write clipped raster
        try:
            with rasterio.open(self.target_path, 'w', **out_meta) as dst:
                dst.write(clipped)
            dst.close()
        except:
            raise GeoEDFError("Error occurred when writing clipped raster")
