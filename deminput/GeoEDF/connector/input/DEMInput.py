#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import math
import numpy as np
import os

from pynhd import NLDI
import urllib.request

""" Module for implementing the DEMInput connector. This connector accepts a siteID
    and resolution string as input. For e.g. 1/3 arc second (code = 13).
    The connector will determine the lat-lon extents of the watershed region and download 
    one DEM raster per lat-lon degree. The connector will also write out a watershed shapefile
    for use in subsequent workflow tasks.
"""

class DEMInput(GeoEDFPlugin):
    __optional_params = []
    __required_params = ['site_id','resolution']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for DEMInput not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()

    # each Connector plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def get(self):
        
        try:
            ## Get the watershed using USGS station number using pynhd module
            watershed=NLDI().get_basins(self.site_id)

            ## Saving the watershed file as a shapefile in the output folder
            shapefile_fileloc_filename=f'{self.target_path}/shape_{self.site_id}.shp'
            watershed.to_file(filename=shapefile_fileloc_filename,
                              driver='ESRI Shapefile',
                              mode='w')            
            ## Get the min and max of latitude and longitude (or easting and northing)
            extents_basin=watershed.total_bounds
            
            extent_left=abs(math.floor(extents_basin[0]))
            extent_right=abs(math.floor(extents_basin[2]))
            extent_bottom=abs(math.ceil(extents_basin[1]))
            extent_top=abs(math.ceil(extents_basin[3]))
            
            num_tiles_download=(((extent_left+1)-extent_right)*((extent_top+1)-extent_bottom))   
            print("number of tiles to download: ",num_tiles_download)
            
            # download DEM rasters
            current_filenum=1
            for lon in (range(extent_right,extent_left+1,1)):
                for lat in (range(extent_bottom,extent_top+1,1)):
                    usgs_filename=f'n{lat:02d}w{lon:03d}'
        
                    print(f'Beginning file {current_filenum} download with urllib2  out of {num_tiles_download}...')
                    url = (f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/{self.resolution}/TIFF'
                           f'/current/{usgs_filename}/USGS_{self.resolution}_{usgs_filename}.tif'
                          )
                
                    ## The r in 'fr' disables backslach escape sequence processing
                    local_fileloc_filename=fr'{self.target_path}/USGS_{self.resolution}_{usgs_filename}.tif'
        
                    urllib.request.urlretrieve(url,local_fileloc_filename) #without progressbar for multiple USGS sites
        
                    print(f'Completed file {current_filenum} download with urllib2 {url} out of {num_tiles_download}...')
                    print(f'*************************************************************************************\n')
        
                    current_filenum+=1            
        except:
            raise GeoEDFError('Error occurred when running DEMInput connector')
            
        return True
