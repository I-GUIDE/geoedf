#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import os
import shutil

import rasterio
from rasterio.merge import merge

""" Module for implementing the DEMMerge processor. This processor accepts a path containing
    a set of tif files. An optional resulting filename can also be provided.
"""

class DEMMerge(GeoEDFPlugin):
    __optional_params = ['merged_filename']
    __required_params = ['input_path']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for DEMMerge not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))
            
        if self.merged_filename is None:
            self.merged_filename = 'merged_raster.tif'
        else:
            self.merged_filename = self.merged_filename + ".tif"

        # class super class init
        super().__init__()

    # each Connector plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def process(self):
        
        ## Get a list of all DEM files in the input folder
        dem_files = [f for f in os.listdir(self.input_path) if f.endswith(".tif")]
            
        if len(dem_files) < 1:
            print("could not find any DEM files to merge")
            return
        else:
            if len(dem_files) == 1: #only 1 file, copy to destination
                print("only one file found, returning as merged result")
                src_filepath = os.path.join(self.input_path, dem_files[0])
                dst_filepath = os.path.join(self.target_path,self.merged_filename)
                shutil.copyfile(src_filepath,dst_filepath)
                return
            else: # more than one file

                ## Create a list to store the raster datasets
                datasets = []

                ## Open each DEM file and append it to the datasets list
                try:
                    for dem_file in dem_files:
                        file_path = os.path.join(self.input_path, dem_file)
                        src = rasterio.open(file_path)
                        datasets.append(src)
                except:
                    raise GeoEDFError("Error occurred when opening DEM rasters in DEMMerge")

                ## Merge the raster datasets into a single mosaic
                try:
                    mosaic, out_trans = merge(datasets)
                except:
                    raise GeoEDFError("Error occurred when merging DEM rasters in DEMMerge")

                ## Copy the metadata from one of the datasets (assuming they all have the same metadata)
                out_meta = datasets[0].meta.copy()
                out_meta.update({
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_trans
                })

                # Write the mosaic to the output file
                try:
                    output_filename = os.path.join(self.target_path,self.merged_filename)
                    with rasterio.open(output_filename, 'w', **out_meta) as dest:
                        dest.write(mosaic)
                except:
                    ## Close all the opened datasets
                    for dataset in datasets:
                        dataset.close()
                    raise GeoEDFError("Error occurred when writing out merged raster in DEMMerge")

                print(f"Merging completed for DEM raster files")
                
            return True
            
