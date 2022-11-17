#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import os
import cdsapi
import shutil

""" Module for implementing the era5 input connector plugin. This plugin will make a api call via the
    cdsapi. The api structure has been prebuilt but requires several inputs, including a user id
    and api key pair. The rest of the input completes the api build. The downloaded file is unzipped 
    and the nc files are the outputs that remain.
"""

class Era5Input(GeoEDFPlugin):

    __optional_params = ['dataset','format','region','origin','variable','time_aggregation','horizontal_aggregation','start_year','end_year','version']
    __required_params = ['uid','api_key']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFInput super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for era5 not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()

    # each Input plugin needs to implement this method
    # if error, raise exception; if not, return True
    def get(self):
        #Checking the optional paramaters and setting default values if unset
        if self.dataset is None: self.dataset = 'insitu-gridded-observations-global-and-regional'
        if self.format is None: self.format = 'zip'
        if self.region is None: self.region = 'conus'
        if self.origin is None: self.origin = 'cpc_conus'
        if self.variable is None: self.variable = 'precipitation'
        if self.time_aggregation is None: self.time_aggregation = 'daily'
        if self.horizontal_aggregation is None: self.horizontal_aggregation = '0_25_x_0_25'
        if self.start_year is None: self.start_year =  '1950'
        if self.end_year is None: self.end_year = '2021'
        if self.version is None: self.version = 'v1.0' 
        # Storing the contents of the config file 
        config_file = "url: https://cds.climate.copernicus.eu/api/v2\nkey:" +self.uid + ":" + self.api_key +"\nverify: 0\n"
        print(config_file)
        # Writing the config file in the target_path location
        f = open(self.target_path+"/.cdsapirc", "w")
        f.write(config_file)
        f.close()
        # Changing the cdsapi configuration file location
        os.environ['CDSAPI_RC'] = self.target_path +"/.cdsapirc"
        # generate list of years
        start_year_num = int(self.start_year)
        end_year_num = int(self.end_year)
        if start_year_num > end_year_num:
            raise GeoEDFError('Start year needs to be earlier than end year in ERA5Input')
        years = []
        for year in range(start_year_num,end_year_num+1):
            years.append(f'{year}')
        # Making the api call
        try:
            download_path = self.target_path + "/download.zip" 
            c = cdsapi.Client()
            c.verify = False
            c.retrieve(    
                self.dataset,   
                 {        
                    'format': self.format,        
                    'region': self.region,        
                    'origin': self.origin,       
                    'variable': self.variable,        
                    'time_aggregation': self.time_aggregation,        
                    'horizontal_aggregation': self.horizontal_aggregation,        
                    'year': years,       
                    'version': self.version,    
                },   
                 download_path)
        except:
            raise GeoEDFError("The UID/API Key Pair was incorrect or the CDS servers are down")
        
        # Unpacking the downloaded file
        try:
                shutil.unpack_archive(download_path, self.target_path)
        except:
            raise GeoEDFError("Could not unzip downloaded files")
        
        # Deleting the unnecessary files to clean the output directory.
        try:
            os.remove(self.target_path+"/.cdsapirc")
            os.remove(download_path)
        except:
            raise GeoEDFError("Could not clean the output directory")
        
        
        return True
            
        
            