#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

import os
import requests
import zipfile
import time
import shutil
import json

""" Module for implementing the CUAHSISubsetterInput connector. This accepts a HUC12 ID
    as input and submits a request to the CUAHSI subsetter to fetch domain data for this HUC12
    watershed.
"""

class CUAHSISubsetterInput(GeoEDFPlugin):
    __optional_params = []
    __required_params = ['huc12_id']

    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for CUAHSISubsetterInput not provided' % param)

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()
        
    # get the extents of the HUC12 watershed given a HUC12 ID
    def get_huc12_extent(self):
        try:
            gethucbbox_url = "https://subset.cuahsi.org/wbd/gethucbbox/lcc?hucID={}".format(self.huc12_id)
            res = requests.get(gethucbbox_url)
            res.raise_for_status()
            gethubbox_res_json = res.json()
            return gethubbox_res_json["bbox"]        
        except:
            raise GeoEDFError('Error occurred in retrieving bounds for given HUC12 ID: %s' % self.huc12_id)

    # each Connector plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def get(self):
        
        uid = None
        
        try:
            # first get the bounds for the huc12 ID
            west, south, east, north = self.get_huc12_extent()
            
            # next run the subsetter for these extents
            submit_url = f'https://subset.cuahsi.org/nwm/v2_0/subset?llat={south}&llon={west}&ulat={north}&ulon={east}&hucs=[]'
            res = requests.get(submit_url)
            res.raise_for_status()

            # grab the job identifier
            uid = res.url.split('jobid=')[-1]

            # query job status
            status_url = f'https://subset.cuahsi.org/jobs/{uid}'

            attempt = 0
            max_attempts = 100

            status = None
            
            while attempt < max_attempts:
                print("subsetting domain files" + ".." * (attempt+1))
                res = requests.get(status_url)
                res.raise_for_status()
                status = json.loads(res.text)['status']
                if status == 'finished':
                    print("downloading...")
                    break
                attempt += 1
                time.sleep(10)
                
            if status != 'finished':
                raise GeoEDFError('Could not determine if CUAHSI subsetter completed execution, assume job failed')
                
        except:
            raise GeoEDFError('Error occurred running the CUAHSI subsetter for the HUC12 watershed: ',self.huc12_id)
    
        try:
            # download the result
            dl_url = f'https://subset.cuahsi.org/download-zip/{uid}'
            local_filename = f'{uid}.zip'
            local_filepath = '%s/%s' % (self.target_path,local_filename)
            with requests.get(dl_url, stream=True) as r:
                with open(local_filepath, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
        except: 
            raise GeoEDFError('Error occurred downloading CUAHSI subsetter result')
            
        try:
            # unzip the domain data in the target folder and delete the zip
            if os.path.exists(local_filepath):
                with zipfile.ZipFile(local_filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.target_path)
                os.remove(local_filepath)
        except:
            raise GeoEDFError('Error occurred when unzipping domain data in CUAHSISubsetterInput connector')
