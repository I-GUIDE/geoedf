#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geoedfframework.utils.GeoEDFError import GeoEDFError
from geoedfframework.GeoEDFPlugin import GeoEDFPlugin

from osgeo import ogr,osr
import os
import subprocess
from subprocess import PIPE
from joblib import Parallel, delayed
import pandas as pd
import xarray as xr
import numpy as np

""" Module for implementing the SubsetAORCForcingData processor. The processor takes a 
    start and end date as well as a HUC12 ID or shapefile or geospatial extents as input. 
    AORC data is clipped to the provided extent and the necessary forcing data input variables 
    are extracted. A path to pre-downloaded AORC data files is required.
"""

class SubsetAORCForcingData(GeoEDFPlugin):
    # input directory or shapefile params are XOR
    # shapefile will take precedence
    # if end is provided, period also needs to be provided
    __optional_params = ['huc12_id','shapefile','extents']
    __required_params = ['start_date','end_date','version','aorc_datapath','gsl']

    # path to HUC2 regions shapefile that is installed as part of this filter package
    __huc12_shapefile = '/compute_shared/AORC_Forcing/HUC12/huc12.shp'
    
    # path to NC file for deriving indices from coordinates
    __ldas_ncfile = '/compute_shared/AORC_Forcing/HUC12/201601010000.LDASOUT_DOMAIN1.comp'
    
    # extents for GSL-buffered in matrix coordinates
    __gsl_bounds = (373, 1227, 1586, 2405)
    
    # Anvil-specific AORC_Forcing datapath
    __anvil_aorc_datapath = '/anvil/datasets/ncar/AORC_Forcing'
    
    # we use just kwargs since we need to be able to process the list of attributes
    # and their values to create the dependency graph in the GeoEDFPlugin super class
    def __init__(self, **kwargs):

        # list to hold all the parameter names; will be accessed in super to 
        # construct dependency graph
        self.provided_params = self.__required_params + self.__optional_params

        # check that all required params have been provided
        for param in self.__required_params:
            if param not in kwargs:
                raise GeoEDFError('Required parameter %s for SubsetAORCForcingData not provided' % param)

        # specific check for conditionally required params
        # either inputdir or shapefile need to be provided
        # shapefile takes precedence
        if 'huc12_id' not in kwargs and 'extents' not in kwargs:
            raise GeoEDFError('Either the HUC12 ID or extents for SubsetAORCForcingData need to be provided.')
            
        if 'shapefile' in kwargs:
            raise GeoEDFError('Shapefile input is not currently supported in SubsetAORCForcingData')

        # set all required parameters
        for key in self.__required_params:
            setattr(self,key,kwargs.get(key))

        # set optional parameters
        for key in self.__optional_params:
            # if key not provided in optional arguments, defaults value to None
            setattr(self,key,kwargs.get(key,None))

        # class super class init
        super().__init__()
        
    # function to subset a LDASIN file
    def subset_forcingdata(self,filePath):
        if os.path.exists(filePath):
            try:
                filename = os.path.split(filePath)[1]
                subPath = '%s/%s' % (self.target_path,filename)
                if self.version == '1.0':
                    subprocess.run(["ncks", filePath, "-d", "west_east,"+str(self.nwm_indices[0])+","+str(self.nwm_indices[1]), "-d", "south_north,"+str(self.nwm_indices[2])+","+str(self.nwm_indices[3]), "-O", subPath],stdout=PIPE,stderr=PIPE)
                else:
                    subprocess.run(["ncks", filePath, "-d", "x,"+str(self.nwm_indices[0])+","+str(self.nwm_indices[1]), "-d", "y,"+str(self.nwm_indices[2])+","+str(self.nwm_indices[3]), "-O", subPath],stdout=PIPE,stderr=PIPE)
            except:
                raise GeoEDFError('Error subsetting forcing file %s in SubsetAORCForcingData' % filePath)


    # get indices for forcing data file given LCC extents
    def get_indices_from_extents(self):
        
        try:
            # get corner coordinates
            lo_y = self.input_extents[2]
            up_y = self.input_extents[3]
            lo_x = self.input_extents[0]
            up_x = self.input_extents[1]

            nc_NWM = xr.open_dataset(self.__ldas_ncfile)
            X_NWM = nc_NWM.coords['x']
            Y_NWM = nc_NWM.coords['y']

            # Calculate the distance between the center of grids from the location of the site
            distance = ((Y_NWM - lo_y)**2 + (X_NWM - lo_x)**2)**0.5
            yindex1, xindex1 = np.where(distance == distance.min())

            # Calculate the distance between the center of grids from the location of the site
            distance = ((Y_NWM - up_y)**2 + (X_NWM - up_x)**2)**0.5
            yindex2, xindex2 = np.where(distance == distance.min())
            
            indices = (xindex1[0],xindex2[0],yindex1[0],yindex2[0])
        
            return indices
        except:
            raise GeoEDFError('Error transforming HUC12 extent to forcing data indices in SubsetAORCForcingData')

    # get extents for geometry in LCC projection
    def get_geom_lcc_extents(self,geom):
        try:
            # reproject geom to LCC
            source = osr.SpatialReference()
            source.ImportFromEPSG(4269)

            target = osr.SpatialReference()
            target.ImportFromProj4('+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m no_defs')
        
            # projection transformer
            transform = osr.CoordinateTransformation(source, target)
        
            # reproject geometry to LCC
            geom.Transform(transform)
        
            return geom.GetEnvelope()
        except:
            raise GeoEDFError('Error getting HUC12 watershed extents in LCC projection in SubsetAORCForcingData')

    # reproject point in WGS84 to LCC
    def reproject_point_to_lcc(self,point):
        try:
            # create projection transformer
            source = osr.SpatialReference()
            source.ImportFromEPSG(4326)

            target = osr.SpatialReference()
            target.ImportFromProj4('+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m no_defs')
        
            # projection transformer
            transform = osr.CoordinateTransformation(source, target)
        
            # reproject point to LCC
            point.Transform(transform)
        
            return point
        except:
            raise GeoEDFError('Error reprojecting point to LCC projection in SubsetAORCForcingData')
        
        
    # each Processor plugin needs to implement this method
    # if error, raise exception
    # assume this method is called only when all params have been fully instantiated
    def process(self):
        
        try:
            geom = None
            env = None
            filePaths = []
            # compute input extents based on HUC12 ID or input shapefile
            if self.huc12_id is not None:
                # process HUC12 ID param
                driver = ogr.GetDriverByName('ESRI Shapefile')
                inDataset = driver.Open(self.__huc12_shapefile, 0)
                if inDataset is None:
                    raise GeoEDFError('Error opening HUC12 shapefile in SubsetAORCForcingData')
                inLayer = inDataset.GetLayer()
                # filter by HUC12 ID
                inLayer.SetAttributeFilter("huc12 = '%s'" % self.huc12_id)
                for feature in inLayer:
                    geom = feature.GetGeometryRef()
                if geom is None:
                    raise GeoEDFError('Error filtering HUC12 shapefile to retrieve this watershed in SubsetAORCForcingData')
                # get geom extents
                self.input_extents = self.get_geom_lcc_extents(geom)
                print('HUC12 extents ',self.input_extents)

                # transform extents to indices
                self.nwm_indices = self.get_indices_from_extents()
                
            elif self.extents is not None:
                # short circuit to nwm_indices
                local_extents = self.extents.split(',')
                self.nwm_indices = (int(local_extents[0]),int(local_extents[1]),int(local_extents[2]),int(local_extents[3]))
                
            elif self.shapefile is not None:
                driver = ogr.GetDriverByName('ESRI Shapefile')
                inDataset = driver.Open(self.shapefile, 0)
                if inDataset is None:
                    raise GeoEDFError('Error opening input shapefile in SubsetAORCForcingData')
                inLayer = inDataset.GetLayer()
                # assuming shapefile input is in WGS84 projection
                shp_extents = inLayer.GetExtent()
                print('Shapefile extents in WGS84: ',shp_extents)
                # create point geometries of lower left and upper right to reproject
                ll_point = ogr.Geometry(ogr.wkbPoint)
                ll_point.AddPoint(shp_extents[0],shp_extents[2])
                print('Lower left point ',ll_point.GetX(),ll_point.GetY())
                ll_reproj_point = self.reproject_point_to_lcc(ll_point)
                
                ur_point = ogr.Geometry(ogr.wkbPoint)
                ur_point.AddPoint(shp_extents[1],shp_extents[3])
                print('Upper right point ',ur_point.GetX(),ur_point.GetY())
                ur_reproj_point = self.reproject_point_to_lcc(ur_point)
                
                # reconstruct extents in LCC
                self.input_extents = (ll_point.GetX(),ur_point.GetX(),ll_point.GetY(),ur_point.GetY())
                print('Shapefile reprojected in LCC extents ',self.input_extents)
                
                # transform extents to indices
                self.nwm_indices = self.get_indices_from_extents()
            
            # if GSL option is set, we need to get relative extents from the larger GSL bounds
            if self.gsl == 'True':
                self.nwm_indices = (self.nwm_indices[0] - self.__gsl_bounds[0] + 1,
                                    self.nwm_indices[1] - self.__gsl_bounds[0] + 1,
                                    self.nwm_indices[2] - self.__gsl_bounds[2] + 1,
                                    self.nwm_indices[3] - self.__gsl_bounds[2] + 1)
            print('Indices ',self.nwm_indices)
            
            # envelope has been retrieved
            # now retrieve list of files based on date range
            start_dt = pd.to_datetime(self.start_date,format='%m/%d/%Y')
            end_dt = pd.to_datetime(self.end_date,format='%m/%d/%Y')
            if end_dt < start_dt:
                raise GeoEDFError('Dates provided in incorrect order for SubsetAORCForcingData')
                
            dates = pd.date_range(start=start_dt, end=end_dt, freq='1H')
            
            # update AORC data path based on version
            if self.version == '1.1':
                self.aorc_datapath = self.__anvil_aorc_datapath
            
            for date in dates:
                if self.version == '1.1':
                    fileName = f'{date.year}/{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}{str(date.hour).zfill(2)}00.LDASIN_DOMAIN1'
                else:
                    if self.gsl == 'True':
                        fileName = f'GSL/{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}{str(date.hour).zfill(2)}.LDASIN_DOMAIN1'
                    else:
                        fileName = f'{date.year}/{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}{str(date.hour).zfill(2)}.LDASIN_DOMAIN1'
                filePath = '%s/%s' % (self.aorc_datapath,fileName)
                #self.subset_forcingdata(filePath)
                filePaths.append(filePath)

            # subset the files
            Parallel(n_jobs=min(len(filePaths),50))(delayed(self.subset_forcingdata)(path) for path in filePaths)
            
        except:
            raise GeoEDFError('Error occurred when running SubsetAORCForcingData processor')
