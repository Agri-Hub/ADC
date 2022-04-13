
import datacube
import math
import calendar
import ipywidgets
import numpy as np
import pandas as pd
import geopandas as gpd
from datacube.utils import geometry
from datacube.utils.geometry import Geometry,CRS
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors as mcolours
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from pyproj import Proj, transform
from osgeo import ogr
import os,sys
import rasterio.features as rf
dc = datacube.Datacube(app="analytics",config="/home/eouser/datacube.conf")
import time,fiona
from tqdm import tqdm
from shapely.geometry import shape
import xarray as xr

def date_range(start, end, intv):
    start = datetime.strptime(start,"%Y-%m-%d")
    end = datetime.strptime(end,"%Y-%m-%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y-%m-%d")
    yield end.strftime("%Y-%m-%d")


def getData_optical(bbox, timeStart, timeEnd, optical_bands, resolution=10):
    product_optical = 's2_preprocessed_{}'.format(pilot)

    all_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'SCL']
    all_indices = ['ndvi', 'ndwi', 'ndmi', 'psri', 'savi', 'evi', 'dvi', 'rdvi', 'rvi', 'tvi', 'tcari', 'gi', 'vigreen',
                   'varigreen', 'gari', 'gdvi', 'sipi', 'wdrvi', 'gvmi', 'gcvi']

    bands = [b for b in optical_bands if b in all_bands]
    indices = [b for b in optical_bands if b in all_indices]

    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    query = {
        'time': (timeStart, timeEnd),
        'product': product_optical,
        'x': (xmin, xmax),
        'y': (ymin, ymax),
        'crs': 'EPSG:3857'
    }
    dc = datacube.Datacube(app="test", config=config)
    data = dc.load(**query, measurements=all_bands, dask_chunks={})
    for i, index in enumerate(optical_bands):
        if index in indices:
            data[index] = calculate_index(data, index)
        data[index] = cloud_data(data, index)
        if i == 0:
            #             to_keep = data[index].dropna(dim='time',how='all').time
            to_keep = data[index].dropna(dim='time', thresh=0.25).time
            data = data.sel(time=to_keep)

    for b in all_bands:
        if b not in bands:
            data = data.drop(b)
    return data

def getIDs(product_ids,xmin,xmax,ymin,ymax):
    query = {
        'product':product_ids,
        'x':(xmin,xmax),
        'y':(ymin,ymax),
        'crs':'EPSG:3857'
    }
    dc = datacube.Datacube(app="test", config=config)
    data = dc.load(**query)
    return data

def calculate_index(data, index):
    if 'B02' in data:
        B02 = data.B02.astype('float16')
    if 'B02' in data:
        B03 = data.B03.astype('float16')
    if 'B04' in data:
        B04 = data.B04.astype('float16')
    if 'B05' in data:
        B05 = data.B05.astype('float16')
    if 'B06' in data:
        B06 = data.B06.astype('float16')
    if 'B07' in data:
        B07 = data.B07.astype('float16')
    if 'B08' in data:
        B08 = data.B08.astype('float16')
    if 'B8A' in data:
        B8A = data.B8A.astype('float16')
    if 'B11' in data:
        B11 = data.B11.astype('float16')
    if 'B12' in data:
        B12 = data.B12.astype('float16')
    try:
        if index.lower() == 'ndvi':
            return (B08 - B04) / (B08 + B04)
        if index.lower() == 'ndwi':
            return (B03 - B08) / (B08 + B03)
        if index.lower() == 'ndmi':
            return (B08 - B11) / (B08 + B11)
        if index.lower() == 'psri':
            return (B04 - B02) / B06
        if index.lower() == 'savi':
            L = 0.428;
            return ((B08 - B04) / (B08 + B04 + L)) * (1.0 + L)
        if index.lower() == 'evi':
            return 2.5 * (B08 - B04) / ((B08 + 6 * B04 - 7.5 * B02) + 1.0)
        if index.lower() == 'dvi':
            return (B08 - B04)
        if index.lower() == 'rdvi':
            return (B08 - B04) / (B08 + B04) ** 0.5
        if index.lower() == 'rvi':
            return (B08 / B04)
        if index.lower() == 'tvi':
            return (120 * (B08 - B03) - 200 * (B04 - B03)) / 2
        if index.lower() == 'tcari':
            return ((B08 - B04) - 0.2 * (B08 - B03) * (B08 / B04)) * 3
        if index.lower() == 'gi':
            return (B03 / B04)
        if index.lower() == 'vigreen':
            return (B03 - B04) / (B03 + B04)
        if index.lower() == 'varigreen':
            return (B03 - B04) / (B03 + B04 - B02)
        if index.lower() == 'gari':
            return (B08 - (B03 - (B02 - B04))) / (B08 - (B03 + (B02 - B04)))
        if index.lower() == 'gdvi':
            return (B08 - B03)
        if index.lower() == 'sipi':
            return (B08 - B02) / (B08 - B04)
        if index.lower() == 'wdrvi':
            alpha = 0.2
            return (alpha * B08 - B04) / (alpha * B08 + B04)
        if index.lower() == 'gvmi':
            return ((B08 + 0.1) - (B12 + 0.02)) / ((B08 + 0.1) + (B12 + 0.02))
        if index.lower() == 'gcvi':
            return (B08 / B03) - 1
        else:
            return None
    except Exception as e:
        return None

def cloud_data(data, index):
    return xr.where((data.SCL>=4) & (data.SCL<=6), data[index], np.nan)



def calculate_index_old(data, index):
    '''
        Calculates statistics on a grouped xarray
        :param data: an xarray 
        :param index: specific index to calculate
    '''
    if index.lower() == 'ndvi':
        B08 = data.B08.astype('float16')
        B04 = data.B04.astype('float16')
        return (B08 - B04) / (B08 + B04)
    if index.lower() == 'ndwi':
        return (B03 - B08) / (B08 + B03)
    if index.lower() == 'ndmi':
        return (B08 - B11) / (B08 + B11)
    if index.lower() == 'psri':
        return (B04 - B02) / B06
    if index.lower() == 'savi':
        L = 0.428;
        B08 = data.B08.astype('float16')
        B04 = data.B04.astype('float16')
        return ((B08 - B04) / (B08 + B04 + L)) * (1.0 + L)
    if index.lower() == 'evi':
        return 2.5 * (B08 - B04) / ((B08 + 6 * B04 - 7.5 * B02) + 1.0)
    if index.lower() == 'dvi':
        return (B08 - B04)
    if index.lower() == 'rdvi':
        return (B08 - B04) / (B08 + B04) ** 0.5
    if index.lower() == 'rvi':
        return (B08 / B04)
    if index.lower() == 'tvi':
        return (120 * (B08 - B03) - 200 * (B04 - B03)) / 2
    if index.lower() == 'tcari':
        return ((B08 - B04) - 0.2 * (B08 - B03) * (B08 / B04)) * 3
    if index.lower() == 'gi':
        return (B03 / B04)
    if index.lower() == 'vigreen':
        return (B03 - B04) / (B03 + B04)
    if index.lower() == 'varigreen':
        return (B03 - B04) / (B03 + B04 - B02)
    if index.lower() == 'gari':
        return (B08 - (B03 - (B02 - B04))) / (B08 - (B03 + (B02 - B04)))
    if index.lower() == 'gdvi':
        return (B08 - B03)
    if index.lower() == 'sipi':
        return (B08 - B02) / (B08 - B04)
    if index.lower() == 'wdrvi':
        alpha = 0.2
        return (alpha * B08 - B04) / (alpha * B08 + B04)
    if index.lower() == 'gvmi':
        return ((B08 + 0.1) - (B12 + 0.02)) / ((B08 + 0.1) + (B12 + 0.02))
    if index.lower() == 'gcvi':
        return (B08 / B03) - 1
    else:
        return None

def geometry_mask(geoms,geobox,all_touched=False,invert=False):
    return rf.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],out_shape=geobox.shape,transform=geobox.affine,all_touched=all_touched,invert=invert)


def get_data_for_validation(parcel,start_date='2019-01-01',end_date='2019-12-31',index='ndvi',method='median',period='month'):
    parcel_geom = parcel[0]
    import shapely, geojson
    g1 = shapely.wkt.loads(parcel_geom)
    g2 = geojson.Feature(geometry=g1, properties={})
    product = 's2_preprocessed_cyprus'
    bands = ['B08', 'B12', 'SCL']
    g3 = {'type': 'Polygon', 'coordinates': g2['geometry']['coordinates']}
    import warnings
    warnings.filterwarnings('ignore')
    from fiona.crs import from_epsg
    crs = from_epsg(3857)
    bands = ['B08', 'B04', 'SCL']
    geom = Geometry(geom=g3, crs=crs)
    query = {
        'geopolygon': geom,
        'product': product,
        'time': (start_date,end_date)

    }
    data = dc.load(output_crs="EPSG:3857", resolution=(-10, 10), measurements=bands, **query, dask_chunks={})
    mask = geometry_mask([geom], data.geobox, invert=True)
    data = data.where(mask)
    data['ndvi'] = calculate_index(data, 'ndvi')
    data['ndvi'] = data['ndvi'].where(((data['SCL'] >= 4) & (data['SCL'] <= 6)), np.nan)
    data = data.drop('B04')
    data = data.drop('B08')
    data = data.drop('SCL')
    data = data.load()
    times = data.time.values
    return data,times
