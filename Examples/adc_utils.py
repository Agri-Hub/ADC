
import datacube
import math
import calendar
import ipywidgets
import numpy as np
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




def calculate_index(data, index):
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
