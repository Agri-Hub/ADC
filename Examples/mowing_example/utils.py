import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
import datacube
from datacube.utils import geometry
from datacube.utils.geometry import Geometry,CRS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors as mcolours
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap,BASE_COLORS,CSS4_COLORS
import itertools
from datetime import datetime,timedelta
from pyproj import Proj, transform
from osgeo import ogr
import csv
import xarray as xr
import fiona
import rasterio.features
from fiona.crs import from_epsg
from shapely.geometry import shape
import os,sys
import warnings
warnings.filterwarnings("ignore")



def fs_creation(shapefile_bbox,start_date,end_date,optical_bands,buffer,ids_to_keep=[]):
    
    config="/home/eouser/datacube.conf"
    dc = datacube.Datacube(app="test",config=config)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shapefile_bbox,0)
    layer = ds.GetLayer()
    xmin,xmax,ymin,ymax = layer.GetExtent()
    bbox = [xmin,xmax,ymin,ymax]

    data_cube = getData_optical(bbox,config,start_date,end_date,optical_bands)
    data_cube = data_cube.load()
    ids = getIDs(bbox,config)
    grouped_data = data_cube.groupby(ids[buffer][0])
    dates = data_cube.time.values
    dates = sorted(set(np.array([str(t).split('T')[0] for t in dates])))  

    fs_mean = []
    fs_std = []
    ids = []
    for f in grouped_data:
        key, parcel_data = f[0],f[1]
        if len(ids_to_keep)>0:
            if key in list(ids_to_keep):
                mean_parcel = np.array([np.nanmean(parcel_data[b].values,axis=1) for b in optical_bands])
                std_parcel = np.array([np.nanstd(parcel_data[b].values,axis=1) for b in optical_bands])
                fs_mean.append(mean_parcel)
                fs_std.append(std_parcel)
                ids.append(key)
        else:
            if key!=-1:
                mean_parcel = np.array([np.nanmean(parcel_data[b].values,axis=1) for b in optical_bands])
                std_parcel = np.array([np.nanstd(parcel_data[b].values,axis=1) for b in optical_bands])
                fs_mean.append(mean_parcel)
                fs_std.append(std_parcel)
                ids.append(key)
            
    fs_mean = np.array(fs_mean)
    fs_std = np.array(fs_std)
    ids = np.array(ids)
    dates = np.array(dates)
    
    return ids,fs_mean,fs_std,dates


def calculate_index(data, index):
    
    """
    Optical Indices Computation

    :param xarray: datacube_object
    :param string: you want to compute

    """
    
    
    B02 = data.B02.astype('float16')
    B03 = data.B03.astype('float16')
    B04 = data.B04.astype('float16')
    B05 = data.B05.astype('float16')
    B06 = data.B06.astype('float16')
    B07 = data.B07.astype('float16')
    B08 = data.B08.astype('float16')
    B8A = data.B8A.astype('float16')
    B11 = data.B11.astype('float16')
    B12 = data.B12.astype('float16')
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
    

def cloud_data(data, index,fill_val=np.nan):
    
    """
    Cloud Masking Computation
    
    :param xarray: datacube_object
    :param index: you want to compute the cloud mask
    :param float: masking value (default:np.nan)
    
    """
    return xr.where((data.SCL>=4) & (data.SCL<=6), data[index.lower()], fill_val)


def getData_optical(bbox,config,timeStart,timeEnd,optical_bands,resolution = 10):
    
    product_optical= 's2_preprocessed_lithuania'
    
    all_bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']
    all_indices = ['ndvi','ndwi','ndmi','psri','savi','evi','dvi','rdvi','rvi','tvi','tcari','gi','vigreen',
                   'varigreen','gari','gdvi','sipi','wdrvi','gvmi','gcvi']
    
    bands = [b for b in optical_bands if b in all_bands]
    indices = [b for b in optical_bands if b in all_indices]
    

    xmin,xmax,ymin,ymax = bbox[0],bbox[1],bbox[2],bbox[3]
    query = {
        'time': (timeStart,timeEnd),
        'product': product_optical,
        'x':(xmin,xmax),
        'y':(ymin,ymax),
        'crs':'EPSG:3857'
    }
    dc = datacube.Datacube(app="test", config=config)
    data = dc.load(**query,measurements=all_bands,dask_chunks={})
    for i,index in enumerate(optical_bands):
        if index in indices:
            data[index] = calculate_index(data,index)
        data[index] = cloud_data(data,index)
        if i == 0:
            to_keep = data[index].dropna(dim='time',thresh=0.25).time
            data = data.sel(time=to_keep)
            tt =  np.array([np.datetime64(datetime.strptime(str(k),'%Y-%m-%dT%H:%M:%S.%f000').strftime('%Y-%m-%d'))for k in data.time.values])
            data = data.assign_coords(time=('time',tt))
            tt = np.unique(tt,return_index=True)[1]
            data = data.isel(time=tt)

    for b in all_bands:
        if b not in bands:
            data = data.drop(b)
    return data




def getIDs(bbox,config):

    xmin,xmax,ymin,ymax = bbox[0],bbox[1],bbox[2],bbox[3]
    
    query = {
        'product':'ids_lithuania_2020',
        'x':(xmin,xmax),
        'y':(ymin,ymax),
        'crs':'EPSG:3857'
    }
    dc = datacube.Datacube(app="test", config=config)
    data = dc.load(**query)
    return data



all_optical_bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']




def geometry_mask(geoms, geobox, touches=False, invertion=False):
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=touches,
                                           invert=invertion)


def getData_2(config,product,geom,geom_buffer,startDate,endDate,bands=all_optical_bands):
    
    """
    return an xarray of the data you want

    :param string: product_name
    :param geom: rasterized geometry
    :param geom: buffered_rasterized geometry
    :param string: initial_date
    :param string: final_date
    :param list[bands]: list of bands to be returned from DC
    """

    
    query = {
        'geopolygon': geom_buffer,
        'time': (startDate,endDate),
        'product': product
        }
    dc = datacube.Datacube(app="test", config=config)
    data = dc.load(output_crs="EPSG:3857",measurements=bands,resolution=(-10,10),**query,dask_chunks={})
    if len(data) == 0:
        return -1
    
    #ndvi calculation and masking of clouds only this index
    data['ndvi'] = calculate_index(data,'ndvi')
    data['ndvi'] = cloud_data(data,'ndvi')
    mask = geometry_mask([geom], data.geobox, invertion=True)
    
    #masked data of ndvi based on input geometry
    masked_data = data.where(mask)
    
    #keep only "clearsky" time instances (real values of higher than a threshold)
    to_keep = masked_data['ndvi'].dropna(dim='time',thresh=0.2).time
    
    data = data.sel(time=to_keep)
    masked_data = masked_data.sel(time=to_keep)
    tt =  np.array([np.datetime64(datetime.strptime(str(k),'%Y-%m-%dT%H:%M:%S.%f000').strftime('%Y-%m-%d'))for k in data.time.values])
    data = data.assign_coords(time=('time',tt))
    masked_data = masked_data.assign_coords(time=('time',tt))
    #drop duplicates
    tt = np.unique(tt,return_index=True)[1]
    data = data.isel(time=tt)
    masked_data = masked_data.isel(time=tt)

    data = data.load()
    masked_data = masked_data.load()
    return data,masked_data



def photo_interpretation(shape_file,id_to_check,start_date,end_date,zoom_out=100):

    ds = fiona.open(shape_file)
    crs = geometry.CRS(ds.crs_wkt)
    product= 's2_preprocessed_lithuania'
    config="/home/eouser/datacube.conf"
    buffer = 50
    gdf = gpd.GeoDataFrame.from_file(filename=shape_file,driver='ESRI Shapefile')
    case = int(gdf[gdf.Parcel_ID==id_to_check].index[0])
    f = ds[case]
    gdf_f = gdf.iloc[case:case+1].copy()

    feature_geom = f['geometry']
    geom = Geometry(feature_geom,crs)

    geom_buffer = geom.buffer(zoom_out)  
    bounds = shape(feature_geom).bounds
    s2,s2_mask = getData_2(config,product,geom,geom_buffer,start_date,end_date)
    dates_str = np.array([str(d).split('T')[0] for d in s2_mask.time.values])

    pseudo_col = ["B04", "B03", "B02"]
    col_n = 4

    # check if an additional row is needed for the plot
    if len(dates_str)%col_n==0:
        row_n = len(dates_str)//col_n
    else:
        row_n = len(dates_str)//col_n + 1


    fig, ax = plt.subplots(row_n,col_n, figsize=(12,30))

    n = 0 

    stop = False

    for i in range(row_n):
        for j in range(col_n):

            gdf_f.geometry.to_crs("EPSG:3857").plot(ax=ax[i][j],facecolor='none',edgecolor='red',linewidth=3)
            s2[pseudo_col].isel(time=n).to_array().plot.imshow(ax=ax[i][j], robust=True, add_labels=False)

            ax[i][j].set_title(dates_str[n],fontsize=15)
            ax[i][j].axis('off')

            n += 1
            if n==len(dates_str):
                stop = True
                break
        if stop:
            break

    fig.delaxes(ax[-1][-1])

    plt.tight_layout()
    plt.axis('off')
    plt.show()





def restore_picture(series,limit_down=-0.005,limit_up=-0.02):
    
    try:
        df_c = pd.DataFrame(index=series.dropna().index)
        df_c['VI'] = series.dropna().values
        df_c['VI_diff'] = series.dropna().diff().fillna(0).values
        df_c['VI_diff_2'] = series.dropna().diff().fillna(0).values - series.dropna().diff(2).fillna(0).values
        df_c['VI_diff_inv'] = series.dropna().diff(-1).fillna(0).values
        df_c['days'] = np.append(np.diff(df_c.index.dayofyear),0)
        df_c['days_inv'] = np.append(0,abs(np.diff(df_c.index[::-1].dayofyear)[::-1]))
        df_c['ratio'] = df_c['VI_diff']/df_c['days']
        df_c['ratio_inv'] = df_c['VI_diff_inv']/df_c['days_inv']

        ids = df_c.index[np.where((df_c.ratio.values<limit_down)&(df_c.ratio_inv.values<limit_up)&(df_c.VI_diff_2.values<=0.0))]
        series[ids] = np.nan
    except:
        pass
    
    return series



class Mowing_Detection:
    
    def __init__(self):
        
        self.columns = ['mow_n','m1_dstart','m1_dend','m1_conf',
                        'm2_dstart','m2_dend','m2_conf',
                        'm3_dstart','m3_dend','m3_conf',
                        'm4_dstart','m4_dend','m4_conf',
                        'm5_dstart','m5_dend','m5_conf']
        self.InitialParameters = (65.,265.,0.2,0.9,0.05,0.05)

               
        
    def predict(self,*args,**kwargs):
        

        return self.model(*args,**kwargs)
            
        
    def model(self,*args,**kwargs):
            

            df = args[0]
            try:
                th = kwargs['threshold']
            except:
                th = 0.3
            try:
                min_rate = kwargs['min_rate']
            except:
                min_rate = 0.1
            columns = self.columns
            final_df = pd.DataFrame(data=np.zeros((df.shape[0],len(columns))),
                                    index=df.index,columns=columns)
            for p in range(df.shape[0]):
                df_temp = df.iloc[p,:].dropna()
                days = df_temp.index
                vi = df_temp.values
                mowing_events = []
                confidence = []
                flag = False
                for t in range(1,len(vi)):
                      
                    if flag:
                        flag = False
                        continue
                        
                    d = days[t]-days[t-1]
                    d = d.days
                    if np.isnan(vi[t]):
                        continue
                    if (vi[t]<vi[t-1]-th) and (vi[t-1]-vi[t] <= d*min_rate):
                        mowing_events.append((days[t-1],days[t]))
                        confidence.append(vi[t-1]-vi[t]-th)
                        flag = True
                        
         
                if mowing_events:
#                     args = np.argsort(confidence)[::-1][:3]
                    args = np.array(sorted(np.argsort(confidence)[::-1][:5]))
                    mowing_events = [mowing_events[n] for n in args]
                    final_df.iloc[p,0] = len(args)
                    final_df.iloc[p,1:3*len(args):3] = [m[0] for m in mowing_events]
                    final_df.iloc[p,2:1+3*len(args):3] = [m[1] for m in mowing_events]
#                     final_df.iloc[p,3:2+3*len(args):3] = np.sort(confidence)[::-1][:len(args)]
                    confidence = np.array(confidence)
                    final_df.iloc[p,3:2+3*len(args):3] = confidence[args]

            return final_df
