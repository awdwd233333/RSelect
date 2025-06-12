# Imports
from __future__ import division
import os, sys, time, warnings, calendar, datetime, pickle, math
from shutil import copyfile, copytree, rmtree

import matplotlib, ee, shapefile, cv2, geemap, imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import morphology as mm
from skimage.util import img_as_ubyte
from skimage.io import imread
from scipy import ndimage
from skimage.measure import regionprops

from osgeo import gdal, gdal_array, osr
try:from osgeo.scripts import gdal_pansharpen
except ModuleNotFoundError:from osgeo_utils import gdal_pansharpen
try:import ogr
except ModuleNotFoundError:from osgeo import ogr

from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiLineString
try: from skimage.filters import threshold_otsu, rank
except ImportError: from skimage.filter import threshold_otsu, rank

plt.rcParams['axes.unicode_minus']=False#显示负号
plt.rcParams["font.size"]=10#全局字号,针对英文
plt.rcParams["mathtext.fontset"]='stix'#设置$$之间是字体与中文不一样,为新罗马,这样就可以中英混用
plt.rcParams['font.sans-serif'] = ['Times New Roman']#单独使用英文时为新罗马字体
# del matplotlib.font_manager.weight_dict['roman']#取消字体加粗
# matplotlib.font_manager._rebuild()
# Suppress Warnings
warnings.filterwarnings("ignore")

__author__ = 'Federico Monegaglia'
__email__ = 'f.monegaglia@gmail.com'
__version__ = '1.0'
__year__ = '2016'


# Import Everything from SubModules

# ============================
#   General Main Functions
# ============================

def lt5_addNDVI(image):
	B3 = image.select(['B3'])
	B4 = image.select(['B4'])
	B3 = B3.updateMask(B3.gte(0.1)).updateMask(B4.gte(0.1))
	B4 = B4.updateMask(B3.gte(0.1)).updateMask(B4.gte(0.1))
	image = image.addBands(srcImg=B3,overwrite=True)
	image = image.addBands(srcImg=B4, overwrite=True)
	return image.addBands(image.normalizedDifference(['B4','B3']).rename(['NDVI']))

def lc8_addNDVI(image):
	B5 = image.select(['B5'])
	B4 = image.select(['B4'])
	B5 = B5.updateMask(B5.gte(0.1)).updateMask(B4.gte(0.1))
	B4 = B4.updateMask(B5.gte(0.1)).updateMask(B4.gte(0.1))
	image = image.addBands(srcImg=B5, overwrite=True)
	image = image.addBands(srcImg=B4, overwrite=True)
	return image.addBands(image.normalizedDifference(['B5','B4']).rename(['NDVI']))
def s2_addNDVI(image):return image.addBands(image.normalizedDifference(['B8','B4']).rename(['NDVI']))
def lt5_addMNDWI(image):return image.addBands(image.normalizedDifference(['B2','B5']).rename(['MNDWI']))
def lc8_addMNDWI(image):return image.addBands(image.normalizedDifference(['B3','B6']).rename(['MNDWI']))
def s2_addMNDWI(image):return image.addBands(image.normalizedDifference(['B3','B11']).rename(['MNDWI']))

def lt5_addNDWI(image):return image.addBands(image.normalizedDifference(['B2','B4']).rename(['NDWI']))
def lc8_addNDWI(image):return image.addBands(image.normalizedDifference(['B3','B5']).rename(['NDWI']))
def s2_addNDWI(image):return image.addBands(image.normalizedDifference(['B3','B8']).rename(['NDWI']))

def clip(col,aoi):
    col_list=[]
    for i in range(col.size().getInfo()):
        image = getImg(col,i)
        col_list.append(image.clip(aoi))
    return ee.ImageCollection.fromImages(col_list)

def getImg(col,index):return ee.Image(col.toList(100000).get(index))
def get_proj_trans(img):
    # pretty code
    bandList = ee.List(ee.Dictionary(ee.Algorithms.Describe(img)).get('bands'))
    b1 = ee.Dictionary(bandList.get(0))
    dimensions = b1.get("dimensions")
    dimensions = ee.List(dimensions)
    sizex = ee.Number(dimensions.get(0)).getInfo()
    sizey = ee.Number(dimensions.get(1)).getInfo()

    crs_transform = b1.get("crs_transform").getInfo()

    origin = b1.get("origin").getInfo()

    EPSG = int(b1.get("crs").getInfo().split(':')[-1])

    # Landsat系列在南半球北半球的北偏移都为0, EPSG=326XX
    # EPSG=327XX的y坐标+1e7则为EPSG=326XX的y坐标
    # if img.get('system:id').getInfo().split('/')[0] == 'LANDSAT' and crs_transform[-1]-origin[1]*crs_transform[0]<0:NF=10**7
    # else:NF=0
    NF = 0
    GeoTransf = {
        'PixelSize': abs(crs_transform[0]),
        'X': crs_transform[2] + origin[0] * crs_transform[0],
        'Y': crs_transform[-1] - origin[1] * crs_transform[0] + NF,
        'Lx': sizex,
        'Ly': sizey
    }

    return GeoTransf, EPSG

def pntsTrans( lines, EPSG, EPSG_out ):
    from pyproj import CRS, Transformer
    # lines为经纬度或者xy, 输出为经纬度或者xy
    # 把 xy 地理坐标转化为投影坐标，gge默认地理坐标是WGS84
    transformer = Transformer.from_crs( CRS.from_epsg(EPSG), CRS.from_epsg(EPSG_out) )
    n_lines=[]
    for line in lines:
        if EPSG==4326:y, x = list(zip(*line))
        else:x, y = list(zip(*line))
        [x, y] = transformer.transform( x, y )
        if EPSG_out==4326:x,y=y,x
        n_lines.append(list(zip(x,y)))
    return n_lines


def cut_by_max_rectarea( pnts, maxarea, geotrans ):
    def get_pnts( poly ):
        # 提取多边形所有的点，不分开
        # poly是shapely的multipoly或者polygon
        if poly.type == 'MultiPolygon':
            xy = []
            for i in poly.geoms:
                xy += list(zip(*i.exterior.coords.xy))
        else:
            xy = list(zip(*poly.exterior.coords.xy))
        return list(zip(*xy))

    from shapely.geometry import Polygon
    # pnts=[[x],[y]]
    # 1.首先划分aoi
    pcsize = geotrans['PixelSize']
    geo_x = geotrans['X']
    geo_y = geotrans['Y']
    poly1 = Polygon(zip(pnts[0],pnts[1]))
    [x,y] = get_pnts(poly1)
    area = (max(x)-min(x))*(max(y)-min(y))
    if area >= maxarea:
        # 需要切分
        sub_aois, rel_coor = [], []
        cut = (max(x)-min(x))/(area//maxarea)
        cut_x1, cut_x2 = min(x), min(x)+cut# 初始切分区域
        while cut_x1 < max(x):
            poly2 = Polygon([[cut_x1,max(y)],[cut_x2,max(y)],[cut_x2,min(y)],[cut_x1,min(y)]])
            poly_insec = poly1.intersection(poly2)
            # 计算满足条件的最小外接矩形
            [sub_x, sub_y] = get_pnts(poly_insec)
            sub_area = (max(sub_x)-min(sub_x))*(max(sub_y)-min(sub_y))
            if sub_area > maxarea:
                small_cut = cut/sub_area*maxarea*0.9
                while sub_area > maxarea:
                    cut_x2 = cut_x1+small_cut
                    poly2 = Polygon([[cut_x1,max(y)],[cut_x2,max(y)],[cut_x2,min(y)],[cut_x1,min(y)]])
                    poly_insec = poly1.intersection(poly2)
                    [sub_x,sub_y] = get_pnts(poly_insec)
                    sub_area = (max(sub_x)-min(sub_x))*(max(sub_y)-min(sub_y))
                    small_cut = small_cut*0.9
                else:
                    cut = small_cut/0.9#定下该次的cut
                    sub_aois.append(get_geometries(poly_insec))
                    #获取相对像元坐标
                    rel_x,rel_y = (min(sub_x)-geo_x)/pcsize,(geo_y-max(sub_y))/pcsize
                    rel_x,rel_y = int(rel_x),int(rel_y)
                    rel_coor.append([rel_x,rel_y])
            else:
                large_cut = cut
                while sub_area <= maxarea and cut_x2 < max(x):
                    large_cut=large_cut*1.1
                    cut_x2=cut_x1+large_cut
                    poly2=Polygon([[cut_x1,max(y)],[cut_x2,max(y)],[cut_x2,min(y)],[cut_x1,min(y)]])
                    poly_insec=poly1.intersection(poly2)
                    [sub_x,sub_y]=get_pnts(poly_insec)
                    sub_area=(max(sub_x)-min(sub_x))*(max(sub_y)-min(sub_y))
                else:
                    large_cut=large_cut/1.1
                    cut_x2=cut_x1+large_cut
                    poly2=Polygon([[cut_x1,max(y)],[cut_x2,max(y)],[cut_x2,min(y)],[cut_x1,min(y)]])
                    poly_insec=poly1.intersection(poly2)
                    [sub_x,sub_y]=get_pnts(poly_insec)
                    cut=large_cut#定下该次的cut
                    sub_aois.append(get_geometries(poly_insec))
                    #获取相对像元坐标
                    rel_x,rel_y=(min(sub_x)-geo_x)/pcsize,(geo_y-max(sub_y))/pcsize
                    rel_x,rel_y=int(rel_x),int(rel_y)
                    rel_coor.append([rel_x,rel_y])
            cut_x1,cut_x2=cut_x2,cut_x2+cut#换到下一个切分区域
    else:
        print('Dont need to cut',end=' ')
        rel_coor=[[0,0]]
        sub_aois=[[list(zip(pnts[0],pnts[1]))]]
    return sub_aois,rel_coor

def get_geometries( poly ):
    # 按每个poly来提取点
    # poly是shapely的multipoly或者polygon
    xy=[]
    if poly.type=='MultiPolygon':
        for i in poly.geoms:
            xy.append(list(zip(*i.exterior.coords.xy)))
    else:
        xy.append(list(zip(*poly.exterior.coords.xy)))

    return xy

def cut_aoi(img,aoi_pnts):
    GeoTransf,EPSG = get_proj_trans(img)
    pcsize=GeoTransf['PixelSize']
    x=GeoTransf['X']
    y=GeoTransf['Y']
    lx=GeoTransf['Lx']
    ly=GeoTransf['Ly']
    # print( GeoTransf, 'EPSG:', EPSG )
    [lon,lat]= list(zip(*aoi_pnts[0]))
    if EPSG==4326:
        xy=[lon,lat]
    else:
        # 转aoi_pnts为投影坐标
        aoi_proj = pntsTrans(aoi_pnts,4326,EPSG)[0]
        # aoi_proj=list(zip(xy[0],xy[1]))

        # 首先要求aoi和img_rect的交集
        xmin,xmax,ymin,ymax=x,x+lx*pcsize,y-ly*pcsize,y
        img_rect=Polygon([(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin),(xmin,ymax)])
        aoi_poly=Polygon(aoi_proj)

        intersect = img_rect.intersection(aoi_poly)
        aoi_pnts = pntsTrans( get_geometries(intersect), EPSG, 4326 )
        aoi_pnts = pntsTrans(aoi_pnts,4326,EPSG)
        xy = list(zip(*aoi_pnts[0]))

    maxarea=262144*pcsize**2*0.9
    sub_aois,rel_coor=cut_by_max_rectarea(xy,maxarea,GeoTransf)
    if EPSG==4326:
        sub_aois_WGS=sub_aois
    else:
        #转换aoi为WGS地理坐标
        sub_aois_WGS=[]
        for idx,sub_aoi in enumerate(sub_aois):
            sub_aois_WGS.append([])
            for poly in sub_aoi:
                sub_aois_WGS[idx].append(pntsTrans([poly],EPSG,4326)[0])
    print('aoi_cut_into:%s parts'%len(sub_aois_WGS),end=' ')

    return sub_aois_WGS,rel_coor

def arr2tif_all( arrdir, geodir, prjdir, tifdir, ROI=None, overwrite=False ):
	print( 'Convert:%s To %s'%( arrdir, tifdir ) )
	if not os.path.isdir(tifdir):os.mkdir(tifdir)
	arrfiles = [os.path.join( arrdir, i ) for i in os.listdir( arrdir ) if os.path.splitext( i )[1] == '.npy']
	for arrfile in arrfiles:
		# print( 'Converting %s...'%( arrfile ) )
		name = os.path.splitext( os.path.basename( arrfile ) )[0]
		geofile = os.path.join( geodir, name+'.p' )
		prjfile = os.path.join( prjdir, name+'.prj' )

		arr = np.load( arrfile )
		proj = load(prjfile)
		GeoTransf = load(geofile)
		if proj is None or GeoTransf is None:continue

		tiffile = os.path.join( tifdir, name+'.tif' )
		if os.path.isfile(tiffile) and not overwrite:continue
		if ROI is not None:
			array2raster( arr, r'temp.tif', proj=proj, GeoTransf=GeoTransf)
			arr, GeoTransf, proj = Tif_clip( r'temp.tif', ROI, mode='n', cropToCutline=False )
			# GeoTransf fmt: [x,width,0,y,0,-height]
			# pcsize, originX, originY = GeoTransf['PixelSize'], GeoTransf['X'], GeoTransf['Y']
			GeoTransf = {
			'PixelSize' : abs( GeoTransf[1] ),
			'X' : GeoTransf[0],
			'Y' : GeoTransf[3],
			'Lx' : arr.shape[1],
			'Ly' : arr.shape[0]
			}
			proj = osr.SpatialReference( proj )
		array2raster( arr, tiffile, proj=proj, GeoTransf=GeoTransf)

def load( fname, *args, **kwargs ):
    from xml.dom.minidom import parse
    '''Load file depending on the extension'''
    if fname is None:return None
    ext = os.path.splitext( fname )[-1]

    if not os.path.isfile(fname):return None

    if ext == '.txt':
        return np.loadtxt( fname, *args, **kwargs )
    elif ext == '.npy':
        arr = np.load( fname, *args, **kwargs )
        if len(arr.shape) == 3 and arr.shape[-1]==1:
            arr = arr[:,:,0]
        return arr
    elif ext == '.p':
        return pickle.load( open(fname,'rb') )
    elif ext == '.prj':
        proj = osr.SpatialReference()
        fn = open(fname,'r')
        Wkt=fn.read()
        fn.close()
        proj.ImportFromWkt(Wkt)
        return proj
    elif ext in ['.kml','.xml']:
        # 目前只能读面的信息
        doc = parse( fname )
        root = doc.documentElement
        coordinates = root.getElementsByTagName("coordinates")
        obj_list=[]
        for coordinate in coordinates:
            old_data = coordinate.childNodes[0].data
            new_data = " ".join([old.replace(",", " ") for old in old_data.split(",0")])
            new_data = new_data.strip('\n').strip('\t').strip(' ')
            #t = [pnt for pnt in new_data.split('  ') if (pnt!=[''] and pnt!=['\n'])]
            pnts=[[float(i.strip(' ').strip('\t').strip('\n')) for i in pnt.split(' ')] for pnt in new_data.split('  ') if pnt!='\n']
            obj_list.append( pnts )
        return obj_list
    elif ext in ['.shp']:
        return [shape.points for shape in shapefile.Reader( fname ).shapes()]
    elif ext == '.ini':
        cf = ConfigParser()
        cf.read( fname )
        return cf
    elif ext in ['.tif','.tiff']:
        arr = gdal_array.LoadFile( fname )
        geo = gdal.Open( fname )
        Geotrans, proj = read_geo(geo)
        return arr, Geotrans, proj
    else:
        e = 'Format %s not supported for file %s. Use either "npy" or "txt"' % ( ext, fname )
        raise TypeError(e)

def array2raster( in_array, raster_path, band_num=None, proj=None, GeoTransf=None, NoDataValue=None ):
    # in_array是2/3维数组,层数在第三维,proj是 osr.SpatialReference()
    # band_num是类似[0,2,3],最小是0

    # 判断栅格数据的数据类型
    if 'int8' in in_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in in_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName("GTiff")
    #outdata = driver.Create(raster_path,ysize=in_array.shape[0],xsize=in_array.shape[1],bands=len(band_num),datatype=datatype)
    if len(in_array.shape) == 2:in_array = in_array[:,:,np.newaxis]
    if band_num is None:band_num = range( 0, in_array.shape[2] )
    outdata = driver.Create( raster_path, in_array.shape[1], in_array.shape[0], len(band_num), datatype )
    for i,idx in enumerate( band_num ):
        outband = outdata.GetRasterBand( i + 1 )
        in_array_band = in_array[:,:,idx]
        outband.WriteArray( in_array_band )
        if NoDataValue is not None:outband.SetNoDataValue(NoDataValue)

    if GeoTransf != None:
        pcsize = GeoTransf['PixelSize']
        originX, originY = GeoTransf['X'], GeoTransf['Y']
        outdata.SetGeoTransform( (originX, pcsize, 0, originY, 0, -pcsize) )
        # geotrans的标准格式 [x,width,0,y,0,-height]
    if proj != None:
        outdata.SetProjection( proj.ExportToWkt() )
    outdata.FlushCache()
    outdata = None

def Tif_clip( RSI, shp, n_RSI=None, n_shp=None, mode='RSI', cropToCutline=True ):
    # RSI即单个影像的文件夹
    if mode=='RSI':
        bands,GeoTransf,proj = LoadLandsatData(RSI)#Read bands Data
        bnames,pan,band_id = ReadSatInfo(RSI)# Read bands Info
    else:
        geo = gdal.Open( RSI )
        name = os.path.basename(RSI)
        GeoTransf, proj = read_geo(geo)
        bands = gdal_array.LoadFile(RSI)
    if str(type(proj))!='''<class 'osgeo.osr.SpatialReference'>''':proj=osr.SpatialReference(proj)

    pcsize,y,x,lx,ly=GeoTransf['PixelSize'],GeoTransf['X'],GeoTransf['Y'],GeoTransf['Ly'],GeoTransf['Lx']
    y_range,x_range=[y,ly*pcsize+y],[x-lx*pcsize,x]
    # print(GeoTransf,y_range,x_range)
    rect=[[(y_range[0],x_range[0]),(y_range[0],x_range[1]),(y_range[1],x_range[1]),(y_range[1],x_range[0]),(y_range[0],x_range[0])]]#y是东西方向


    clipshp=r'.\stl_range.shp';pnts2shppoly(rect,clipshp,proj)

    #求遥感范围并裁剪shp掩膜
    clipshp_proj,shp_clip=r'.\stl_range_proj.shp',r'.\shpclip.shp' if n_shp==None else n_shp
    driver = ogr.GetDriverByName('ESRI Shapefile')
    inputDataSource = driver.Open(shp, 0)
    inputLayer = inputDataSource.GetLayer(0)
    EPSG_NUM=inputLayer.GetSpatialRef().GetAttrValue('AUTHORITY', 1)
    projTrans(clipshp,clipshp_proj,int(EPSG_NUM))#转换clipshp的投影到shpmask的投影
    shpclip( shp, clipshp_proj, shp_clip )#用clipshp裁剪shpmask
    #判断shp与tif是否相交
    if shapefile.Reader( shp_clip ).shapes()==[]:
        print('ERROR: shp_mask and tif didnt cross!！ CHECK YOUR SHPFILE!')
        sys.exit(1)

    #裁剪栅格数据
    if mode=='RSI':
        bands=[]
        if pan!=None and n_RSI!=None:gdal.Warp( os.path.join( n_RSI, os.path.basename(pan) ), pan, cutlineDSName = shp_clip, cropToCutline=cropToCutline)
        for idx,bname in enumerate(bnames):
            if n_RSI!=None:bandClip=os.path.join(n_RSI,os.path.basename(bname))
            else:bandClip=r'.\bandClip.tif'
            gdal.Warp(bandClip, bname, cutlineDSName=shp_clip, cropToCutline=cropToCutline)
            bands.append(gdal_array.LoadFile(bandClip))
        geo = gdal.Open( bandClip );proj=geo.GetProjection();geotrans=geo.GetGeoTransform()
        data = np.array( bands, dtype=bands[0].dtype )
    else:
        if n_RSI!=None:bandClip=os.path.join(n_RSI,os.path.basename(name))
        else:bandClip=r'.\bandClip.tif'
        gdal.Warp(bandClip, RSI, cutlineDSName=shp_clip, cropToCutline=cropToCutline)
        geo = gdal.Open( bandClip );proj=geo.GetProjection();geotrans=geo.GetGeoTransform()
        bands = gdal_array.LoadFile(bandClip)
        data = np.array( bands, dtype=bands[0].dtype )
    # Remove temporary files
    geo=None
    if n_RSI==None:os.remove(bandClip)
    if n_shp==None:os.remove(shp_clip)

    os.remove(clipshp)
    os.remove(clipshp_proj)
    for t in ['.shx','.dbf','.prj']:
        if n_shp==None:os.remove(shp_clip.replace('.shp',t))
        os.remove(clipshp.replace('.shp',t))
        os.remove(clipshp_proj.replace('.shp',t))

    return data, geotrans, proj

def pnts2shppoly( polys, shppath, proj=None ):
    # polys = [[[x1,y1],[x2,y2],...],[poly2],[],...],x是东西方向
    # proj是osr.SpatialReference()
    name = os.path.splitext( os.path.basename(shppath) )[0]
    gdal.SetConfigOption( "GDAL_FILENAME_IS_UTF8", "NO" )  # 为了支持中文路径
    gdal.SetConfigOption( "SHAPE_ENCODING", "CP936" )  # 为了使属性表字段支持中文
    ogr.RegisterAll()  # 注册所有的驱动

    driver = ogr.GetDriverByName("ESRI Shapefile")
    outputDataSource = driver.CreateDataSource( shppath )
    if driver == None:print("%s 驱动不可用！\n")
    if outputDataSource == None:print("创建文件【%s】失败！")

    oLayer = outputDataSource.CreateLayer( name, proj, ogr.wkbPolygon, [] )

    oFieldID = ogr.FieldDefn( "FieldID", ogr.OFTInteger )  # 创建一个叫FieldID的整型属性
    oLayer.CreateField( oFieldID, 1 )

    oFieldName = ogr.FieldDefn( "FieldName", ogr.OFTString )  # 创建一个叫FieldName的字符型属性
    oFieldName.SetWidth(100)  # 定义字符长度为100
    oLayer.CreateField( oFieldName, 1 )

    oDefn = oLayer.GetLayerDefn()  # 定义要素

    '''
    yard = ogr.Geometry(ogr.wkbPolygon )  #  构建几何类型:多边形
    yard.CloseRings()'''

    for pnts in polys:
        poly = ogr.Geometry(ogr.wkbPolygon )
        ring = ogr.Geometry(ogr.wkbLinearRing)  #  构建几何类型:线
        for i,item in enumerate(pnts):ring.AddPoint(item[0],item[1])  #  添加点
        poly.AddGeometry(ring)
        # 创建单个面要素
        oFeatureTriangle = ogr.Feature(oDefn)
        oFeatureTriangle.SetField(0, 0)
        oFeatureTriangle.SetField(1, "单个面")
        oFeatureTriangle.SetGeometry(poly)
        oLayer.CreateFeature(oFeatureTriangle)
        oFeatureTriangle.Destroy()

    '''
    geomTriangle = ogr.CreateGeometryFromWkt(str(yard))  # 将封闭后的多边形集添加到属性表

    oFeatureTriangle.SetGeometry(geomTriangle)
    oLayer.CreateFeature(oFeatureTriangle)
    outputDataSource.Destroy()'''

def up_sample( mask, times=2 ):
	n_mask = np.zeros(np.array(mask.shape)*times)
	for i in range(times):
		for j in range(times):
			n_mask[i::times,j::times]=mask
	return n_mask

def LoadLandsatData( dirname ):
    '''
    Load Relevant Bands for the Current Satellite Data
    '''

    if any( [os.path.split(dirname)[-1].startswith( s ) for s in ['LC8', 'LC08']] ): bidx = range( 2, 8 );dataset='Landsat'
    elif any( [os.path.split(dirname)[-1].startswith( s ) for s in ['LE7', 'LE07','LT5','LT05']] ): bidx = list(range( 1, 6 ))+[7];dataset='Landsat'
    else: bidx = list(range( 2, 5 ))+[8,11,12];dataset='S2'

    #提取文件夹下的文件名信息
    bnames=[]
    for band_idx in bidx:
        for name in os.listdir(dirname):
            if ((os.path.splitext(name.split('_')[-1])[0] == 'B{}'.format(band_idx)) or (os.path.splitext(name.split('_')[-1])[0] == 'B{}0'.format(band_idx))) and os.path.splitext(name)[-1] in ['.tif','.TIF','.tiff','.TIFF']:
                bnames.append(os.path.join(dirname,name))

    if dataset=='Landsat':
        bands = [gdal_array.LoadFile(band) for band in bnames]
        geo = gdal.Open( bnames[0] )
        proj = geo.GetProjection()
        #X，Y影像左上角的东、北方向坐标
        GeoTransf = {
            'PixelSize' : abs( geo.GetGeoTransform()[1] ),
            'X' : geo.GetGeoTransform()[0],
            'Y' : geo.GetGeoTransform()[3],
            'Lx' : bands[0].shape[1],
            'Ly' : bands[0].shape[0]
            }
        geo = None

    if dataset=='S2':
        # set all bands with the same shape
        x,y,lx,ly=[],[],[],[]
        for idx,bname in enumerate(bnames):
            band = gdal_array.LoadFile(bname)
            dtype = band.dtype
            geo = gdal.Open( bname )
            if idx==len(bnames)-1:band = up_sample(band,times=2);pcsize=abs( geo.GetGeoTransform()[1] )/2
            #X，Y影像左上角的东、北方向坐标
            x.append(geo.GetGeoTransform()[0])
            y.append(geo.GetGeoTransform()[3])
            lx.append(band.shape[1])
            ly.append(band.shape[0])
        # bbox
        x0,y0=min(x),max(y)
        x1,y1,x2,y2=[],[],[],[]
        for idx,i in enumerate(x):
            x1.append(int((x[idx]-x0)//pcsize))
            y1.append(int((y0-y[idx])//pcsize))
            x2.append(int((x[idx]-x0)//pcsize+lx[idx]))
            y2.append(int((y0-y[idx])//pcsize+ly[idx]))

        # 计算左上角、右下角相对坐标
        mask0=np.zeros((1,max(y2),max(x2)),dtype=dtype)
        bands=np.empty((0,max(y2),max(x2)),dtype=dtype)
        for idx, bname in enumerate(bnames):
            band = gdal_array.LoadFile(bname)
            if idx==len(bnames)-1:band = up_sample(band,times=2)
            mask = mask0
            mask[:,y1[idx]:y2[idx],x1[idx]:x2[idx]]=band[np.newaxis,:,:]
            bands = np.append(bands,mask,axis=0)

        proj = geo.GetProjection()
        GeoTransf ={
            'PixelSize' : pcsize,
            'X' : x0,
            'Y' : y0,
            'Lx' : bands[0].shape[1],
            'Ly' : bands[0].shape[0]
            }
        geo=None

    return bands, GeoTransf, proj


def ReadSatInfo( dirname ):
    '''
    Get band path for the Current Satellite Info
    '''
    if any( [os.path.split(dirname)[-1].startswith( s ) for s in ['LC8', 'LC08', 'LC09', 'LC9']] ): bidx = range( 2, 8 )
    elif any( [os.path.split(dirname)[-1].startswith( s ) for s in ['LE7', 'LE07','LT5','LT05']] ): bidx = list(range( 1, 6 )) + [7]
    else: bidx = list(range( 2, 5 ))+[8,11,12]
    bands = ['B','G','R','NIR','MIR','SWIR']

    #提取文件夹下的文件名信息
    bnames, band_id = [], []
    for idx, band_idx in enumerate( bidx ):
        for name in os.listdir(dirname):
            band = os.path.splitext( name.split('_')[-1])[0]
            ext = os.path.splitext( name )[-1]
            if band in ['B{}'.format(band_idx), 'B{}0'.format(band_idx)] and ext in ['.tif','.TIF','.tiff','.TIFF']:
                bnames.append( os.path.join( dirname, name ) )
                band_id.append( bands[idx] )

    #_B80, _B8
    for name in os.listdir( dirname ):
        band = os.path.splitext( name.split('_')[-1])[0]
        ext = os.path.splitext( name )[-1]
        if band in ['B8', 'B80'] and ext in ['.tif','.TIF','.tiff','.TIFF']:
            pan = os.path.join( dirname, name )
            break
        else:
            pan = None

    return bnames, pan, band_id

def read_geo( geo ):
    proj = geo.GetProjection()
    proj = osr.SpatialReference( proj )
    GeoTransf = {
        'PixelSize' : abs( geo.GetGeoTransform()[1] ),
        'X' : geo.GetGeoTransform()[0],
        'Y' : geo.GetGeoTransform()[3],
        'Lx' : geo.RasterXSize,
        'Ly' : geo.RasterYSize
        }
    return GeoTransf, proj


def projTrans(inshp, outshp, outEPSG, inEPSG=4326):
    # Set/Transfer projection to targeted one(outEPSG); inEPSG by default is set to be 4326, which is the EPSG code of WGS84, set inEPSG when shp is not defined.
    name = os.path.splitext(os.path.basename(outshp))[0]
    driver = ogr.GetDriverByName('ESRI Shapefile')

    inputDataSource = driver.Open(inshp, 0)
    inputLayer = inputDataSource.GetLayer(0)
    shp_srs = inputLayer.GetSpatialRef()
    if shp_srs == None or shp_srs.GetAuthorityCode('PROJCS') == None:
        EPSG_NUM = inEPSG
    else:
        EPSG_NUM = shp_srs.GetAuthorityCode('PROJCS')
    inosr = osr.SpatialReference()

    # print(dir(shp_srs))
    # print(help(shp_srs))
    inosr.ImportFromEPSG(int(EPSG_NUM))

    outosr = osr.SpatialReference()
    outosr.ImportFromEPSG(outEPSG)
    # print(inosr,outosr)
    trans = osr.CoordinateTransformation(inosr, outosr)
    # 读取矢量文件，获取图层
    driver = ogr.GetDriverByName('ESRI Shapefile')
    inds = ogr.Open(inshp, 0)
    inlayer = inds.GetLayer()
    layerDefn = inlayer.GetLayerDefn()
    # print(layerDefn.GetName(),layerDefn.GetGeomType())

    # 创建输出文件
    outds = driver.CreateDataSource(outshp)
    outlayer = outds.CreateLayer(name, outosr, geom_type=layerDefn.GetGeomType())
    # driver.CopyDataSource(inds, outshp)
    # outds.CopyLayer(inlayer,name+'_cvt',['OVERWRITE=YES'])#ogr.wkbPolygon

    # 定义输出属性表信息
    feature = inlayer.GetFeature(0)  # 读取一个要素，以便获取表头信息
    infieldcount = feature.GetFieldCount()
    # print(infieldcount)
    for attr in range(infieldcount):
        infielddefn = feature.GetFieldDefnRef(attr)
        # print(infielddefn.GetName())#输出输入矢量的所有字段名称
        outlayer.CreateField(infielddefn)
    # inlayer.ResetReading()
    # print(dir(feature))
    # 获取输出文件属性表信息
    outfielddefn = outlayer.GetLayerDefn()

    # 遍历输入矢量文件，对每一要素投影转换
    infeature = inlayer.GetNextFeature()
    while infeature:
        geom = infeature.GetGeometryRef()
        # print(geom)
        geom.Transform(trans)
        outfeature = ogr.Feature(outfielddefn)
        outfeature.SetGeometry(geom)
        # outfeature.SetField('name',infeature.GetField('name'))
        outlayer.CreateFeature(outfeature)

        infeature.Destroy()
        outfeature.Destroy()
        infeature = inlayer.GetNextFeature()
    # 清除缓存
    inds.Destroy()
    outds.Destroy()

def tif2png_all(tifdir,pngdir):
	if not os.path.isdir(pngdir):os.mkdir(pngdir)
	for f in os.listdir(tifdir):
		tiffile = os.path.join( tifdir, f )
		name = os.path.splitext(f)[0]
		pngfile = os.path.join(pngdir,name+'.png')
		tif2png( tiffile, pngfile )


def tif2png( tiffile, pngfile ):
    '''
    convert tif file to png file
    :param tiffile:
    :param pngfile:
    :return:None
    '''
    arr,_,_ = load( tiffile )
    if len(arr.shape)==2:pass
    elif len(arr.shape)==3:arr=arr.transpose((1,2,0))
    arr2png( arr, pngfile, STD=True )


def tifdir2gif( tifdir, ngif, fps=10, fsize=20, color = ( 255,255,255 ), loc=(0.1,0.9) ):
    pngdir = os.path.splitext( ngif )[0]
    rawdir = os.path.join(pngdir, 'raw')
    wordir = os.path.join(pngdir, 'word')
    if not os.path.isdir( pngdir ):os.mkdir( pngdir )
    if not os.path.isdir(rawdir): os.mkdir(rawdir)
    if not os.path.isdir(wordir): os.mkdir(wordir)

    tif_files = [os.path.join(tifdir, i) for i in os.listdir(tifdir)]

    names = []
    for idx, tiffile in enumerate(tif_files):
        name = os.path.splitext(os.path.basename(tiffile))[0]
        # if not name.startswith('2018'):continue
        names.append(name)
        # print(name)
        arr = gdal_array.LoadFile(tiffile)# [:, :, np.newaxis]  # [:3,:,:]#
        if len(arr.shape)==3:arr = arr.transpose((1,2,0))
        arr2png( arr, os.path.join( rawdir, '%s.png' % idx) )

    # png_files = [os.path.join(pngdir, i) for i in os.listdir(pngdir)]

    create_gif( rawdir, ngif, framesPerSec = fps )

    save_date = names
    add_text( ngif, save_date, ngif, framesPerSec = fps, check = False, size = fsize, color = color, loc = loc )

def create_gif( picdir, gif_name, framesPerSec=10 ):
    """
    :param picdir: directory of pngfiles
    :param gif_name:the path of giffile
    :param framesPerSec:
    """
    time = 1/framesPerSec
    image_list = [os.path.join(picdir,i) for i in os.listdir(picdir) if os.path.splitext(i)[-1] in ['.png','.tif']]
    # image_list.sort(key=lambda x:eval(os.path.splitext(os.path.basename(x))[0]),reverse=0)
    frames = []
    for image_name in image_list:
        image = imageio.imread(image_name)#[:,:,:]
        image = Image.fromarray(np.uint8(image))
        frames.append(image)
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', fps=framesPerSec, loop=0 )# loop=0为循环播放,3为3次播放
    return


def add_word( image_list, location, size, font, color, text ):
    # text可以是list或单个字符串
    newdir = os.path.join(os.path.dirname(os.path.dirname(image_list[0])), 'word')
    if not os.path.isdir(newdir): os.mkdir(newdir)

    i = 0
    if size > 72:
        size = 72
    elif size < 5:
        size = 5
    for idx, image in enumerate(image_list):
        img = Image.open(image)
        draw = ImageDraw.Draw(img)
        ttfont = ImageFont.truetype(font, size)
        if type(text) == type([]):
            print(text[idx])
            draw.text(location, text[idx], fill=color, font=ttfont)
        elif type(text) == type(''):
            draw.text(location, text, fill=color, font=ttfont)

        img.save( newdir + '/%s.png'%text[idx], 'png' )
        image_list[i] = newdir + '/%s.png'%text[idx]
        i = i + 1
    return image_list


def analyseImage(path):
    '''''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def processImage( path ):
    '''''
    Iterate the GIF, extracting each frame.
    '''
    pngdir = os.path.splitext( path )[0]
    rawdir = os.path.join(pngdir, 'raw')
    if not os.path.isdir( pngdir ):os.mkdir( pngdir )
    if not os.path.isdir(rawdir): os.mkdir(rawdir)

    mode = analyseImage(path)['mode']
    im = Image.open(path)

    i = 0
    p = im.getpalette()

    last_frame = im.convert('RGBA')

    # try:
    #     while True:
    #         #print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))

    #         '''''
    #         If the GIF uses local colour tables, each frame will have its own palette.
    #         If not, we need to apply the global palette to the new frame.
    #         '''

    #         print(len(im.getpalette()))
    #         if not im.getpalette():
    #             im.putpalette(p)
    #
    #         new_frame = Image.new('RGBA', im.size)

    #         '''''
    #         Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
    #         If so, we need to construct the new frame by pasting it on top of the preceding frames.
    #         '''
    #         if mode == 'partial':
    #             new_frame.paste(last_frame)

    #         new_frame.paste(im, (0,0), im.convert('RGBA'))
    #         pngpath=os.path.join(rawdir,'%s-%d.png'%(os.path.basename(path).split('.')[-2], i))
    #         new_frame.save( pngpath, 'png')
    #         i += 1
    #         last_frame = new_frame
    #         im.seek( im.tell()+1 )

    try:
        pngpath = os.path.join( rawdir, '%d.png'%i )
        im.save(pngpath, 'png')
        while True:
            im.seek(im.tell() + 1)
            pngpath = os.path.join(rawdir, '%d.png'%i )
            im.save(pngpath, 'png')
            i += 1
    except EOFError:pass
    return (i)


def add_text( gifpath, txt_list, newgif_path, size=20, color = ( 255,255,255 ), framesPerSec=10, font='', loc=(0.1,0.9), check=True ):
    pngdir = os.path.splitext( gifpath )[0]
    rawdir = os.path.join(pngdir, 'raw')
    wordir = os.path.join(pngdir, 'word')
    if not os.path.isdir( pngdir ):os.mkdir( pngdir )
    if not os.path.isdir(rawdir): os.mkdir(rawdir)
    if not os.path.isdir(wordir): os.mkdir(wordir)

    # font = input("字体文件名字（在C:/Windows/Fonts/，默认黑体）:")
    n = processImage( gifpath ) # read pngs from gif
    i = 0
    image_list = []
    while i<n:
        image_list.append( rawdir+'\%d.png'%i )
        i += 1

    if size > 72:size = 72
    elif size < 5:size = 5

    if font == '':font = 'simhei.ttf'
    font = 'C:\\Windows\\Fonts\\' + font

    img = Image.open(image_list[0])
    location = (int(img.size[0]*loc[0]), int(img.size[1])*loc[1] - size*4/3)

    word_list = add_word( image_list, location, size, font, color, txt_list)
    if check:input('check the word_pic dir to delete bad pics\n Enter if finished:')
    create_gif( wordir, newgif_path, framesPerSec )

def arr2png( arr, pngpath, STD=False ):
    '''
    Converts an 2D or 3D nd-array to a png,as uint-8 format; if 2D gray image, (Lx,Ly,1) is illegal.
    :param arr:nd-array
    :param pngpath:
    :param STD:if true, normalize to 0-255;else, normalize to by [-1,1]
    :return:
    '''

    # arr格式转换为uint8，首先得把数据域拉伸到对应格式的范围0-255,再用np.uint8()才行
    # arr为二维或者三维,三维的话位深在第三维度
    # imageio.imsave(png_path,-IDX)
    dtype = arr.dtype
    if STD:arr = (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))*255
    elif dtype in ['float32','float64']:arr = (arr+1)*255/2
    elif dtype in ['uint16']:arr = arr/np.nanmax(arr)*255#65535
    elif dtype == 'uint8':pass
    arr = arr.astype( np.uint8 )
    img = Image.fromarray( arr )
    img.save( pngpath )

def shpclip( inputfilename, clipfilename, outputfilename ):
    '''
    usage: clip a layer through an input Methodlayer like ArcMAP clip tool\n
    parameters:
    inputfilename type str,
    clipfilename type str,
    outputfilename type str
    '''
    # cliping...
    driver = ogr.GetDriverByName('ESRI Shapefile')
    inputDataSource = driver.Open(inputfilename, 0)
    clipDataSource = driver.Open(clipfilename, 0)
    outputDataSource = driver.CreateDataSource(outputfilename)

    inputLayer = inputDataSource.GetLayer(0)
    clipLayer = clipDataSource.GetLayer(0)
    papszLCO = []

    shp_srs = inputLayer.GetSpatialRef()
    EPSG_NUM = shp_srs.GetAttrValue('AUTHORITY', 1)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG( int(EPSG_NUM) )

    #print(inputLayer.GetNextFeature().GetGeometryRef())

    Defn = inputLayer.GetLayerDefn()
    #ogr.wkbPolygon.（wkbPoint,wkbLineString,wkbPolygon）
    outputLayer = outputDataSource.CreateLayer('clipped', proj, Defn.GetGeomType(), papszLCO)

    for i in range(Defn.GetFieldCount()):
        fielddefn = Defn.GetFieldDefn(i)
        outputLayer.CreateField(fielddefn)
        # print("%s:  %s"%(fielddefn.GetNameRef(),fielddefn.GetFieldTypeName(fielddefn.GetType())))

    outputLayer = outputDataSource.GetLayer(0)

    '''
    strFilter = "Shape * = '面'"
    clipLayer.SetAttributeFilter(strFilter)'''
    inputLayer.Clip(clipLayer, outputLayer)

    outputDataSource.Release()



def select_img(collection, aoi, max_cld=None, txtfile=None, refile=None, subfile=None, nunfile=None, supple=True,
               CF=True, max_v=2000, rm_incomplete_images=False):
    # folium.Map.addLayer = addLayer
    # my_Map = folium.Map(location=aoi.centroid().getInfo()['coordinates'].reverse(),zoom_start=8,height=500)

    my_Map = geemap.Map()
    display(my_Map)
    my_Map.centerObject(ee.FeatureCollection(aoi), zoom=13)

    # ----------determine of visparam----------
    Satellite = collection.get('system:id').getInfo()
    if Satellite == 'COPERNICUS/S2':
        if CF: L8_col_CF = collection.map(maskS2sr)
        visParams = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 6000}
        res = 10
    elif Satellite in ['LANDSAT/LC09/C02/T1', 'LANDSAT/LC09/C02/T1_TOA', 'LANDSAT/LC08/C02/T1',
                       'LANDSAT/LC08/C02/T1_TOA', 'LANDSAT/LE07/C02/T1', 'LANDSAT/LE07/C02/T1_TOA',
                       'LANDSAT/LT05/C02/T1', 'LANDSAT/LT05/C02/T1_TOA', 'LANDSAT/LT04/C02/T1',
                       'LANDSAT/LT04/C02/T1_TOA']:
        if Satellite in ['LANDSAT/LC08/C02/T1', 'LANDSAT/LC08/C02/T1_TOA']:
            if CF: L8_col_CF = collection.map(maskL8sr)
            visParams = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 30000}
        if Satellite in ['LANDSAT/LC09/C02/T1', 'LANDSAT/LC09/C02/T1_TOA']:
            if CF: L8_col_CF = collection.map(maskL9sr);
            visParams = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 30000}
        if Satellite in ['LANDSAT/LT04/C02/T1', 'LANDSAT/LT04/C02/T1_TOA', 'LANDSAT/LT05/C02/T1',
                         'LANDSAT/LT05/C02/T1_TOA', 'LANDSAT/LE07/C02/T1', 'LANDSAT/LE07/C02/T1_TOA']:
            if CF: L8_col_CF = collection.map(maskL8sr)
            visParams = {'bands': ['B3', 'B2', 'B1'], 'min': 0, 'max': max_v}
        res = 30
    print(visParams)
    # -----------get image infomation that have been done--------------
    if supple:
        if txtfile is not None and os.path.isfile(txtfile):
            save_date = list(set(open(txtfile, 'r').read().split(';')))
        else:
            save_date = []
        if refile is not None and os.path.isfile(refile):
            ref_date = list(set(open(refile, 'r').read().split(';')))
        else:
            ref_date = []
        if subfile is not None and os.path.isfile(subfile):
            sub_date = list(set(open(subfile, 'r').read().split(';')))
        else:
            sub_date = []
        if nunfile is not None and os.path.isfile(nunfile):
            nun_date = list(set(open(nunfile, 'r').read().split(';')))
        else:
            nun_date = []
    col_size = collection.size().getInfo()
    indexList = collection.reduceColumns(ee.Reducer.toList(), ["system:index"]).get("list")
    dt_time = indexList.getInfo()

    # --------去掉重复日期的影像----------
    if Satellite == 'COPERNICUS/S2':
        date = np.array([int(i.split('_')[0][:8]) for i in dt_time])
    elif Satellite in ['LANDSAT/LC09/C02/T1', 'LANDSAT/LC09/C02/T1_TOA', 'LANDSAT/LC08/C02/T1',
                       'LANDSAT/LC08/C02/T1_TOA', 'LANDSAT/LE07/C02/T1', 'LANDSAT/LE07/C02/T1_TOA',
                       'LANDSAT/LT05/C02/T1', 'LANDSAT/LT05/C02/T1_TOA', 'LANDSAT/LT04/C02/T1',
                       'LANDSAT/LT04/C02/T1_TOA']:
        date = np.array([int(i.split('_')[-1]) for i in dt_time])
    _, idx = np.unique(date, return_index=True)
    dt_time = [i for idxx, i in enumerate(dt_time) if idxx in idx]

    # ---------------filter images by cloud coverage--------------
    if CF:
        cloud_per, rm_idx = [], []
        for idx in dt_time:
            if supple and (idx in save_date + ref_date + sub_date + nun_date):
                rm_idx.append(idx)
                continue
            image = ee.Image(collection.filter(ee.Filter.eq('system:index', idx)).first()).clip(aoi)
            image_cf = ee.Image(L8_col_CF.filter(ee.Filter.eq('system:index', idx)).first()).clip(aoi)
            # points = scene.addBands(ee.Image.pixelLonLat()).sample({'region': roi,'geometries': True}).size().getInfo()
            # _,data1=getImgData(scene)
            # _,data2=getImgData(scene_cf)
            # print(len(data1),len(data2))
            # cloud_per=1-len(data2)/len(data1)

            # 计算image的云量
            properties = image.propertyNames()
            mask = image.select('B1').lt(1000000)
            mask_cf = image_cf.select('B1').lt(1000000)
            cnt1 = mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=res,
                maxPixels=1e13).getInfo()['B1']
            cnt2 = mask_cf.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=res,
                maxPixels=1e13).getInfo()['B1']
            if cnt1 == 0:
                rm_idx.append(idx)
                continue
            cloud_per.append((1 - cnt2 / cnt1) * 100)
            # print('%s云量：%s'%(idx,(1-cnt2/cnt1)*100))
        dt_time = [i for i in dt_time if i not in rm_idx]
        if cloud_per == [] and dt_time == []: return save_date, ref_date, sub_date, nun_date
        # 将影像id按云量从小到大排序
        id_and_cloud = list(zip(dt_time, cloud_per))
        id_and_cloud.sort(key=lambda x: x[1], reverse=0)
        dt_time, cloud_per = list(zip(*id_and_cloud))
        if max_cld is None:
            max_idx = len(cloud_per) - 1
        else:
            for idx, i in enumerate(cloud_per):
                if i > max_cld:
                    max_idx = idx
                    break
        dt_time, cloud_per = dt_time[:max_idx + 1], cloud_per[:max_idx + 1]
    else:
        rm_idx = []
        for idx in dt_time:
            if supple and (idx in save_date + ref_date + sub_date + nun_date):
                rm_idx.append(idx)
                continue
        dt_time = [i for i in dt_time if i not in rm_idx]
    # ----------filter images by valid images-------------
    if rm_incomplete_images:
        valid_cnt = []
        for idx in dt_time:
            image = ee.Image(collection.filter(ee.Filter.eq('system:index', idx)).first())
            cnt = image.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=aoi, maxPixels=1e13,
                scale=res).values().get(0).getInfo();
            valid_cnt.append(int(cnt))
        max_valid_cnt = ee.Feature(aoi).area().getInfo() / res ** 2
        rm_idx = []
        for id, idx in enumerate(dt_time):
            if valid_cnt[id] / max_valid_cnt < 0.7:
                rm_idx.append(idx)
                if nun_date is not None:
                    nun_date.append(idx)
                    f = open(nunfile, "w")
                    f.write(';'.join(nun_date))
                    f.close()
        dt_time = [i for i in dt_time if i not in rm_idx]

    print('共%s幅图需要筛分' % len(dt_time))
    # --------mannual select needed images--------
    for img_id, idx in enumerate(dt_time):
        image = collection.filter(ee.Filter.eq('system:index', idx)).first().clip(aoi)
        my_Map.addLayer(image, visParams, '%s' % idx)
        if CF:
            key = input('%s是否保存%s,云量：%s' % (img_id, idx, cloud_per[img_id]))
        else:
            key = input('%s是否保存%s' % (img_id, idx))

        if key in ['y', 'Y']:
            save_date.append(idx)
            f = open(txtfile, "w")
            f.write(';'.join(save_date))
            f.close()
        if key in ['r', 'R']:
            ref_date.append(idx)
            f = open(refile, "w")
            f.write(';'.join(ref_date))
            f.close()
        if key in ['s', 'S']:
            sub_date.append(idx)
            f = open(subfile, "w")
            f.write(';'.join(sub_date))
            f.close()
        if key in ['n', 'N', '']:
            nun_date.append(idx)
            f = open(nunfile, "w")
            f.write(';'.join(nun_date))
            f.close()
        if key == 'done': break

    # valid_Col = ee.ImageCollection(col_name).filter(ee.Filter.inList("system:index",save_date)).sort('system:time_start',True)
    # valid_Col = clip(valid_Col,aoi)
    # image_ids = valid_Col.aggregate_array('system:index').getInfo()
    return save_date, ref_date, sub_date, nun_date


def RS_info( name ):
    # get row/col,date,satellite based on name
    # just for landsat now
    name = os.path.splitext(os.path.basename(name))[0]
    if '_' not in name:
        if len(name)==8:
            RowCol = None
            date = name
            sat = None
        else:
            RowCol = name[3:9]
            date = yjday2date( name[9:16] )
            sat = name[:2]+'0'+name[:2]
    else:
        collection = name.split('_')
        if collection[0].startswith('L'):# Landsat
            sat = collection[0]
            if len(collection)==7: # Landsat Collection from USGS, LC08_L1TP_131037_20130723_20200912_02_T1
                RowCol,date = collection[2],collection[3]
            elif len(collection)==3: # Landsat from GEE, LC09_124039_20220824
                RowCol, date = collection[1], collection[2]
            else: # PreCollection
                # Already in Julian Calendar
                year = name[9:13]
                jday = name[13:16]
                date = yjday2date( year+'_'+jday )
        else:# S2 dataset, 20180626T042701_20180626T043258_T46SFD
            sat = 'S2'
            RowCol = None
            date = collection[0][:8]
    return sat, RowCol, date

def yjday2date( yjdaystr ):
    year, jday = yjdaystr[:4], yjdaystr[-3:]
    date = datetime.datetime.strptime(year+'_'+jday, '%Y_%j').date()
    return date.strftime('%Y%m%d')


def dn_col(collection, aoi_pnts, mdir, bands, overwrite=False, fps=2):
    aoi = ee.Geometry.Polygon(aoi_pnts)
    col_size = collection.size().getInfo()
    dt_time = collection.reduceColumns(ee.Reducer.toList(), ["system:index"]).get("list").getInfo()
    imgdir = os.path.join(mdir, 'img')
    geodir = os.path.join(mdir, 'geo')
    prjdir = os.path.join(mdir, 'prj')
    tifdir = os.path.join(mdir, 'tif')
    pngdir = os.path.join(mdir, 'png')
    if not os.path.isdir(mdir): os.mkdir(mdir)
    if not os.path.isdir(geodir): os.mkdir(geodir)
    if not os.path.isdir(prjdir): os.mkdir(prjdir)
    if not os.path.isdir(imgdir): os.mkdir(imgdir)
    if not os.path.isdir(tifdir): os.mkdir(tifdir)
    if not os.path.isdir(pngdir): os.mkdir(pngdir)

    # ------------去重--------------
    date = [int(RS_info(i)[2]) for i in dt_time]
    _, idx = np.unique(date, return_index=True)
    dt_time = [i for idxx, i in enumerate(dt_time) if idxx in idx]
    # -------------------------------
    flag = False
    for idx in dt_time:
        print('processing:%s' % idx)
        imgfile = os.path.join(imgdir, '.'.join((idx, 'npy')))
        geofile = os.path.join(geodir, '.'.join((idx, 'p')))
        prjfile = os.path.join(prjdir, '.'.join((idx, 'prj')))
        if all([os.path.isfile(imgfile), os.path.isfile(geofile), os.path.isfile(prjfile), not overwrite]):
            print('skipped:%s' % imgfile)
            continue

        img = ee.Image(collection.filter(ee.Filter.eq('system:index', idx)).first()).clip(aoi).select(bands)

        if not flag: flag = True
        # img = collection.first().clip(aoi).select(bands)
        GeoTransf, EPSG = get_proj_trans(img)
        sub_aois_WGS, rel_coor = cut_aoi(img, aoi_pnts)
        lx, ly = GeoTransf['Lx'], GeoTransf['Ly']

        # set data_type
        bandList = ee.List(ee.Dictionary(ee.Algorithms.Describe(img)).get('bands'))
        b1 = ee.Dictionary(bandList.get(0))
        data_type = b1.get("data_type").getInfo()['precision']
        if data_type == 'int':
            dtype = np.uint16
        else:
            dtype = np.float64

        whole_img = np.zeros((ly + 25, lx + 25, len(bands)), dtype=dtype)  # 在你的aoi切割到影像边缘时候，尺寸可能会少一点点，不过不影响数据准确性
        for aoi_id, sub_aoi in enumerate(sub_aois_WGS):
            time.sleep(1)
            sub_aoi = ee.Geometry.Polygon(sub_aoi)
            np_img = geemap.ee_to_numpy(img, bands=bands, region=sub_aoi)# , default_value=0
            # show_npy( np_img )
            try:
                np_img = np_img.astype(dtype)
            except AttributeError:
                time.sleep(60)
                dn_col(collection, aoi_pnts, mdir, bands, overwrite=overwrite)
                return
            sub_ly, sub_lx = np_img.shape[0], np_img.shape[1]
            whole_img[rel_coor[aoi_id][1]:rel_coor[aoi_id][1] + sub_ly,
            rel_coor[aoi_id][0]:rel_coor[aoi_id][0] + sub_lx, :] = np_img
        np.save(imgfile, whole_img[:ly, :lx, :])
        print(r'saved:%s' % imgfile)

        with open(geofile, 'wb') as gf:
            pickle.dump(GeoTransf, gf)

        proj = osr.SpatialReference()
        proj.ImportFromEPSG(EPSG)
        fn = open(prjfile, 'w')
        fn.write(proj.ExportToWkt())
        fn.close()
    arr2tif_all(imgdir, geodir, prjdir, tifdir, ROI=None, overwrite=overwrite)
    tif2png_all(tifdir, pngdir)
    tifdir2gif(tifdir, r'%s\Gif.gif' % mdir, fps=fps)


def Dn_Imgs( pdir, aoi_pnts, year_range, Satellite, band='Multi', mode=1, WRS_PATH=None, WRS_ROW=None, CLOUD_COVER=None, overwrite=False, month_range=[5,10], supple=True, CF=True, max_v=2000, rm_incomplete_images=False, max_cld=50, TOA=False ):
    def dn_imgs_with_different_bands( valid_Col, aoi_pnts, dir1, Satellite, band, overwrite ):
        if band == 'Multi':
            if Satellite in ['LC8', 'LC08', 'LC09']:
                dn_col( valid_Col, aoi_pnts, dir1, ['B2', 'B3', 'B4', 'B6'], overwrite=overwrite )
            elif Satellite in ['LE07', 'LT04', 'LT05']:
                dn_col(valid_Col, aoi_pnts, dir1, ['B1', 'B2', 'B3', 'B5'], overwrite=overwrite)
            if Satellite in ['LE07', 'LC08', 'LC09']:
                dn_col(valid_Col, aoi_pnts, dir2, ['B8'], overwrite=overwrite)
            elif Satellite == 'S2':
                dn_col(valid_Col, aoi_pnts, dir1, ['B2', 'B3', 'B4'], overwrite=overwrite)
                dn_col(valid_Col, aoi_pnts, dir2, ['B11'], overwrite=overwrite)
        elif band == 'MNDWI':
            if Satellite in ['LC8', 'LC08', 'LC09']:
                dn_col(valid_Col.map(lc8_addMNDWI).select('MNDWI'), aoi_pnts, dir1, ['MNDWI'], overwrite=overwrite)
            elif Satellite in ['LE07', 'LT04', 'LT05']:
                dn_col(valid_Col.map(lt5_addMNDWI).select('MNDWI'), aoi_pnts, dir1, ['MNDWI'], overwrite=overwrite)
            elif Satellite == 'S2':
                dn_col(valid_Col.map(s2_addMNDWI), aoi_pnts, dir1, ['MNDWI'], overwrite=overwrite)
        elif band == 'NDWI':
            if Satellite in ['LC8', 'LC08', 'LC09']:
                dn_col(valid_Col.map(lc8_addNDWI).select('NDWI'), aoi_pnts, dir1, ['NDWI'], overwrite=overwrite)
            elif Satellite in ['LE07', 'LT04', 'LT05']:
                dn_col(valid_Col.map(lt5_addNDWI).select('NDWI'), aoi_pnts, dir1, ['NDWI'], overwrite=overwrite)
            elif Satellite == 'S2':
                dn_col(valid_Col.map(s2_addNDWI), aoi_pnts, dir1, ['NDWI'], overwrite=overwrite)
    # mode: 1, select & download; 2, just select; 3, just download; 4, dn ref; 5, dn sub imgs.
    if type( aoi_pnts ) is str:
        roi = ee.FeatureCollection( aoi_pnts ) # 矢量数据
        aoi = roi.geometry()
        aoi_pnts= aoi.coordinates().getInfo()
    else:
        aoi = ee.Geometry.Polygon(aoi_pnts)#aoi_pnts
        roi = ee.FeatureCollection(aoi)

    syear, eyear = year_range[0], year_range[1]

    sat = {'LT04':'LANDSAT/LT04/C02/T1_TOA','LT05':'LANDSAT/LT05/C02/T1_TOA','LE07':'LANDSAT/LE07/C02/T1_TOA','LC08':'LANDSAT/LC08/C02/T1_TOA','LC09':'LANDSAT/LC09/C02/T1_TOA','S2':'COPERNICUS/S2'}
    show_sat = {'LT04':'LANDSAT/LT04/C02/T1','LT05':'LANDSAT/LT05/C02/T1','LE07':'LANDSAT/LE07/C02/T1','LC08':'LANDSAT/LC08/C02/T1','LC09':'LANDSAT/LC09/C02/T1','S2':'COPERNICUS/S2'}
    if TOA: show_sat = sat
    col_name = sat[Satellite]

    IMGdir = os.path.join(pdir,'GEE_IMG')
    mdir = os.path.join(IMGdir,Satellite)

    txtfile = os.path.join(mdir,'info.txt')
    refile = os.path.join(mdir,'ref.txt')
    subfile = os.path.join(mdir,'sub.txt')
    nunfile = os.path.join(mdir,'nun.txt')

    if not os.path.isdir(pdir):os.mkdir(pdir)
    if not os.path.isdir(IMGdir):os.mkdir(IMGdir)
    if not os.path.isdir(mdir):os.mkdir(mdir)


    col = ee.ImageCollection(show_sat[Satellite]).filterBounds(roi).filter(ee.Filter.calendarRange(syear, eyear,'year'))\
    .filter(ee.Filter.calendarRange( month_range[0], month_range[1], 'month'))
    print( get_proj_trans(col.first().clip(aoi)) )
    if WRS_PATH is not None:col = col.filter(ee.Filter.eq('WRS_PATH', WRS_PATH))
    if WRS_ROW is not None:col = col.filter(ee.Filter.eq('WRS_ROW', WRS_ROW))
    if CLOUD_COVER is not None:col = col.filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER))
    #--------选择影像---------
    if mode in [1,2]:
        try:
            save_date,ref_date,sub_date,nun_date = select_img( col, aoi, txtfile=txtfile, refile=refile, subfile=subfile, nunfile=nunfile, supple=True, CF=CF, max_v=max_v, rm_incomplete_images=rm_incomplete_images, max_cld=max_cld )
        except:
            time.sleep(60)
            save_date,ref_date,sub_date,nun_date = select_img( col, aoi, txtfile=txtfile, refile=refile, subfile=subfile, nunfile=nunfile, supple=True, CF=CF, max_v=max_v,rm_incomplete_images=rm_incomplete_images, max_cld=max_cld )
    #--------下载影像----------
    if mode in [1,3,4,5]:
        #读取index信息
        if mode == 4:
            with open(refile,'r') as f:date=f.read()
        elif mode == 5:
            with open(subfile,'r') as f:date=f.read()
        else:
            with open(txtfile,'r') as f:date=f.read()
        save_date = date.split(';')

        valid_Col = ee.ImageCollection( col_name ).filter(ee.Filter.inList("system:index",save_date)).sort('system:time_start',True).filterBounds(roi).filter(ee.Filter.calendarRange(syear, eyear,'year')).filter(ee.Filter.calendarRange( month_range[0], month_range[1], 'month'))
        valid_Col = clip( valid_Col, aoi )
        image_ids = valid_Col.aggregate_array('system:index').getInfo()
        print('Images need to be Dn: ', len(image_ids))

        # 下载影像
        if band == 'Multi':
            if mode in [1, 2, 3]:
                dir1 = os.path.join(mdir, 'b1')
                dir2 = os.path.join(mdir, 'b2')
            elif mode == 4:
                dir1 = os.path.join(mdir, 'b1_ref')
                dir2 = os.path.join(mdir, 'b2_ref')
            elif mode == 5:
                dir1 = os.path.join(mdir, 'b1_sub')
                dir2 = os.path.join(mdir, 'b2_sub')
            if not os.path.isdir(dir1): os.mkdir(dir1)
            if not os.path.isdir(dir2): os.mkdir(dir2)
        else:
            if mode in [1, 2, 3]:
                dir1 = os.path.join(mdir, band)
            elif mode == 4:
                dir1 = os.path.join(mdir, '%s_ref'%band )
            elif mode == 5:
                dir1 = os.path.join(mdir, '%s_sub'%band )
            if not os.path.isdir(dir1): os.mkdir(dir1)

        try:
            dn_imgs_with_different_bands(valid_Col, aoi_pnts, dir1, Satellite, band, overwrite)
        except:
            time.sleep(10)
            dn_imgs_with_different_bands(valid_Col, aoi_pnts, dir1, Satellite, band, overwrite)

