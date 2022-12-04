import geopandas as gpd
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup 
import gzip
import glob
import os
from pathlib import Path
import json
import logging
from tqdm import tqdm 
import shutil

def untar_file(tar_file, PARENT_PATH) :
    '''
    Untar and save a tar file.
    Input : filepath str. 
    Return : content Json 
    '''
    print(tar_file)
    with gzip.open(tar_file) as f :
        filename = tar_file.split('\\')[-1].replace('.gz','')
        filepath = os.path.join(PARENT_PATH, filename)
        content = f.readlines()
        content = [line.decode('utf8') for line in content]
        #df = gpd.read_file(content)
        with open(filepath,'w') as w :
            w.writelines(content)
    return content

def load_geopandas(json_file) :
    '''
    Load json file and convert to geopandas dataframe.
    Input : json_file Json
    Return : gdf geopandas.dataframe 
    '''
    with open(json_file) as f : 
        data = json.load(f)
        gdf = gpd.GeoDataFrame.from_features(data["features"]) # geopandas dataframe 
    return gdf

def download_file(url_file, couche,save_folder = False) :
    '''
    
    Download and save file from a cadastre.data.gouv.fr base url. Couche is a keyword for a specific file. If save_folder is specified, create a new directory.
    Input : url_file str,
            couche str,
            save_folder str 
    '''
    if save_folder == False :
        save_folder = url_file.split('/')[-2]
        
    with requests.Session() as S :
        r = S.get(url_file, stream = True)
        soup = BeautifulSoup(r.content, 'html.parser')
        for i in soup.findAll('a') :
            if couche in i.text :
                couche = '/'+i.text
                print(couche)
        r = S.get(url_file+couche, stream = True)
    with open(save_folder+couche, 'wb') as f:
        for chunk in r.iter_content():
            f.write(chunk)

class picture_geometry() :
    '''
    Picture as a class. Use a polygone (with Latitude/longitute coordonates) to center a google maps satellite picture
    with a specified height and width. Zoom should always be 20 for maximum detail.
    '''
    
    def __init__(self,zipcode, havepool, key, _id, polygon, h, w, zoom) :
        self.zipcode = zipcode
        self.key = key
        self.havepool = 1 if havepool == True else 0
        self._id = _id # parcelle id
        self.polygon = polygon
        self.height = h
        self.width = w
        self.zoom = zoom
        
        # --- TO CHANGE 
        self.LOGFILE = './datalog.csv'
        self.API_KEY = 'AIzaSyCshJpLZumLqbStsPdU0BRRqntNHZLFjlU'
        self.BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
        self.form = 'png'
        self.maptype = 'satellite'
        self.border = 'color:0xff0000ff|weight:0|'
        self.filepath = f'{self.havepool}_{self._id}_{self.height}x{self.width}.{self.form}'
        
    def coord_lister(self,geom):
        '''
        Convert polygon into multiple coordinates.
        Return : coords List of tuples(x,y) coordinates
        '''
        coords = list(geom.exterior.coords)
        return (coords)

    def coordinate_features(self) :
        '''
        convert a list of coordinates into a format accepted by the google maps api. Ex : (6.8499536, 43.5275064) will become 
        43.5275064,6.8499536|
        Return a merged string with all coordinates.
        '''
        coordinates = self.polygon.apply(self.coord_lister)
        coordinates = coordinates.values[0]
        coordinates = [f'{cords[1]},{cords[0]}' for cords in coordinates]
        coordinates_str = "|".join(coordinates)
        self.coordinates_str = coordinates_str
        return coordinates_str
    
    def api_url_generator(self):
        '''
        Merge all necessary parameters to call google maps api 
        Return : url str
        '''
        url = f'{self.BASE_URL}format={self.form}&path={self.border}{self.coordinates_str}&size={self.height}x{self.width}&zoom={self.zoom}&maptype={self.maptype}&key={self.API_KEY}'
        #print(url)
        return url
        
    def api_call(self, url) :
        '''
        Context Manager with google maps api call as content 
        Return r requests.Response object
        '''
        with requests.Session() as S :
            r = S.get(url, stream = True)
        
        return r
            
    def save_picture(self, folder = './') :
        '''
        Save picture in . Create new folder if folder doesn't exists
        Input : folder str
        
        '''
        coordinates_str = self.coordinate_features()
        url = self.api_url_generator()
        r = self.api_call(url)
        
        if os.path.exists(folder) == False and folder != './' :
             Path(f"./{folder}").mkdir(parents=True, exist_ok=True)
        
        
        with open(f'{folder}/{self.filepath}', 'wb') as f:
            #print(f'fichier : {self.filepath} // len url {len(url)}')
            for chunk in r.iter_content():
                f.write(chunk)
        
        #with open(self.LOGFILE,'+a') as f :
            #output = f'{self._id};{self.zipcode};{self.havepool};{self.filepath}\n'
            #f.write(output)

# ----------

def create_data_directories(data_dir):
    '''
    Helper to create train/val/test directory for Yolov5 architecture
    '''
    # Supress dataset directory if exists
    if os.path.exists(f"../datasets/{data_dir}/labels") == True :
        shutil.rmtree(f"../datasets/{data_dir}/labels")
    if os.path.exists(f"../datasets/{data_dir}/images") == True :
        shutil.rmtree(f"../datasets/{data_dir}/images")
        
    Path(f"../datasets/{data_dir}/images/train").mkdir(parents=True, exist_ok=True)
    Path(f"../datasets/{data_dir}/images/val").mkdir(parents=True, exist_ok=True)
    Path(f"../datasets/{data_dir}/images/test").mkdir(parents=True, exist_ok=True)
    Path(f"../datasets/{data_dir}/labels/train").mkdir(parents=True, exist_ok=True)
    Path(f"../datasets/{data_dir}/labels/val").mkdir(parents=True, exist_ok=True)
    Path(f"../datasets/{data_dir}/labels/test").mkdir(parents=True, exist_ok=True)

def split_data(labelised_dataset, data_dir, train_ratio = 0.85, valid_ratio = 0.10) :
    '''
    Helper to split data and prepare train phase. Data must be labelised and conform to Yolov5 architecture (images and labels)
    
    '''
    list_png = glob.glob(f'{labelised_dataset}\\images\\*') # put '..\' if labelised_dataset is outside yolo directory
    list_label = glob.glob(f'{labelised_dataset}\\labels\\*')
    n_data = len(list_png)
    print(f'total size : {n_data}')
    #images
    train_list_img = list_png[0: int(n_data * train_ratio)]
    val_list_img = list_png[len(train_list_img) : len(train_list_img)+int(n_data * valid_ratio)]
    test_list_img = list_png[len(train_list_img) + len(val_list_img)::]
    #labels
    train_list_lab = list_label[0:int(n_data * train_ratio)]
    val_list_lab = list_label[len(train_list_lab) : len(train_list_lab)+ int(n_data * valid_ratio)]
    test_list_lab = list_label[len(train_list_lab) + len(val_list_lab)::]
    
    
    print(f'train size : {len(train_list_img)}\nvalidation size : {len(val_list_img)}\ntest size :{len(test_list_img)}')
    for train_img, train_lab in zip(train_list_img, train_list_lab) :
        name_png = train_img.split('\\')[-1]
        name_label = train_lab.split('\\')[-1]
        #print(name_png, name_label)
        shutil.copyfile(train_img, f"..\\datasets\\{data_dir}\\images\\train\\{name_png}")
        shutil.copyfile(train_lab, f"..\\datasets\\{data_dir}\\labels\\train\\{name_label}")
    
    print(f'--- Train set : Done ---')
   
    for val_img, val_lab in zip(val_list_img, val_list_lab) :
        name_png = val_img.split('\\')[-1]
        name_label = val_lab.split('\\')[-1]

        shutil.copyfile(val_img, f"..\\datasets\\{data_dir}\\images\\val\\{name_png}")
        shutil.copyfile(val_lab, f"..\\datasets\\{data_dir}\\labels\\val\\{name_label}")
        
    print(f'--- Validation set : Done ---')
    
    for test_img, test_lab in zip(test_list_img, test_list_lab) :
        name_png = test_img.split('\\')[-1]
        name_label = test_lab.split('\\')[-1]

        shutil.copyfile(test_img, f"..\\datasets\\{data_dir}\\images\\test\\{name_png}")
        shutil.copyfile(test_lab, f"..\\datasets\\{data_dir}\\labels\\test\\{name_label}")
        
    print(f'--- Test set : Done ---')

# ----------

class picture_address() :
    '''
    Picture as a class. Use an address to center a google maps satellite picture
    with a specified height and width. Zoom should always be 20 for maximum detail.
    '''
    
    def __init__(self,address, h, w, zoom) :
        self.address = address
        self.height = h
        self.width = w
        self.zoom = zoom
        
        # --- TO CHANGE 
        self.LOGFILE = './datalog.csv'
        self.API_KEY = 'AIzaSyCshJpLZumLqbStsPdU0BRRqntNHZLFjlU'
        self.BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
        self.form = 'png'
        self.maptype = 'satellite'
        self.filepath = f'{self.address}_{self.height}x{self.width}.{self.form}'
        
    def coord_lister(self,geom):
        '''
        Convert polygon into multiple coordinates.
        Return : List of tuples(x,y) coordinates
        '''
        coords = list(geom.exterior.coords)
        return (coords)

    def coordinate_features(self) :
        '''
        convert a list of coordinates into a format accepted by the google maps api. Ex : (6.8499536, 43.5275064) will become 
        43.5275064,6.8499536|
        Return a merged string with all coordinates.
        '''
        coordinates = self.polygon.apply(self.coord_lister)
        coordinates = coordinates.values[0]
        coordinates = [f'{cords[1]},{cords[0]}' for cords in coordinates]
        coordinates_str = "|".join(coordinates)
        self.coordinates_str = coordinates_str
        return coordinates_str
    
    def api_url_generator(self):
        '''
        Merge all necessary parameters to call google maps api 
       '''
        url = f'{self.BASE_URL}format={self.form}&center={self.address}&size={self.height}x{self.width}&zoom={self.zoom}&maptype={self.maptype}&key={self.API_KEY}'
        #print(url)
        return url
        
    def api_call(self, url) :
        '''
        Context Manager with picture as content 
        '''
        with requests.Session() as S :
            r = S.get(url, stream = True)
        
        return r
            
    def save_picture(self, folder = './') :
        '''
        Save picture in in specific filepath
        '''
        #coordinates_str = self.coordinate_features()
        url = self.api_url_generator()
        r = self.api_call(url)
        
        if os.path.exists(folder) == False and folder != './' :
            Path(f"./{folder}").mkdir(parents=True, exist_ok=True)
        elif os.path.exists(folder) == True and folder != './' :
            shutil.rmtree(f"./{folder}")
            Path(f"./{folder}").mkdir(parents=True, exist_ok=True)
        
        with open(f'{folder}/{self.filepath}', 'wb') as f:
            #print(f'fichier : {self.filepath} // len url {len(url)}')
            for chunk in r.iter_content():
                f.write(chunk)
    