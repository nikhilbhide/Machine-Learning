"""


@author: nik
"""
import cv2

def jpeg_res(filename):
   """"This function prints the resolution of the jpeg image file passed into it"""
   img = cv2.imread(filename)
   if(type(img)!=type(None)):
       height, width, channels = img.shape
       print(height,width,channels)
       if height<224 or width< 224:
           print("height and width are not matching. removing file "+filename)
           os.remove(filename)
   else:
       os.remove(filename)
        
           
           
import os
from pathlib import Path
import pathlib

basepath = Path('.')
files_in_basepath = basepath.iterdir()

#iterate over all entries in the  directory
#remove files whih are not jped or jpg
for file in files_in_basepath:
    if file.is_file():
        filename, file_extension = os.path.splitext(file.name)
        jpeg_res(file.name)
    else:
        print("file format problem so removing file "+filename)
        os.remove(file)
