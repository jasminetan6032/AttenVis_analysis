#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:17:20 2024

@author: jwt30
"""

import mne
import os
from PIL import Image

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

local_dir = local_dir = '/local_mount/space/hypatia/2/users/Jasmine/AttenVis'

report = mne.Report(title="STCs: visual and atten labels")

tiff_files = find_files('label.tiff',local_dir)

for image_path in tiff_files:
    participant = os.path.split(image_path)[1].split('_')[0]
    section = participant
    label_type = os.path.split(image_path)[1].split('_')[-2]
    png_path = image_path.replace('.tiff','.png')
    if not os.path.isfile(png_path):
        im = Image.open(image_path)
        im.save(image_path.replace('.tiff','.png'))
    
    report.add_image(
        image=png_path, 
        title=label_type,
        section = section,
        tags = label_type
    )
report.save("vis_atten_labels.html", overwrite=True)