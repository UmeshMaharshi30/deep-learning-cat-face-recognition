# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:02:15 2019

@author: umesh
"""
import shutil
import os
from PIL import Image, ImageDraw
import numpy as np;
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import losses
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import pickle

original_dir_path = "A:\\Machine Learning\\cat dataset\\cat-dataset\\cats\\CAT_06\\";
# duplicate_dir_path = "A:\\Machine Learning\\cat dataset\\duplicate\\";

# original_image = "00001292_000";
# file_extension = ".jpg";

processed_data = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed";

def create_duplicate_of_original_image(original_image_file, root_dir, duplicate_dir):
    shutil.copy(root_dir + original_image_file + ".png", duplicate_dir + original_image_file + ".png");
    

def convert_line_into_lines_array(annot_line):
    annot_line = annot_line.strip();
    points_list = annot_line.split(" ");
    lines = [];
    if points_list[0] != '9': 
        return lines;    
    eyes_and_mouth = [(int(points_list[1]), int(points_list[2])),(int(points_list[3]), int(points_list[4])),(int(points_list[5]), int(points_list[6])),(int(points_list[1]), int(points_list[2]))];
    left_ear = [(int(points_list[7]), int(points_list[8])),(int(points_list[9]), int(points_list[10])),(int(points_list[11]), int(points_list[12])),(int(points_list[7]), int(points_list[8]))];
    right_ear = [(int(points_list[13]), int(points_list[14])),(int(points_list[15]), int(points_list[16])),(int(points_list[17]), int(points_list[18])),(int(points_list[13]), int(points_list[14]))];
    lines.append(eyes_and_mouth);
    lines.append(left_ear);
    lines.append(right_ear);
    return lines;


def annotate_image_based_on_points(file_full_path, original_image_name, lines_arr, annot_dir_path):
    im = Image.open(file_full_path)
    d = ImageDraw.Draw(im)
    line_color = (0, 0, 255)
    index = 0;
    scale = 32;
# =============================================================================
#     scaled_arr = [];
#     for line in lines_arr:
#         scaled_line = [];
#         for p in line:
#             oi = Image.open(raw_original_image_dir + original_image_name);
#             width, height = oi.size;
#             scaled_point = ((scale/width)*p[0],(scale/height)*p[1]);
#             scaled_line.append(scaled_point);
#         scaled_arr.append(scaled_line);
# =============================================================================
        
    for line in lines_arr:
        d.line(line, fill=line_color, width=2)
        im.save(annot_dir_path + "annot_" + original_image_name);
        im = Image.open(annot_dir_path + "annot_" + original_image_name);
        d = ImageDraw.Draw(im);
        index = index + 1
    im.save(annot_dir_path + "annot_" + original_image_name);
    im.close();
 
# create_duplicate_of_original_image(original_image, original_dir_path, duplicate_dir_path);


def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(path + "\\" + name)
    files.sort();
    return files 

# print(len(list_files("A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotations")));    

def append_multiple_files_into_one(files, output_file):
    with open(output_file, 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())
                outfile.write(fname)
                outfile.write('\n')
        outfile.close();

def file_name_from_path(full_path):
    dirs = full_path.split("\\");
    if len(dirs) == 0: 
        return full_path.split(".")[0];
    full_path_with_ext = dirs[len(dirs) - 1];
    return full_path_with_ext.split(".")[0];

def draw_annotations_on_images(annot_data_file, raw_images_dir, annotated_images, original_size_dir):
    annot_file = open(annot_data_file, "r");
    lines = annot_file.readlines();
    for l in lines:
        line_split = l.split(" ");
        print("Annotating " + line_split[len(line_split) - 1]);
        image_name = file_name_from_path(line_split[len(line_split) - 1]) + ".png";
        image = raw_images_dir + "\\" + image_name;
        annotate_image_based_on_points(image, image_name, convert_line_into_lines_array(l),annotated_images, original_size_dir);

def draw_annotations_on_output_images(annot_data_file, raw_images_dir, annotated_images, original_size_dir):
    annot_file = open(annot_data_file, "r");
    lines = annot_file.readlines();
    for l in lines:
        line_split = l.split(" ");
        print("Annotating " + line_split[len(line_split) - 1]);
        image_name = file_name_from_path(line_split[len(line_split) - 1]) + ".png";
        image = raw_images_dir + "\\" + image_name;
        annotate_image_based_on_points(image, image_name, convert_line_into_lines_array(l),annotated_images, original_size_dir);

        
def resize_image(image_full_path, color_output_dir, black_and_white_output_dir):
    img = Image.open(image_full_path)
    new_width  = 64
    new_height = 64
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(color_output_dir + file_name_from_path(image_full_path) + ".png");
    img = img.convert('1')
    img.save(black_and_white_output_dir + file_name_from_path(image_full_path) + ".png");
    img.close();
    
def resize_images_inside_directory(image_dir, color_output_dir, black_and_white_output_dir):
    count = 0;
    files = list_files(image_dir);
    for f in files:
        resize_image(f,color_output_dir,black_and_white_output_dir);
        count = count + 1;

    
def convert_into_byte_arr(scaled_annotation_data_file, image_dir, byte_output_dir):
    scaled_annotation_data_file = open(scaled_annotation_data_file, "r");
    lines = scaled_annotation_data_file.readlines();
    byte_output = byte_output_dir + "\\" + "byte_output.csv";
    byte_output = open(byte_output, 'w')
# =============================================================================
#     left_eye = byte_output_dir + "\\" + "left_eye.csv";
#     left_eye = open(left_eye, 'w')
#     mouth = byte_output_dir + "\\" + "mouth.csv";
#     mouth = open(mouth, 'w')
#     right_eye = byte_output_dir + "\\" + "right_eye.csv";
#     right_eye = open(right_eye, 'w')
#     left_ear_1 = byte_output_dir + "\\" + "left_ear_1.csv";
#     left_ear_1 = open(left_ear_1, 'w')
#     left_ear_2 = byte_output_dir + "\\" + "left_ear_2.csv";
#     left_ear_2 = open(left_ear_2, 'w')
#     left_ear_3 = byte_output_dir + "\\" + "left_ear_3.csv";
#     left_ear_3 = open(left_ear_3, 'w')
#     right_ear_1 = byte_output_dir + "\\" + "right_ear_1.csv";
#     right_ear_1 = open(right_ear_1, 'w')
#     right_ear_2 = byte_output_dir + "\\" + "right_ear_2.csv";
#     right_ear_2 = open(right_ear_2, 'w')
#     right_ear_3 = byte_output_dir + "\\" + "right_ear_3.csv";
#     right_ear_3 = open(right_ear_3, 'w')
# =============================================================================
    for line in lines:
        line_split = line.split(" ");
        image_name = file_name_from_path(line_split[len(line_split) - 1]) + ".png";
        print("Converting into bit map " + image_name)
        image = image_dir + "\\" + image_name;
        im = Image.open(image);
        pixels = list(im.getdata())
        pixels = np.array(pixels).reshape((64, 64))
        #byte_output.write(','.join(map(str, pixels)))
        pixels = scale_down_pixel(pixels)
        for i in range(0,16):
            byte_output.write(",".join(map(str,pixels[i])))
            if (i < 15):
                byte_output.write(",")
        byte_output.write("\n")
    byte_output.close()
    
    
def scale_down_pixel(pixels):
    scaled_down_pixel = np.zeros((16,16));
    for i in range(0, 16):
        for j in range(0, 16):
            val = 0;
            x = i*4;
            y = j*4;
            for z in range(0,4):
                for k in range(0,4):
                    if pixels[x + z][y + k] == 255:
                        val = val + 1;
            scaled_down_pixel[i][j] = val;
    return scaled_down_pixel;
    
def scale_annotations(annotation_data_file, scaled_annotation_data_file, raw_images_dir):  
    annot_file = open(annotation_data_file, "r");
    lines = annot_file.readlines();
    scale_size = 64;
    file = open(scaled_annotation_data_file, 'w');
    for l in lines:
        line_split = l.split(" ");
        print("scaling " + line_split[len(line_split) - 1]);
        image_name = file_name_from_path(line_split[len(line_split) - 1]) + ".png";
        image = raw_images_dir + "\\" + image_name;
        im = Image.open(image);
        dim = im.size;
        for i in range(1, 19):
            if i % 2 != 0:
                line_split[i] = int((scale_size/dim[0])*int(line_split[i]));
            else:
                line_split[i] = int((scale_size/dim[1])*int(line_split[i]));
        file.write(" ".join(map(str,line_split)));
    file.close();
    
def generate_test_out(scaled_annotation_data_file, test_out_put):
    scaled_annotation_data_file = open(scaled_annotation_data_file, "r");
    test_out_put = open(test_out_put, "w");
    lines = scaled_annotation_data_file.readlines();
    for line in lines:
        line_split = line.split(" ");
        for i in range(1, 19):
            test_out_put.write(line_split[i]);
            if i < 18:
                test_out_put.write(",");
        test_out_put.write("\n");
    test_out_put.close();
    
def read_file(input_file, values_file):
    input_file_data = pd.read_csv(input_file, header=None)
    value_file_data = pd.read_csv(values_file, header=None)
    data = input_file_data.iloc[:].values
    points = value_file_data.iloc[:].values
    return data[0:1000, :],points[0:1000, 0:6] 


def encode_data(train_points, test_points):
    encoded_train_points = np_utils.to_categorical(train_points) 
    encoded_test_points = np_utils.to_categorical(test_points)
    return encoded_train_points,encoded_test_points


def create_convo():
    trainX, trainy = read_file(black_and_white_byte_arr_file + "\\byte_output.csv",test_out_put) 
    testX, testy = read_file(black_and_white_byte_arr_file + "\\byte_output.csv",test_out_put)
    model = Sequential()
    model.add(Dense(200, input_dim=256, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, validation_data=(testX, testy), epochs=350, batch_size=10, verbose=1)
    output_values = model.predict(testX[:500, :]);
    print(output_values[:10, :]);
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    
def load_convo():
    filename = 'finalized_model.sav'
    testX, testy = read_file(black_and_white_byte_arr_file + "\\byte_output.csv",test_out_put)
    loaded_model = pickle.load(open(filename, 'rb'))
    output_values = loaded_model.predict(testX);
    scaled_annot_val = open(scaled_annotation_data_file, 'r');
    annot_lines = scaled_annot_val.readlines();
    index = 0;
    for annot_line in annot_lines:
        temp_line = annot_line.strip();
        line_split = temp_line.split(" ");
        file_name = line_split[len(line_split) - 1];
        temp = file_name.split(".")[0] + ".png";
        file_name = temp;
        print("Annotating " + file_name)
        lines = [];
        image_full_Path = resized_images_dir + "\\" + file_name;
        original_image_path = resized_images_dir + "\\" + file_name;
        dim = Image.open(original_image_path).size;
        points_list = output_values[index];
        x = dim[0]/2;
        y = dim[1]/2;
        eyes_and_mouth = [(int(points_list[0])*(x/32), int(points_list[1])*(y/32)),(int(points_list[2])*(x/32), int(points_list[3])*(y/32)), (int(points_list[4])*(x/32), int(points_list[5])*(y/32)), (int(points_list[0])*(x/32), int(points_list[1])*(y/32))];
        #left_ear = [(int(points_list[6])*(x/32), int(points_list[7])*(y/32)),(int(points_list[8])*(x/32), int(points_list[9])*(y/32)),(int(points_list[10])*(x/32), int(points_list[11])*(y/32)),(int(points_list[6])*(x/32), int(points_list[7])*(y/32))];
        #right_ear = [(int(points_list[12])*(x/32), int(points_list[13])*(y/32)),(int(points_list[14])*(x/32), int(points_list[15])*(y/32)),(int(points_list[16])*(x/32), int(points_list[17])*(y/32)),(int(points_list[12])*(x/32), int(points_list[13])*(y/32))];
        lines.append(eyes_and_mouth);
        #lines.append(left_ear);
        #lines.append(right_ear);
        annotate_image_based_on_points(image_full_Path, file_name, lines, annotated_Images_original_size)
        ++index;
        
    
                
annotation_data_file = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotations_data.ssv";
annotated_images_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotated Images\\";
raw_images_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\images";
resized_images_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\resized\\";               
black_and_white_image_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\black and white\\";
original_size_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\images\\";
color_byte_arr_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\byte arr\\";
black_and_white_byte_arr_file = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\black and white bitmap\\";
scaled_annotation_data_file = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\scaled_annotation_data_file.ssv";
test_out_put = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\test_out_put.csv";
annotated_Images_original_size = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotated Images original size\\";
# append_multiple_files_into_one(list_files("A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotations data"), "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotations_data.ssv");    
# resize_images_inside_directory(raw_images_dir, resized_images_dir, black_and_white_image_dir);
#draw_annotations_on_images(annotation_data_file, resized_images_dir, annotated_images_dir, original_size_dir)
# scale_annotations(annotation_data_file, scaled_annotation_data_file, raw_images_dir);
# convert_into_byte_arr(scaled_annotation_data_file, black_and_white_image_dir, black_and_white_byte_arr_file);
# generate_test_out(scaled_annotation_data_file, test_out_put);


# create_convo();        
load_convo();