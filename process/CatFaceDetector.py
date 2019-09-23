# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:14:32 2019

@author: umesh
"""
import os
from PIL import Image, ImageDraw
import random 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D, AveragePooling2D
import pickle
from pathlib import Path

class Deep_Learning_Model():
    
    path_separator = os.path.sep
    
    def __init__(self, images_dir, original_annotation_file, debug=False, convert_image=-1, scale=(64,64), 
                 model_root_dir=os.getcwd() + path_separator + "model", train_to_test_ratio=0.85,
                 model_file_name="deep_learn_cat_face.mod", limit=10000):
         print("Initializing the model") 
         self.debug = debug
         self.original_images_dir = images_dir+ self.path_separator
         self.convert_image = convert_image
         self.scale=scale
         self.model_root_dir=model_root_dir + self.path_separator
         self.train_to_test_ratio=train_to_test_ratio
         self.original_annotation_file=original_annotation_file
         self.rescaled_images_dir=self.model_root_dir + "rescaled_images" + self.path_separator
         self.rescaled_converted_images_dir=self.model_root_dir + "rescaled_converted_images_dir" + self.path_separator
         self.rescaled_annotation_data_file=self.model_root_dir + "rescaled_annotations_file.ssv"
         self.limit = limit
         self.create_directories()
         print("finished generating the model") 
    
    def create_directories(self):
        dirs = [self.model_root_dir, self.rescaled_images_dir, self.rescaled_converted_images_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        
        
    def generate_rescaled_annotation_file(self):
        print("Generating rescaled annotation file")
        annot_file = open(self.original_annotation_file, "r")
        lines = annot_file.readlines()
        file = open(self.rescaled_annotation_data_file, 'w')
        for l in lines:
            line_split = l.split(" ")
            print("scaling " + line_split[len(line_split) - 1])
            image_name = self.file_name_from_path(line_split[len(line_split) - 1], self.path_separator) + ".png"
            image = self.original_images_dir + self.path_separator + image_name
            im = Image.open(image)
            dim = im.size
            for i in range(1, 19):
                if i % 2 != 0:
                    line_split[i] = int((self.scale[0]/dim[0])*int(line_split[i]))
                else:
                    line_split[i] = int((self.scale[1]/dim[1])*int(line_split[i]))
            file.write(" ".join(map(str,line_split)))
        file.close()
        annot_file.close()
        print("Finished rescaled annotation file")
    
    @staticmethod
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
    
    def rescale_images(self):
        print("Generating rescaled images")
        annot_file = open(self.original_annotation_file, "r")
        lines = annot_file.readlines()
        for line in lines:  
            line = line.strip()
            line_split = line.split(" ")
            image_name = self.file_name_from_path(line_split[len(line_split) - 1], self.path_separator) + ".png"
            image_full_path = self.original_images_dir + self.path_separator + image_name
            img = Image.open(image_full_path)
            new_width  = self.scale[0]
            new_height = self.scale[1]
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save(self.rescaled_images_dir + image_name);
            if self.convert_image != -1:
                img = img.convert(self.convert_image)
                img.save(self.rescaled_converted_images_dir + image_name);
            img.close();
        print("Finished rescaled images")
     
    def draw_annotations_on_all_images(self, rescaled=True, converted=True):
        print("Generating Annotations on images")
        annot_file = open(self.rescaled_annotation_data_file, "r")
        lines = annot_file.readlines()
        annot_file.close()
        for line in lines:
            line = line.strip()
            line_split = line.split(" ")
            image_name = self.file_name_from_path(line_split[len(line_split) - 1], self.path_separator) + ".png"
            image_full_path = self.rescaled_images_dir + image_name
            line_arr = self.convert_line_into_lines_array(line)
            # print("Annotating " + image_full_path)
            if rescaled:
                self.draw_annotations_on_image(image_full_path, image_full_path, line_arr)
            if converted:
                image_full_path = self.rescaled_converted_images_dir + image_name
                self.draw_annotations_on_image(image_full_path, image_full_path, line_arr)
        print("Finished Annotations on images")
    
    def draw_annotations_on_image(self, given_image, annotated_image, lines, scale_values=False):
        im = Image.open(given_image)
        d = ImageDraw.Draw(im)
        dimensions = im.size
        line_colors = [(0, 0, 255), (178, 255, 102), (102,178,255)]
        for drawline,line_color in zip(lines, line_colors):
            if scale_values:    
                scaled_line = []
                for point in drawline:
                    scaled_line.append((point[0]*dimensions[0]/self.scale[0],point[1]*dimensions[1]/self.scale[1]))
                drawline = scaled_line
            if self.convert_image == '1':
                d.line(drawline, fill=255, width=2)    
            else :
                d.line(drawline, fill=line_color, width=2)
            im.save(annotated_image);
        im.close();
        
    @staticmethod    
    def file_name_from_path(full_path, path_separator):
        dirs = full_path.split(path_separator);
        if len(dirs) == 0: 
            return full_path.split(".")[0];
        full_path_with_ext = dirs[len(dirs) - 1];
        return full_path_with_ext.split(".")[0];
    
    
    def get_train_and_test(self):
        print("Generating test and training data")
        annot_file = open(self.rescaled_annotation_data_file, "r")
        self.trainX = [] 
        self.trainY = []
        self.testX = [] 
        self.testY = []
        self.train_image_name=[]
        self.test_image_name=[]
        count = 0;
        for line in annot_file:
            line = line.strip()
            line_split = line.split(" ")
            image_name = self.file_name_from_path(line_split[len(line_split) - 1], self.path_separator) + ".png"
            image_full_path = self.rescaled_images_dir + image_name
            bitmap = np.array(list(Image.open(image_full_path).getdata())).reshape((self.scale[0], self.scale[0], 3))
            # bitmap = np.array(list(Image.open(image_full_path).getdata()))
            classification = [];
            for i in range(1, 19):
                classification.append(int(line_split[i]))
            random_num = random.random()
            if random_num <= 0.85:
                self.trainX.append(bitmap)
                self.trainY.append(classification)
                self.train_image_name.append(image_name);
            else :
                self.testX.append(bitmap)
                self.testY.append(classification)
                self.test_image_name.append(image_name);
            count = count + 1;
            if count == self.limit:
                break;
        annot_file.close()
        for data in [self.trainX, self.trainY, self.testX, self.testY]:
            data = np.asarray(data)        
        np.save(self.model_root_dir + self.path_separator + "train_input.tra", self.trainX)
        np.save(self.model_root_dir + self.path_separator + "train_output.tra", self.trainY)
        np.save(self.model_root_dir + self.path_separator + "test_input.tes", self.testX)
        np.save(self.model_root_dir + self.path_separator + "test_output.tes", self.testY)
        np.save(self.model_root_dir + self.path_separator + "train_image_name.name", self.train_image_name)
        np.save(self.model_root_dir + self.path_separator + "test_image_name.name", self.test_image_name)
        
        self.model_data = [self.trainX, self.trainY, self.testX, self.testY]
        print("Finished generating test and training data")
        return self.model_data
    
    def set_up_model(self, saved_data=True):   
        print("Generating convo model")
        if saved_data:
            self.load_from_files()
            
        temp_scale = self.scale[0]
        self.trainX = np.array(self.trainX).reshape(self.trainX.shape[0],3,temp_scale,temp_scale)
        self.testX = np.array(self.testX).reshape(self.testX.shape[0],3,temp_scale,temp_scale)
        self.trainY = self.trainY
        self.testY = self.testY
        print(self.trainX.shape)
        print(self.trainY.shape)
        print(self.testX.shape)
        print(self.testY.shape)
        model = Sequential()
        model.add(Convolution2D(int(temp_scale/2),4,strides=4,data_format="channels_first",padding='valid',input_shape=(3, temp_scale, temp_scale)))
        model.add(MaxPooling2D(pool_size = (4, 4)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(16, 4,strides=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.trainY.shape[1], activation='relu'))
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
        self.convo_model = model
        model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), epochs=50, batch_size=50, verbose=1)
        filename = self.model_root_dir + self.path_separator + "convo_model.sav" 
        pickle.dump(model, open(filename, 'wb'))
        output_values = np.asarray(model.predict(self.testX))
        np.save(self.model_root_dir + self.path_separator + "test_predictions.name", output_values)
        self.predict_values = output_values
        print(output_values)
        print("finished generating convo model")
        return model
    
    def load_from_files(self):
        self.testX = np.load(self.model_root_dir + self.path_separator + "test_input.tes.npy")
        self.testY = np.load(self.model_root_dir + self.path_separator + "test_output.tes.npy")
        self.trainX = np.load(self.model_root_dir + self.path_separator + "train_input.tra.npy")
        self.trainY = np.load(self.model_root_dir + self.path_separator + "train_output.tra.npy")        
    
    def train_the_model(self, load_model=True):
        print()
     
    def get_lines_from_prediction_line(self, line):
        index = 0
        lines = []
        for i in range(0,3):
            index = i * 6
            part = [(line[index], line[index + 1]), (line[index + 2], line[index + 3]), (line[index + 4], line[index + 5]),(line[index], line[index + 1])]
            lines.append(part)
        return lines
    
    def read_predictions_and_annotation(self, original_size=False):
        print("Generating annotations on predictions")
        predictions = np.load(self.model_root_dir + self.path_separator + "test_predictions.name.npy")
        annotated_dir = self.model_root_dir + self.path_separator + "predict annotation on images" + self.path_separator
        test_image_names = np.load(self.model_root_dir + self.path_separator + "test_image_name.name.npy")
        if not os.path.exists(annotated_dir):
            os.makedirs(annotated_dir)
        [f.unlink() for f in Path(annotated_dir).glob("*") if f.is_file()] 
        image_base = self.rescaled_images_dir
        if original_size:
            image_base = self.original_images_dir
        for image_name,predict in zip(test_image_names, predictions):
            image_full_path = image_base + image_name
            predict_annot_dir = annotated_dir + image_name
            lines = self.get_lines_from_prediction_line(predict)
            self.draw_annotations_on_image(image_full_path, predict_annot_dir, lines, scale_values=original_size)
        print("finished generating annotations on predictions") 
        
def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()