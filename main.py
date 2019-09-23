# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:02:15 2019

@author: umesh
"""


from process.CatFaceDetector import Deep_Learning_Model as Cat_Face


raw_images_dir = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\images";
annotation_data_file = "A:\\Machine Learning\\cat dataset\\cat-dataset\\processed\\annotations_data.ssv";
model_root = "A:\Machine Learning\cat dataset\cat-dataset\processed\model"

machine_model = Cat_Face(raw_images_dir, annotation_data_file, model_root_dir=model_root,convert_image=-1,limit=1000);

# machine_model.generate_rescaled_annotation_file()
# machine_model.rescale_images()
# machine_model.draw_annotations_on_all_images(converted=False)

# train_and_test_data = machine_model.get_train_and_test()
machine_model.set_up_model()
# machine_model.read_predictions_and_annotation(original_size=False)