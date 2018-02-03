# Object_Detection
API Prototyping 
Object Detection API
Prototyping

In this document I will cover two parts first how to add new images to training data and second how to train different models


How to add new images:

Directory Structure

-data/
--test_labels.csv
--train_labels.csv
-annotation/
--xml_files
-images/
--test/
---testingimages.jpg
--train/
---testingimages.jpg
--...yourimages.jpg
-training
-xml_to_csv.py

Generate xml files for new images(labelling)
Run xml_to_csv.py in respective folders(string formatting returned null csv’s hence this modification) and move to data/*.csv
Move all annotations(xml files) to a new dir
Generate TFR(Create a copy of object detection to working file as we need to import) steps:

git clone https://github.com/tensorflow/models.git

Then, following the installation instructions:
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib
And then:
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.

Download the python version, extract, navigate into the directory and then do:
sudo ./configure
sudo make check
sudo make install
After that, try the protoc command again (again, make sure you are issuing this from the models dir).
and
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
install the object_dection library formally by doing the following from the models directory:
sudo python3 setup.py install
Now we can run the generate_tfrecord.py script. We will run it twice, once for the train TFRecord and once for the test TFRecord.
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
Now, in your data directory, you should have train.record and test.record.

-data/
--test_labels.csv
--train_labels.csv
--train.record
--test.record

Now copy data, images and training dir to models/research/object_detection and follow training steps below. 


Train Different Models:

Download checkpoint file(here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) of the model you wish to try out and extract in models/research/object_detection

Find the config file in models/research/object_detection/samples and save it to models/research/object_detection/training make changes in 
Fine_tune_checkpoint: "name_of_model_dir/model.ckpt"
Example: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"

 And 

train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"
  }
  label_map_path: "data/cycle.pbtxt"
}

eval_config: {
  num_examples: 40
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"
  }
  label_map_path: "training/cycle.pbtxt"
  shuffle: false
  num_readers: 1 #only edit the bold lines
}

Your training folder should consist of config file and our cycle.pbtxt


Once this is done from models/research run:

protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Then cd to /object_detection and run:

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/name_of_our_config_file.config 

Example: ssd_mobilenet_v1_pets.config

This will start the training process, run:

Check progress with tensorboard --logdir='training'

Once trained we need our models inference graph run:

python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/name_of_our_config_file.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory bike_inference_graph


jupyter notebook

# What model to download.
MODEL_NAME = ‘bike_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', cycle.pbtxt')

NUM_CLASSES = 1

Add images to test_images and run all cells 
 
And you should get the output :)

