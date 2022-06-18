## About file directions
This training setting is the proposed approach without using dynamic memory module. The amount of memory data allocated for each previous task is the same.
Similar to D-GSM-training, you need to create folder "checkpoint" with sub-folders named "social-stgcnn-XX" to contain trained models. Under default settings, "XX" includes "MA", "FT", "ZS", "EP", and "SR".
If you want to set your own directions, please change the directions in the python files, too.

## Running
You can use .sh file to run the code.
Key changable parameters can be altered in .sh file.

To continually train the models, please save and loaded trained-model in observed scenarios when starting a training with the new scenario.
The code for model loading is also provided in the script. You may need to change the file name and directions when using your own models.
