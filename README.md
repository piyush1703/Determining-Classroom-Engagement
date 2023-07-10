# Determining-Classroom-Engagement
## ➢ AIM of the Study<br>
• The aim of this study is to determine the classroom engagement of 
students with the help of computer vision and audio analysis techniques.<br>
• Traditional methods of measuring classroom engagement often rely 
on subjective observations or self-reporting by students, which may 
introduce biases and inaccuracies.<br>
• By using computer vision techniques (like YOLOv5), we can assess 
students' attention and engagement by analyzing their visual focus 
towards the projector and by conducting audio analysis, we can 
identify specific audio components that may induce sleep or promote 
engagement among students.<br>
## ➢ Dataset
• The dataset consists of around 700 raw back camera images from 
classroom and two videos of around 20 mins each out of which 
frames are also sampled and used in training dataset images.<br>
• For audio analysis audios are extracted from above two videos and 
used as testing dataset for audio analysis and for training dataset 
audios are extracted YouTube videos of around 100 mins for each 
class.<br>
• Out of all these images 136 images is manually annotated on 
roboflow and preprocessing steps like image augmentation is done 
to increase annotated dataset size.<br>
• Train/validation/test split is Train[118 images (87%)], 
Validation[ 11 images(8%) ] and Test[7 images (5%)].<br>
## ➢ Training Models and Analysis
• Custom YOLOv5 model is trained on the preprocessed dataset for 150 
epochs with batch-size of 16. Then all the labels from detected frames 
are stored in txt file with minimum confidence threshold of 25% which is 
used for analyzing engagement of students by plotting graph for each 
student of probability with which they are seeing towards projector per 
frame.<br>
• For getting the probabilities of each student per frame, each student is 
assigned unique id according to its bounding box coordinates present in 
space as students position almost remain same during the lecture.
Engagement score is calculated my taking mean over all frame scores, 
frame scores is calculated by adding the probabilities of students which 
are detected looking at the projector divided by maximum frame score 
equal to number of students present in frame.<br>
• For audio analysis part 100 mins audio for each class (sleep inducing and 
engaging) is preprocessed by splitting it into 5 mins segments and MFCC 
features are extracted using librosa library for each 5 min segments and 
stored in array along with its class which is used as training data for the 
CNN model. Splitting the audio into 5-minute lengths allows for a more 
granular analysis of engagement over time, we can capture variations in 
engagement levels within different sections of the audio. This helps in 
understanding how engagement evolves throughout the duration of the 
audio and identifying specific time intervals that are more engaging or 
sleep-inducing.<br>
• This prepared dataset is trained using sequential CNN model which is 5 
layers model, first layer is 1D convolutional layer with a 32 filters of 
kernel size 3 with Relu-activation, the output obtained from this layer is 
passed on to MaxPooling layer with pool size 2 which reduces spatial 
dimension of the input, then this input is flattened using flattened layer 
so that it can be fed to fully connected layer, then output from fully 
connected layer is input for 64 neuron layer with relu-activation, then 
output from this layer is passed on to last layer with 2 neurons with 
softmax activation, in these two neurons predicted probabilities for each 
class is stored.<br>

## ![image](https://github.com/piyush1703/Determining-Classroom-Engagement/assets/97897537/2d92e6fe-a326-4e78-bb5f-8a958fd7f649)



