<!-- Banner -->
<div align="center">
  <p>
    <a href="https://yolovision.ultralytics.com/" target="_blank">
      <img width="100%" src="readme_assets/banner.png"></a>
  </p>

</div>


<!-- Technologies used, right under banner -->
<div align="center">
    <img src="https://github.com/devicons/devicon/blob/master/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="50" height="50"/>
    <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" alt="OpenCV" width="50" height="50"/>
    <img src="readme_assets/yolo_logo.png" width="75" height="50" alt="YOLO"/>
 </div>

<!-- Project description -->
***
<div align="center">
<h1>Traffic signs recognition</h1>
<p >Hello, this is my personal project for traffic sign recognition. "Self-driving 
cars" has 
been a 
hot topic for a lot of years. With the rise in popularity of electric vehicles we're starting to 
see more and more autonomous (or smart?) features ranging from self-parking, lane assist to 
fully autonomous driving, like the Tesla Pilot.
</p>
<p>Recognizing traffic signs, it's essential for autonomous driving, at least for now. With this 
project I tried to train a model that can recognize most common traffic signs, or groups of 
traffic signs. </p>
</div>


<!-- Project goals -->
<div align="left">
<h1 align="center">Goals</h1>
<p>
I started by playing around with different models and datasets and when I realised that I want 
to take this further I set some goals for myself. 
</p>
<ul>
<li>Create a dataset that contains of somewhat big and high quality images.</li>
<li>Train a model that has pretty high accuracy and can reliably detect traffic signs.</li>
<li>Use the model with high accuracy to collect more data, thus saving a lot of time.</li>
</ul>
<p>
I had a bunch of other ideas,
but this was a solid base that would eventually allow me to build upon.
</p>
</div>
<!-- Dataset -->
<div align="left">
<h1 align="center">Dataset</h1>
<a href="https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-detection-europe">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
<a href="https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-hyipi">
  <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
<p>There are a lot of free datasets for this particular task but there are some recurring issues 
with them.</p>
<ul>
<li>Isolated images: most datasets contain images with one traffic sign per image. This does 
not reflect reality.</li>
<li>No bounding boxes: some datasets had thousands of images, but they were meant for image 
classifications. This would require a lot of work to label the images.</li>
<li>Small images: 50x50px for an image is rather small </li>
</ul>

<p>With these in mind, I created my own dataset. I tried a bunch of different sources and 
ultimatelly I achieved my goal of automatically collecting data with a trained model (collected 150-200 images this way).
</p>
<p></p>
</div>

<!-- Experiments -->
<div align="left">
<h1 align="center">Experiments</h1>
<p>Training and testing a model is where the fun actually begins in machine learning. 
Collecting, 
cleaning and labeling data is entertaining, but there's nothing like training a model for 8 hours 
just to realize that it's bad and it's not going to get better.

</p> 

### TensorFlow

I started with some TensorFlow models, from TF model zoo. TF provides a wide range of
architectures, from very fast to very performant. Starting with a model that trains very fast helped
me understand if the dataset is good or not. I got pretty good results from R-CNN's and
MobileNet's.
<br>

### YOLO

The next step was YOLO. In my opinion, this is one of the best architectures right now.
It trains decently quick, has good framerates and you can find a heavily pretrained model very
easily.

### Augmentation

Generating synthetic images is always a possibility. It's an excellent way to make more out of
your data, but there needs to be a balance between authentic and synthetic images. RoboFlow's
built in augmentation system is easy to use and can generate a lot of data very fast. I also
tried some libraries such as 'imgaug'.<br/>
The synthetic images really helped, they increased the accuracy and removed a lot of false
positive detections.

### Changing the dataset

I decided to go from detecting unique traffic signs to detecting broader groups. This was really
easy due to the naming scheme that I chose for the classes. First I started with the four classes
of signs that I had (mandatory, priority, forbidden, informational) and then I tried with a
single class for all the signs. The results were good, the underrepresented classes were
detected more often. <br/>
These methods are good for automatic data collection, especially if you plan on manually
checking the data later.

</div>

<!-- Results -->
<div align="left">
<h1 align="center">Results</h1>
<p> COMING SOON
</p>
</div>

<!-- Documentation -->

