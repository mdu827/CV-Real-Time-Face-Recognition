# CV-Real-Time-Face-Recognition
Face recognition YOLOv8 model trained on [FDDB dataset](https://universe.roboflow.com/fddb/face-detection-40nq0/dataset/1). Provides facial detection and segmentation on videos/images/live camera recording(video stream).

### Installation
```
git clone https://github.com/mdu827/CV-Real-Time-Face-Recognition.git
pip install -r requirements.txt
```

### Usage
Video stream from live camera:
```
python3 main.py --mode image --source path/to/image.jpg
```
Images processing:
```
python3 main.py --mode image --source path/to/image.jpg
```
Video processing:
```
python3 main.py --mode video --source path/to/video.mp4
```
<p float="left">
  <img src="img6.jpg" width="300"/>
  <img src="img7.jpg" width="300"/>
  <img src="img6p.jpg" width="300"/>
  <img src="img7p.jpg" width="300"/>
</p>
