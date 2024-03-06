# Age-Gender-Detection

## Age and Gender Estimation using OpenCV

This Python script utilizes OpenCV to perform age and gender estimation on faces detected in an image. It loads pre-trained deep learning models for face detection, age estimation, and gender classification.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib

You can install the required packages using pip:

```bash
pip install opencv-python matplotlib
```

## Usage
1. Clone or download this repository to your local machine.

2. Place your image in the same directory as the script, or specify the path to the image in the code.

3. Ensure you have the following model files in the same directory as the script:

  * opencv_face_detector.pbtxt
  * opencv_face_detector_uint8.pb
  * age_deploy.prototxt
  * age_net.caffemodel
  * gender_deploy.prototxt
  * gender_net.caffemodel

4. Run the script:
```bash
python AgeGenderDetector.py
```
* The script will detect faces in the image, estimate their age and gender, and annotate the image with the results. The annotated image will be displayed using Matplotlib.


## Acknowledgments
* This script utilizes pre-trained models for face detection, age estimation, and gender classification provided by OpenCV.
* The face detection model is based on Single Shot MultiBox Detector (SSD) framework with a MobileNet backbone.
* The age estimation and gender classification models are based on Convolutional Neural Networks (CNNs).

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License.

```csharp
This README provides clear instructions on how to use the program, lists the requirements, acknowledges the sources of pre-trained models, encourages contributions, and includes licensing information. You can add more details or sections as needed.
```
