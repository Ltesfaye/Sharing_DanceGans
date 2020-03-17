# danceGANs

## Dependencies

For running this code and reproducing the results, you need the following packages. Python 2.7 has been used.

Packages:

- TensorFlow
- NumPy
- cv2
- scikit-video
- scikit-image

## How to Run

Place the videos inside a folder called "trainvideos".
Run main.py with the required values for each flag variable (set in main.py).

If the number of frames/video-size flag is changed, modify it directly in the parameter. Then modify the read-and-process-videos function in utils.py to pick the correct number of frames from the read video.
