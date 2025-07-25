import numpy as np

# From pyrealsense2
# Command: rs-enumerate-devices -c
# 640x480
CAMERA_MATRICES = {
    # "cam_1": np.array([[604.97, 0, 314.83], [0.0, 604.79, 249.03], [0, 0, 1]]),
    # "cam_2": np.array([[609.41, 0, 314.85], [0.0, 609.65, 240.52], [0, 0, 1]]),
    "cam_4": np.array([[606.82, 0, 328.42], [0.0, 606.53, 244.06], [0, 0, 1]]),
    # "cam_5": np.array([[607.33, 0, 316.14], [0.0, 606.80, 247.48], [0, 0, 1]]),
    "cam_6": np.array([[606.71, 0, 311.34], [0.0, 606.45, 245.18], [0, 0, 1]]),
}
# 2： Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
#   Width:      	640
#   Height:     	480
#   PPX:        	314.846893310547
#   PPY:        	240.515563964844
#   Fx:         	609.406127929688
#   Fy:         	609.652893066406
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	55.41 x 42.98

# 4:  Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
#   Width:      	640
#   Height:     	480
#   PPX:        	328.422546386719
#   PPY:        	244.061157226562
#   Fx:         	606.820434570312
#   Fy:         	606.532897949219
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	55.6 x 43.17

# 5：  Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
#   Width:      	640
#   Height:     	480
#   PPX:        	316.141754150391
#   PPY:        	247.484756469727
#   Fx:         	607.333190917969
#   Fy:         	606.8017578125
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	55.57 x 43.15

# 6:  Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
#   Width:      	640
#   Height:     	480
#   PPX:        	311.34228515625
#   PPY:        	245.182678222656
#   Fx:         	606.710632324219
#   Fy:         	606.452575683594
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	55.61 x 43.18


DISTORTION_COEFFICIENTS = {
    # "cam_1": np.zeros((5)),
    # "cam_2": np.zeros((5)),
    "cam_4": np.zeros((5)),
    # "cam_5": np.zeros((5)),
    "cam_6": np.zeros((5)),
}
