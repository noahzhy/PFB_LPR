# Related frame based license plate recognition

This is a project for related frame based license plate recognition.

- [Related frame based license plate recognition](#related-frame-based-license-plate-recognition)
    - [System Consist](#system-consist)
    - [System Architecture](#system-architecture)
    - [License](#license)

## System Consist

The system consists of two parts:

1. [License Plate Detection](#license-plate-detection)
2. [License Plate Recognition](#license-plate-recognition)

### License plate detection

TBD

### License plate recognition

The method of license plate recognition is based on related frames (previous frame) to improve the accuracy of recognition. Noticing that the license plate is not moving in the video, we can use the previous frame to help the current frame to recognize the license plate that prevent the blur and nosie of the some frames.

Key points:

1. There are two kinds of license plate in South Korea, one is single line and the other is double line. To distinguish them is the first step of recognition. The key point is get the divider line of the double line license plate.

2. The situation of license plate is not always the same, it may be not exist in the previous frame. In addition, the license plate in the previous frame and the current frame may be not the same one.

## System Architecture

## License

BSD 3-Clause License (see LICENSE file)
