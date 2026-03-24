# AI-Based Face Tracking with Tello Drone

Real-time autonomous face tracking using computer vision and drone control.

---

## Hardware

![Tello Drone](screenshots/tello.drone.jpeg)

---

## Demo (Face Tracking)

![Face Tracking Demo](screenshots/demo.jpeg)

---

## Features

- Real-time video streaming from Tello drone  
- Face detection using OpenCV Haar Cascade  
- Automatic yaw control to keep the face centered  
- Forward/backward movement based on face size (distance control)  
- Optional vertical tracking (height adjustment)  
- Live debug information (battery, face position, control signals)  
- Manual control script included  
- Safe landing and clean shutdown  

---

## How It Works

The system follows a perception → control loop:

1. The drone streams video to the laptop  
2. OpenCV detects faces using Haar Cascade  
3. The largest detected face is selected  
4. The position of the face is compared to the image center  
5. Control commands are sent to the drone:
   - Rotate left/right (yaw)
   - Move forward/backward (distance)
   - Adjust height (optional)

---

## Technologies Used

- Python  
- OpenCV  
- NumPy  
- djitellopy  

---


