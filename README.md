---
title: Face Recognition Attendance System
emoji: 👤
colorFrom: blue  
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Face Recognition Attendance System

A comprehensive face recognition system with anti-spoofing for secure attendance tracking.

## Features
- **YOLOv5 Face Detection**: Accurate face detection
- **Anti-Spoofing**: Liveness detection to prevent photo attacks
- **Face Recognition**: dlib-based recognition with 99%+ accuracy
- **Real-time Processing**: Live camera feed with overlay
- **Student & Teacher Portals**: Role-based access
- **Attendance Tracking**: Automated attendance marking
- **Performance Metrics**: Real-time system analytics

## Tech Stack
- Backend: Flask, MongoDB
- ML Models: YOLOv5, ONNX Runtime, dlib
- Frontend: HTML/CSS/JavaScript with Bootstrap
- Computer Vision: OpenCV

## Usage
1. Register as Student/Teacher with face capture
2. Login using face recognition or credentials  
3. Mark attendance with live face verification
4. View attendance records and system metrics
