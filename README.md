# AI-Smart-Surveillance-System

AI-powered smart surveillance system developed as my Final Year Graduation Project.

The project focuses on improving traditional surveillance systems by adding real-time video monitoring, face recognition, video recording, and a web dashboard for managing users and recordings.

## Features

- Real-time video monitoring through a Flask web dashboard.
- Face detection and recognition using computer vision.
- Face registration for new users.
- Video recording from the live camera feed.
- Recording history with play, download, delete, and search options.
- Admin dashboard for user management.
- Role-based access for Admin, Manager, and Employee users.

## Technologies Used

- Python
- Flask
- OpenCV
- MTCNN
- FaceNet
- PyTorch
- HTML, CSS, JavaScript
- CSV-based local storage

## Results

- Achieved **93% face recognition accuracy** during project testing.
- Achieved **89% anomaly detection accuracy** in the full graduation project evaluation.
- Generated alerts with an average response time of **1.4 seconds**.
- Demo face recognition result shows **99.0% confidence** for a sample recognized user.
- Video recording is configured at **640×480 resolution** with **15 FPS**.
- Recording history demo shows **1 saved recording** with a total size of **5.1 MB**.
- Built **6 main modules**: Login, Live Dashboard, Face Recognition, Face Registration, User Management, and Recording History.

## Screenshots

### Live Dashboard
<p align="center">
  <img src="Images/Admin%20Main%20Dashboard.png" width="650">
</p>

### Face Recognition Result
<p align="center">
  <img src="Images/The%20result.png" width="280">
</p>

### Video Recording Functionality
<p align="center">
  <img src="Images/Video%20Recording%20Functionality.png" width="650">
</p>

### Recording History
<p align="center">
  <img src="Images/recording-history.png" width="650">
</p>

### User Management
<p align="center">
  <img src="Images/User%20Management%20%28Admin%29.png" width="650">
</p>

### Login Interface
<p align="center">
  <img src="Images/Login%20Interface.png" width="420">
</p>

## Project Structure

```text
AI-Smart-Surveillance-System/
├── app.py
├── intelligent_surveillance.py
├── requirements.txt
├── templates/
├── static/
├── Images/
└── README.md
```
