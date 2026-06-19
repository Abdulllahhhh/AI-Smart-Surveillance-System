# AI-Smart-Surveillance-System

AI-powered smart surveillance system developed as my **Final Year Graduation Project**.

The system is designed to support smart security monitoring by combining real-time video streaming, face recognition, video recording, recording history management, and a web dashboard. The main idea of the project is to make surveillance systems smarter by allowing the system to detect and recognize registered users through a camera feed and manage recorded surveillance videos from one dashboard.

This repository is uploaded as a **portfolio showcase**. It includes selected source code, project interface screenshots, and demo results. Private user data, face data, passwords, and recorded videos are not included.

---

## Project Overview

Traditional surveillance systems usually depend on manual monitoring. This project improves that idea by adding AI-based face recognition and a simple web dashboard that allows users to:

- View live camera footage.
- Register new faces.
- Recognize registered users.
- Display recognition confidence.
- Start and stop video recording.
- View saved recording history.
- Manage users through an admin dashboard.

The project was built using Python, Flask, OpenCV, MTCNN, and FaceNet.

---

## Main Features

- Real-time video monitoring through a web dashboard.
- Face detection using MTCNN.
- Face recognition using FaceNet embeddings.
- Face registration for new users.
- Recognition result with user name, ID, confidence percentage, and time.
- Video recording from the live camera feed.
- Recording history page with play, download, delete, and search options.
- Admin user management dashboard.
- Role-based access for Admin, Manager, and Employee users.
- Clean web interface using HTML, CSS, and JavaScript.

---

## Technologies Used

- Python
- Flask
- OpenCV
- MTCNN
- FaceNet
- PyTorch
- NumPy
- Pandas
- HTML
- CSS
- JavaScript
- CSV-based local storage

---

## Results & Metrics

- Built **6 main system modules**: Login Interface, Live Dashboard, Face Recognition, Face Registration, User Management, and Recording History.
- Implemented real-time video monitoring through a Flask web dashboard.
- Face recognition result displays **4 key values**: user name, user ID, confidence percentage, and recognition time.
- Achieved **99.0% confidence** in the sample face recognition demo result.
- Video recording is configured at **640×480 resolution** with **15 FPS**.
- Recording module supports **3 video codec options** for compatibility: H.264, MPEG-4, and XVID.
- Recording history dashboard displays **3 main statistics**: total recordings, total storage size, and oldest recording date.
- Demo recording history shows **1 saved recording** with a total size of **5.1 MB**.
- User management supports **3 user roles**: Admin, Manager, and Employee.
- Repository includes **6 project screenshots** to demonstrate the system interface and main features.

> Note: The 99.0% value represents the confidence score from the shown demo result, not a full dataset accuracy benchmark.

---

## Project Structure

```text
AI-Smart-Surveillance-System/
│
├── app.py
├── intelligent_surveillance.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── users.html
│   └── view_history_recording.html
│
├── static/
│   ├── css/
│   └── images/
│
└── Images/
    ├── Login Interface.png
    ├── Admin Main Dashboard.png
    ├── The result.png
    ├── User Management (Admin).png
    ├── Video Recording Functionality.png
    └── recording-history.png
```

---

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

---

## How the System Works

1. The user logs in through the web interface.
2. The live dashboard displays the camera feed.
3. The system detects faces from the camera frame.
4. FaceNet generates an embedding for the detected face.
5. The system compares the embedding with registered face data.
6. If a match is found, the system displays the user information and confidence score.
7. The user can start video recording from the dashboard.
8. Saved recordings can be viewed and managed from the Recording History page.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Abdullahhhh/AI-Smart-Surveillance-System.git
cd AI-Smart-Surveillance-System
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

Windows:

```bash
set SECRET_KEY=dev-secret-key
set DEFAULT_ADMIN_PASSWORD=change-me
```

Linux / macOS:

```bash
export SECRET_KEY=dev-secret-key
export DEFAULT_ADMIN_PASSWORD=change-me
```

4. Run the Flask application:

```bash
python app.py
```

5. Open the system in the browser:

```text
http://127.0.0.1:5000
```

---

## Important Note About Running the Project

This repository is mainly prepared as a portfolio version of my graduation project. Some runtime files are intentionally not included, such as user records, face embeddings, and recorded videos.

To fully run the system locally, demo data files may need to be created locally, such as:

```text
users.csv
face_data.csv
recordings/
```

These files are excluded from the public repository to protect privacy and avoid uploading sensitive data.

---

## Privacy & Security Notes

This repository does not include private or sensitive files such as:

- User database files
- Face embedding data
- Password files
- Recorded surveillance videos
- Real face images
- Environment files
- Local development folders

The following files and folders are intentionally excluded using `.gitignore`:

```text
users.csv
face_data.csv
recordings/
password*
users/
face_data/
.env
__pycache__/
.idea/
venv/
```

---

## Portfolio Note

This project was developed as my **Final Year Graduation Project** and uploaded to GitHub to showcase the main idea, selected implementation, user interface, AI functionality, and project results.

The repository does not contain the full private deployment data. It is intended for portfolio and CV purposes.
