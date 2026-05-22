#  Family Face Recognition System

A real-time face recognition system built with Python and OpenCV that identifies family members using webcam input and EID documents stored in a MySQL database.

---

## ✨ Features

-  Real-time face detection and recognition via webcam
-  Enrolls members using their EID document (PDF)
-  Multi-angle webcam enrollment (5 prompts: straight, left, right, smile, up)
-  Saves and loads face encodings from a local file for fast startup
-  MySQL database integration for member management
-  Simple terminal menu to navigate between modes

---

##  Tech Stack

- Python 3
- OpenCV
- face_recognition (dlib)
- MySQL + mysql-connector-python
- pdf2image + Poppler
- python-dotenv

---

## Project Structure

```
face-recognition/
│
├── src/
│   └── main.py               # Main application
├── images/                   # EID/passport PDFs (not uploaded)
├── encodings.pkl             # Auto-generated face encodings cache
├── .env                      # Your local DB credentials (not uploaded)
├── .env.example              # Template for credentials
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/family-face-recognition.git
cd family-face-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `face_recognition` requires `dlib` which needs CMake. On Mac: `brew install cmake`. On Windows: install Visual Studio Build Tools.

### 3. Set up your `.env` file
```bash
cp .env.example .env
```
Then fill in your MySQL credentials in `.env`.

### 4. Set up MySQL
Create a database called `Family_Data_OpenCV` and a `members` table with your family data.

### 5. Run
```bash
python src/main.py
```

---

##  Menu Options

```
==========================================
      FAMILY FACE RECOGNITION SYSTEM
==========================================
  1. Skip enrollment / Start live detection
  2. Add new member
  3. Enroll existing database members
  4. Exit
==========================================
```

---

##  Security Note

Never commit your `.env` file. It is listed in `.gitignore` by default.

---

## 👤 Author

**Ahmed Lateef**  
AI Engineering Student @ Abu Dhabi University  
AI Intern @ FirstFintech Corporation  
[LinkedIn](https://www.linkedin.com/in/ahmed-lateef-5a59b93b1)
