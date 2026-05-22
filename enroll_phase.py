import cv2
import numpy as np 
import face_recognition
import mysql.connector
import os 
import pickle

mydb = mysql.connector.connect(

    host = "localhost",
    user = "root",
    password = "ahmed1234",
    database = "Family_Data_OpenCV"
    
)

cursor = mydb.cursor()

ENCODINGS_FILE = "encodings.pkl"

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    print(f"Loaded {len(known_encodings)} existing encodings.")

else: 

    known_encodings = []
    known_names =[]

    cursor.execute("SELECT * FROM members")
    members = cursor.fetchall()

    prompts = ["Look straight", "Look left", "Look right", "Smile", "Look up"]

    def capture_prompt(text, cap):

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.putText(frame, f"follow the prompts {text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                return frame
            
    for member in members:
        
        name = member[1]

        if name in known_names:
            print(f"{name} already enrolled, skipping...")
            continue

        print(f"\nEnrolling: {name} - Please sit in front of the camera")
        cap = cv2.VideoCapture(0)
        enrolled_count = 0

        for prompt in prompts:
            frame  = capture_prompt(prompt, cap)

            if frame is None:        
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_enc = face_recognition.face_encodings(rgb_frame)

            if frame_enc:
                known_encodings.append(frame_enc[0])
                known_names.append(name)
                enrolled_count += 1
                print(f"  Captured {prompt}")
            else:
                print(f"  No face detected for {prompt}, skipping")

        cap.release()
        cv2.destroyAllWindows()
        print(f"  Done! {enrolled_count}/5 shots captured for {name}")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"\nSaved! Total encodings: {len(known_encodings)} for {len(set(known_names))} people.")