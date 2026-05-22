import cv2 
import numpy as np
import face_recognition 
import pdf2image 
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "../images/")

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    print(f"Loaded {len(known_encodings)} existing encodings.")

else:
    known_encodings = []
    known_names = []

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

    eid_path = member[4]
    pass_path = member[5]
    eid_y1, eid_y2, eid_x1, eid_x2 = member[6], member[7], member[8], member[9]

    print(f"Processing {name}...")

        #eid face
    eid_img = np.array(pdf2image.convert_from_path(IMAGES_PATH + eid_path, dpi = 300)[0])
    eid_enc = face_recognition.face_encodings(eid_img, known_face_locations=[(eid_y1, eid_x2, eid_y2, eid_x1)])

    if eid_enc:
        known_encodings.append(eid_enc[0])
        known_names.append(name)

        print(f"\nDone! Loaded {len(known_encodings)} encodings for {len(members)} people.")
        print("People loaded:", list(set(known_names)))

    else:
        print(f"    WARNING: No face found in {name}'s EID")

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    face_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_loca = face_recognition.face_locations(face_frame)
    face_enc = face_recognition.face_encodings(face_frame, known_face_locations=face_loca)

    for face_encoding, face_location in zip(face_enc, face_loca):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match = np.argmin(distances)
        name = known_names[best_match]

        if matches[best_match] == True:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)  #(0,255,0) green, 2 thickness of text
            cv2.putText(frame, f"person {name}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Family Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break