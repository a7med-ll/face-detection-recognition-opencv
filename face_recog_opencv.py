import cv2 
import numpy as np
import face_recognition 
import pdf2image 
import mysql.connector
import os 
import pickle
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "databse.env"))  

#  DB + CONFIG

mydb = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME", "Family_Data_OpenCV")
)
cursor = mydb.cursor()

ENCODINGS_FILE = "encodings.pkl"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "../images/")


#  LOAD ENCODINGS

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['encodings'])} existing encodings.")
        return data["encodings"], data["names"]
    return [], []

def save_encodings(encodings, names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"Saved! Total encodings: {len(encodings)} for {len(set(names))} people.")


#  MENU

def show_menu():
    print("\n" + "="*45)
    print("       FAMILY FACE RECOGNITION SYSTEM")
    print("="*45)
    print("  1. Skip enrollment / Start live detection")
    print("  2. Add new member")
    print("  3. Enroll existing database members")
    print("  4. Exit")
    print("="*45)
    choice = input("Enter your choice (1-4): ").strip()
    return choice


#  CAPTURE PROMPT (for enrollment)

prompts = ["Look straight", "Look left", "Look right", "Smile", "Look up"]

def capture_prompt(text, cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Follow prompt: {text}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            return frame
    return None


#  ENROLL FROM DATABASE (existing members)

def enroll_from_database(known_encodings, known_names):
    cursor.execute("SELECT * FROM members")
    members = cursor.fetchall()

    for member in members:
        name = member[1]
        if name in known_names:
            print(f"{name} already enrolled, skipping...")
            continue

        eid_path = member[4]
        eid_y1, eid_y2, eid_x1, eid_x2 = member[6], member[7], member[8], member[9]
        print(f"Processing {name}...")

        eid_img = np.array(pdf2image.convert_from_path(IMAGES_PATH + eid_path, dpi=300)[0])
        eid_enc = face_recognition.face_encodings(
            eid_img, known_face_locations=[(eid_y1, eid_x2, eid_y2, eid_x1)]
        )

        if eid_enc:
            known_encodings.append(eid_enc[0])
            known_names.append(name)
            print(f"  EID face loaded for {name}")
        else:
            print(f"  WARNING: No face found in {name}'s EID")

        print(f"  Enrolling via webcam: {name} - Please sit in front of the camera")
        cap = cv2.VideoCapture(0)
        enrolled_count = 0

        for prompt in prompts:
            frame = capture_prompt(prompt, cap)
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_enc = face_recognition.face_encodings(rgb_frame)
            if frame_enc:
                known_encodings.append(frame_enc[0])
                known_names.append(name)
                enrolled_count += 1
                print(f"  Captured: {prompt}")
            else:
                print(f"  No face detected for {prompt}, skipping")

        cap.release()
        cv2.destroyAllWindows()
        print(f"  Done! {enrolled_count}/5 shots captured for {name}")

    save_encodings(known_encodings, known_names)
    return known_encodings, known_names


#  ADD NEW MEMBER (manual, no DB needed)

def add_new_member(known_encodings, known_names):
    name = input("Enter the new member's name: ").strip()
    if not name:
        print("No name entered, cancelling.")
        return known_encodings, known_names

    cap = cv2.VideoCapture(0)
    enrolled_count = 0

    print(f"\nEnrolling {name} — please sit in front of the camera.")
    for prompt in prompts:
        frame = capture_prompt(prompt, cap)
        if frame is None:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_enc = face_recognition.face_encodings(rgb_frame)
        if frame_enc:
            known_encodings.append(frame_enc[0])
            known_names.append(name)
            enrolled_count += 1
            print(f"  Captured: {prompt}")
        else:
            print(f"  No face detected for {prompt}, skipping")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! {enrolled_count}/5 shots captured for {name}")
    save_encodings(known_encodings, known_names)
    return known_encodings, known_names


#  LIVE DETECTION  (lag-optimised)
#  - resize frame to 1/2 before recognition
#  - only process every 3rd frame

def live_detection(known_encodings, known_names):
    if not known_encodings:
        print("No encodings found! Please enroll members first.")
        return

    cap = cv2.VideoCapture(0)

    # --- reduce lag: set smaller buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    face_locations  = []
    face_names      = []
    frame_count     = 0
    PROCESS_EVERY_N = 3        # only run recognition every 3rd frame
    SCALE           = 0.5      # shrink frame for faster recognition

    print("Live detection started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N == 0:
            # Shrink for speed
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb_small   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations_small = face_recognition.face_locations(rgb_small)
            face_encodings_list  = face_recognition.face_encodings(rgb_small, face_locations_small)

            face_names = []
            face_locations = []

            for enc, loc in zip(face_encodings_list, face_locations_small):
                matches   = face_recognition.compare_faces(known_encodings, enc)
                distances = face_recognition.face_distance(known_encodings, enc)
                best_match = np.argmin(distances)

                if matches[best_match]:
                    label = known_names[best_match]
                    color = (0, 255, 0)   # green
                else:
                    label = "Unknown"
                    color = (0, 0, 255)   # red

                # Scale location back to full frame size
                top, right, bottom, left = loc
                top    = int(top    / SCALE)
                right  = int(right  / SCALE)
                bottom = int(bottom / SCALE)
                left   = int(left   / SCALE)

                face_locations.append((top, right, bottom, left, color, label))

        # Draw boxes on every frame
        for (top, right, bottom, left, color, label) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Family Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # Q, q, or ESC
            break

        # Also exit if the window is closed with the X button
        if cv2.getWindowProperty("Family Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  


#  MAIN

if __name__ == "__main__":
    known_encodings, known_names = load_encodings()

    while True:
        choice = show_menu()

        if choice == "1":
            live_detection(known_encodings, known_names)

        elif choice == "2":
            known_encodings, known_names = add_new_member(known_encodings, known_names)

        elif choice == "3":
            known_encodings, known_names = enroll_from_database(known_encodings, known_names)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice, please enter 1-4.")