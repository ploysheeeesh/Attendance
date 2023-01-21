import face_recognition
import cv2
import numpy as np
import csv
import math
from datetime import datetime

video_capture = cv2.VideoCapture(0)

musk_image = face_recognition.load_image_file("photos/Elon Musk.jpg")
musk_encoding = face_recognition.face_encodings(musk_image)[0]

mark_image = face_recognition.load_image_file("photos/Mark Zukaburng.jpg")
mark_encoding = face_recognition.face_encodings(mark_image)[0]

know_face_encoding = [
    musk_encoding,
    mark_encoding,
    ploy_encoding
]
know_face_names = [
    "Elon Musk",
    "Mark Zukaburng",
    "Va Kunthea"
]
students = know_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0, 0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(know_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = know_face_names[best_match_index]

            face_names.append(name)
            if name in know_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("attendence system", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()
