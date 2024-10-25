import cv2
import numpy as np
import dlib
from imutils import face_utils
import os
import time

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath, actionModelPath, actionClassesPath):
        self.videoPath         = videoPath
        self.configPath        = configPath
        self.modelPath         = modelPath
        self.classesPath       = classesPath
        self.actionModelPath   = actionModelPath
        self.actionClassesPath = actionClassesPath

        # Load object detection model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Load action recognition model
        self.actionNet = cv2.dnn.readNet(self.actionModelPath)

        # Initialize dlib's face detector and shape predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_landmark_path = os.path.join("server", "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        self.readClasses()

        # Camera matrix and distortion coefficients for head pose estimation
        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                      [1.330353, 7.122144, 6.903745],
                                      [-1.330353, 7.122144, 6.903745],
                                      [-6.825897, 6.760612, 4.402142],
                                      [5.311432, 5.485328, 3.987654],
                                      [1.789930, 5.393625, 4.413414],
                                      [-1.789930, 5.393625, 4.413414],
                                      [-5.311432, 5.485328, 3.987654],
                                      [2.005628, 1.409845, 6.165652],
                                      [-2.005628, 1.409845, 6.165652],
                                      [2.774015, -2.080775, 5.048531],
                                      [-2.774015, -2.080775, 5.048531],
                                      [0.000000, -3.116408, 6.097667],
                                      [0.000000, -7.415691, 4.070434]])

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classeslist = f.read().splitlines()
        self.classeslist.insert(0, '__Background__')

        with open(self.actionClassesPath, 'r') as f:
            self.actionClasseslist = f.read().splitlines()
        self.actionClasseslist.insert(0, '__Background__')

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return euler_angle

    def determine_gaze_direction(self, euler_angle, bbox_center=None, image_size=None):
        pitch, yaw, roll = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]

        yaw_threshold             = 20
        pitch_threshold           = 20
        center_distance_threshold = 0.2  # Proportion of the image size

        if bbox_center and image_size:
            image_center         = (image_size[0] // 2, image_size[1] // 2)
            bbox_x, bbox_y       = bbox_center
            distance_from_center = np.sqrt((bbox_x - image_center[0])**2 + (bbox_y - image_center[1])**2) / np.linalg.norm(image_size)

            if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and distance_from_center < center_distance_threshold:
                return "Looking at Camera"
        else:
            if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold:
                return "Looking at Camera"

        if yaw > yaw_threshold:
            return "Looking Right"
        elif yaw < -yaw_threshold:
            return "Looking Left"
        elif pitch > pitch_threshold:
            return "Looking Down"
        elif pitch < -pitch_threshold:
            return "Looking Up"
        else:
            return "Looking Straight"

    def calculate_attention_score(self, action_label, gaze_direction, using_phone, yaw, pitch):
        # Base score based on action
        if action_label in ["reading", "looking_forward", "hand_raising"]:
            attention_score = 8
        elif action_label in ["using_phone", "turning_around", "sleeping"]:
            attention_score = 3
        else:
            attention_score = 5

        # Adjust score based on gaze direction
        if gaze_direction == "Looking at Camera":
            attention_score += 2
        elif gaze_direction == "Looking Straight":
            attention_score += 1
        elif gaze_direction in ["Looking Left", "Looking Right"]:
            attention_score -= 2
        elif gaze_direction in ["Looking Up", "Looking Down"]:
            attention_score -= 1

        # Further reduce score if the person is using a phone
        if using_phone:
            attention_score -= 5

        # Adjust based on yaw (left/right) and pitch (up/down) angles
        if abs(yaw) > 25:
            attention_score -= 2
        if abs(pitch) > 20:
            attention_score -= 1

        # Ensure the score is within the range of 1 to 10
        return min(max(attention_score, 1), 10)
      

    def get_student_attention(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video...")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Handle zero FPS by setting a default frame skip interval
        if fps == 0:
            print("Warning: FPS is 0, setting default frame skip interval to 5.")
            frame_skip_interval = 5  # Default to skipping 5 frames
        else:
            frame_skip_interval = int(fps / 5)

        print(f"Processing 5 frames per second, skipping {frame_skip_interval} frames")

        count = 0
        person_bboxes = []
        attention_scores = []
        person_idx = 0  # To track which person is being processed

        # Initial 5-second pause
        print("Waiting for 5 seconds...")
        time.sleep(5)

        start_time = time.time()
        start_detection = False  # To control when to start student identification

        while True:
            success, image = cap.read()
            if not success:
                break

            # Show the zoomed frame with padding
            # cv2.imshow('Original', image)

            count += 1
            if count % frame_skip_interval != 0:
                continue

            # If it's time to start detection
            if not start_detection:
                start_detection = True
                print("Starting detection after 5 seconds...")

            # Detect people in the first frame after initial 5 seconds
            if not person_bboxes:
                classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
                bboxs = list(bboxs)
                confidences = list(np.array(confidences).reshape(1, -1)[0])
                confidences = list(map(float, confidences))

                bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

                if len(bboxIdx) != 0:
                    for i in range(len(bboxIdx)):
                        bbox = bboxs[np.squeeze(bboxIdx[i])]
                        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                        classLabel = self.classeslist[classLabelID]

                        if classLabel.lower() == "person":
                            person_bboxes.append(bbox)

                # If no people are found, continue to next frame
                if not person_bboxes:
                    continue

            # Zoom into one person at a time based on person_idx
            person_bbox = person_bboxes[person_idx]
            x, y, w, h = person_bbox

            # Add padding around the bounding box
            padding = 20
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, image.shape[1] - x)
            h = min(h + 2 * padding, image.shape[0] - y)

            # Crop the image to the new bounding box with padding
            personROI = image[y:y + h, x:x + w]

            # Reapply detection to ensure only one person is in the frame
            classLabelIDs, confidences, bboxs = self.net.detect(personROI, confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            # phone_bboxes = []

            if len(bboxIdx) != 0:
                for i in range(len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    sx, sy, sw, sh = bbox
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classeslist[classLabelID]
                    
                    if classLabel.lower() == "person":
                        # If a person is detected, proceed with attention calculations
                        gray = cv2.cvtColor(personROI, cv2.COLOR_BGR2GRAY)
                        rects = self.face_detector(gray, 0)
                    # elif classLabel.lower() == "cell phone":
                    #     phone_bboxes.append(bbox)    

                        using_phone = False
                        actionLabel, gaze_direction, attention_score = None, None, None

                        # for phone_bbox in phone_bboxes:
                        #     px, py, pw, ph = phone_bbox
                        #     if (px > sx and px + pw < sx + sw and py > sy and py + ph < sy + sh):
                        #         using_phone = True
                        #         break

                        for rect in rects:
                            shape = self.predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            euler_angle = self.get_head_pose(shape)
                            yaw, pitch, roll = euler_angle[1, 0], euler_angle[0, 0], euler_angle[2, 0]
                            gaze_direction = self.determine_gaze_direction(euler_angle, bbox_center=(x + w // 2, y + h // 2), image_size=image.shape[:2])

                            # Action recognition
                            blob = cv2.dnn.blobFromImage(personROI, scalefactor=1.0 / 255, size=(224, 224), mean=(0, 0, 0), swapRB=True, crop=False)
                            self.actionNet.setInput(blob)
                            actionPreds = self.actionNet.forward()

                            actionClassID = np.argmax(actionPreds)
                            actionLabel = self.actionClasseslist[actionClassID]

                            # Calculate attention score
                            # if using_phone:
                            #     actionLabel = 'using phone'
                            #     attention_score = self.calculate_attention_score(actionLabel, gaze_direction, using_phone, yaw, pitch)
                            # else:  
                            attention_score = self.calculate_attention_score(actionLabel, gaze_direction, using_phone, yaw, pitch)  

                            # Store the results
                            attention_scores.append({
                                'student_id': person_idx,
                                'action': actionLabel,
                                'gaze_direction': gaze_direction,
                                'attention_score': attention_score
                            })
                            # Draw bounding box for person
                            cv2.rectangle(personROI, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                            text = f"{actionLabel}, {gaze_direction}, {attention_score}"
                            cv2.putText(personROI, text, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            print(text)

                        # Display the image with bounding box and attention details
                        # cv2.imshow('Zoomed', personROI)

            # After 3 seconds, move to the next person
            if time.time() - start_time >= 3:
                person_idx = (person_idx + 1) % len(person_bboxes)  # Move to the next person
                start_time = time.time()  # Reset the timer

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()
        return attention_scores

    def get_current_frame(self):
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    

        



        

