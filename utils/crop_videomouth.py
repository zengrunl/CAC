import cv2
import numpy as np

# Load the landmarks data


lines = open('/root/data1/LSR/LSR2/lrs2_rebuild/filename_input.csv').read().splitlines()
for filename_idx, line in enumerate(lines):
    filename_elements = line.split(',')
    for filename in filename_elements:
        videos_path = '/root/data1/LSR/LSR2/lrs2_rebuild/landmark/' + filename + '.npz'
        face_path = '/root/data1/LSR/LSR2/lrs2_rebuild/faces/' + filename + '.mp4'
        out_path = '/root/data1/LSR/LSR2/lrs2_rebuild/mouth_only/' + filename + '.mp4'
        landmarks_data = np.load(videos_path)["data"]

        # You need to know the specific indices for the mouth in your landmarks_data
        # Here, I suppose indices 48 ~ 60 are for the mouth
        mouth_indices = list(range(48, 68))

        # Define video capture object
        cap = cv2.VideoCapture(face_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (96, 96))
        frame_count = 0
        while True:
            ret, frame = cap.read()

            if ret:
                # Note that, you should make sure that the frame_count does not exceed your landmarks_data length
                landmarks = landmarks_data[frame_count]

                # Extract the coordinates for the mouth
                mouth_landmarks = landmarks[0][mouth_indices]

                # Get the min and max coordinates, and use them to crop the mouth from the frame
                x = min(mouth_landmarks[:, 0])
                y = min(mouth_landmarks[:, 1])
                w = max(mouth_landmarks[:, 0]) - x
                h = max(mouth_landmarks[:, 1]) - y

                mouth = frame[int(y):int(y + h), int(x):int(x + w)]

                # You may need to adapt the size of the output frame to fit your VideoWriter's size.
                # In this example, the output frame size is (100, 100)
                resized_mouth = cv2.resize(mouth, (96, 96))
                out.write(resized_mouth)
                # Show the mouth region
                # cv2.imshow("Mouth", mouth)

                frame_count += 1


            # Quit program when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()