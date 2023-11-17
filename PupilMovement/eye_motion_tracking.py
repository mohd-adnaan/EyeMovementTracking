import cv2

cap = cv2.VideoCapture("eye_recording.flv")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object
output_video = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("Gray ROI", gray_roi)
    cv2.imshow("Region of Interest", roi)
    
    # Write the frame to the output video file
    output_video.write(frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

# Release the VideoWriter and close all windows
output_video.release()
cv2.destroyAllWindows()
