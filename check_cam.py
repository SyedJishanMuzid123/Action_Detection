import cv2

print("Scanning for cameras... (This may take a few seconds)")

# Check the first 5 indexes
for index in range(5):
    # Try to open the camera with DirectShow
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if cap.isOpened():
        print(f"✅ Camera found at Index {index}")
        ret, frame = cap.read()
        if ret:
            print(f"   - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            # Show a quick snapshot so you know which camera it is
            cv2.imshow(f'Camera Index {index}', frame)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyAllWindows()
        else:
            print("   - Warning: Camera opened but returned no image.")
        cap.release()
    else:
        print(f"❌ No camera at Index {index}")