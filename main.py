import cv2
import argparse
from ultralytics import YOLO
import os

def run_on_image(model, image_path):
    results = model.predict(source=image_path, save=True, conf=0.5, verbose=False)
    annotated_img = results[0].plot()
    num_faces = len(results[0].boxes)
    cv2.putText(annotated_img, f'Faces: {num_faces}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_dir = results[0].save_dir  
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, annotated_img)
    return results

def run_on_video(model, video_source):
    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_path='runs/detect/video_predict/output.mp4'
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame,save=False ,conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        num_faces = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Faces: {num_faces}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(annotated_frame)
        cv2.imshow("Detection", annotated_frame)
        annotated_frame = cv2.resize(annotated_frame, (1280, 720))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Face Detection CLI")
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], required=True, help="Input type")
    parser.add_argument('--source', nargs='+', help="Path to image or video file")
    parser.add_argument('--output', default='result.mp4', help="Output video/image file path")
    parser.add_argument('--model', default='model/best.pt', help="Path to YOLOv8 model file")

    args = parser.parse_args()
    model = YOLO(args.model)

    if args.mode == 'image':
        if not args.source:
            print("[ERROR] Image path required with --source")
        else:
            for path in args.source:
                run_on_image(model, path)

    elif args.mode == 'video':
        if not args.source:
            print("[ERROR] Video path required with --source")
        else:
            for path in args.source:
                run_on_image(model, path)

    elif args.mode == 'camera':
        run_on_video(model, 0)

if __name__ == "__main__":
    main()
