import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import logging

# Assumo che questi file esistano nella tua cartella
from camera.get_rs_info import *
from camera.read_D435 import *

import time

class ObjectDetector:
    def __init__(self, device="cuda"):
        self.device = device
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        self.model.eval()

    def detect(self, cv2_image, text_query, threshold=0.3):
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self.processor(
            images=pil_image,
            text=text_query,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            xyxy = box.cpu().tolist()
            u_center = (xyxy[0] + xyxy[2]) / 2.0
            v_center = (xyxy[1] + xyxy[3]) / 2.0
            detections.append({
                "label": label,
                "score": round(score.item(), 3),
                "center": (u_center, v_center),
                "bbox": [round(b, 2) for b in xyxy],
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()


def main():
    logging.basicConfig(level=logging.INFO)
    detector = ObjectDetector(device="cuda")

    devices = get_rs_info()
    if not devices:
        logging.error("Nessun dispositivo RealSense trovato.")
        exit(1)

    device_serial = str(devices[0])
    
    config = {        
        "serial_number": device_serial,        
        "width": 640,
        "height": 480,        
        "fps": 6    
    }

    camera = RealSenseCamera(            
        serial_number=config["serial_number"],            
        width=config["width"],
        height=config["height"],            
        fps=config["fps"]
    )
    
    if not camera.init_realsense():            
        logging.error("[ImageServer] Camera initialization failed. Aborting show_process().")
        return
    
    try:
        while True:  
            # color_image, depth_image = camera.get_frame()
            color_image, _ = camera.get_frame()

            # if color_image is None or depth_image is None:       
            if color_image is None:               
                logging.warning("[ImageServer] Skipping frame due to capture error.")
                continue
            
            # Detect objects
            detections = detector.detect(color_image, "face. tie. paper glass", threshold=0.3)

            # Creiamo una copia per il disegno per non sporcare l'originale
            display_image = color_image.copy()

            if detections:
                for det in detections:
                    xmin, ymin, xmax, ymax = [int(b) for b in det["bbox"]]
                    
                    # Disegno Rettangolo
                    cv2.rectangle(display_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Testo Label + score
                    label_str = f"{det['label']} {det['score']}"
                    cv2.putText(display_image, label_str, (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Disegno Centro
                    cx, cy = int(det["center"][0]), int(det["center"][1])
                    cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)
            
            # Mostra il frame aggiornato
            cv2.imshow("RealSense Detection", display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                    
                logging.info("[ImageServer] 'q' pressed – exiting.")
                break
        
            time.sleep(0.1)
                
    except KeyboardInterrupt:            
        logging.info("[ImageServer] KeyboardInterrupt received.")

    finally:            
        cv2.destroyAllWindows()
        # Se RealSenseCamera ha un metodo di chiusura, chiamalo qui
        # camera.release() 

if __name__ == "__main__":
    main()