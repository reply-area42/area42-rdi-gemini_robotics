import cv2
import os

import argparse

parser = argparse.ArgumentParser(description="Genera un video da singoli frame")
parser.add_argument("--input",  "-i", type=str, required=True, help="Cartella contenente i frame")
parser.add_argument("--output", "-o", type=str, default="output.mp4", help="Path del video di output")
parser.add_argument("--fps",    "-f", type=int, default=30, help="Frame per secondo (default: 30)")

args = parser.parse_args()

# Utilizzo
print(args.input)   # es. ./frames
print(args.output)  # es. output.mp4
print(args.fps)     # es. 30


frame_dir = args.input        
output_path = args.output
fps = args.fps                               
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec

frames = sorted([f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg"))])

first = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w, _ = first.shape
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

for fname in frames:
    img = cv2.imread(os.path.join(frame_dir, fname))
    writer.write(img)

writer.release()