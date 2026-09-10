import sys
import numpy as np
import cv2

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
total_frames = int(fps * duration)

output_file = sys.argv[3] if len(sys.argv) > 3 else 'simulation.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (800, 600))

frame_num = 0
while frame_num < total_frames:
    line = sys.stdin.readline()
    if not line:
        break
    if not line.startswith("FRAME"):
        continue
    
    N = int(line.split()[1])
    positions = np.zeros((N, 3))
    for i in range(N):
        parts = sys.stdin.readline().split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    scale = 3
    offset = 400
    x = (positions[:, 0] * scale + offset).astype(int)
    y = (positions[:, 1] * scale + 300).astype(int)
    mask = (x >= 0) & (x < 800) & (y >= 0) & (y < 600)
    img[y[mask], x[mask]] = (255, 255, 255)
    
    cv2.putText(img, f'Frame {frame_num}/{total_frames}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out.write(img)
    frame_num += 1

out.release()
print(f"Saved {output_file}", file=sys.stderr)