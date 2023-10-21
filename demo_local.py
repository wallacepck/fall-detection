from argparse import ArgumentParser

import clip
import torch
from util import cvtransform

import cv2

from packaging import version
assert version.parse(torch.__version__) >= version.parse("1.7.1"), "pytorch version must be >= 1.7.1"

parser = ArgumentParser()
parser.add_argument("-i", dest="input",
                    help="video to infer from", required=False, metavar="INPUT")
args = parser.parse_args()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

targets = [
    "a man lying down on the floor",
    "a man standing up",
    "no people"
]

# sigh...
preprocess = cvtransform(preprocess)

# Prepare the inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in targets]).to(device)
print(len(targets))

def draw_text_box(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0),
          margin=3,
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x - margin, y - margin), (x + text_w + margin, y + text_h + margin), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def infer(src, text_features) -> (bool, torch.tensor):
    image_input = preprocess(src).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(targets))
        results = list(zip(values, indices))

        # Print the result
        # print("\nTop predictions:\n")
        for value, index in results:
            # print(f"{targets[index]:>24s}: {100 * value.item():.2f}%")
            if index == 0 and value > 0.1:
                return True, results
    
    return False, results

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Video
videoname = "falling.mp4"
if args.input != None:
    videoname = args.input
cap = cv2.VideoCapture(videoname)
framecount = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = 640 / frame.shape[0]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    warn, pred = infer(image, text_features)
    for i, (value, index) in enumerate(pred):
        text = f"{targets[index]}: {100 * value.item():.2f}%"
        (w,h),b = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, thickness=None)
        cv2.rectangle(frame, (0, i*16+16-b//2), (0+w, i*16+16+h+b//2), color=(0,0,0), thickness=cv2.FILLED)
        cv2.putText(frame, text, (0, i*16+16+h), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), lineType=cv2.LINE_AA)

    if warn:
        draw_text_box(frame, "POTENTIAL FALL DETECTED", pos=(0, 640-32), text_color=(0,0,0), text_color_bg=(64,64,255))

    cv2.imshow("out", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    framecount += 1