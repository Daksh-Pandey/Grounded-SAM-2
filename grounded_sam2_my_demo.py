import os
import shutil
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# Updated hyperparameters
TEXT_PROMPT = "leg. head. tail. torso."
INPUT_DIR = "./data/dog_mesh_views/"
OUTPUT_DIR = Path("./outputs/dog/")
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.device_count() > 1 else "cpu"
DUMP_JSON_RESULTS = True

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for item in OUTPUT_DIR.iterdir():
    if item.is_file():
        item.unlink()  # Remove file
    elif item.is_dir():
        shutil.rmtree(item)

# Build SAM2 predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# Iterate over all images in the input directory
for img_filename in os.listdir(INPUT_DIR):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, img_filename)
        
        # Load image and set it in SAM2 predictor
        image_source, image = load_image(img_path)
        sam2_predictor.set_image(image_source)

        # Autocast setup for GPU performance
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(1).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Generate predictions using Grounding DINO
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


        # Perform mask prediction
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Squeeze masks if necessary
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels
        class_ids = np.array(list(range(len(class_names))))

        # Prepare labels for annotation
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        # Visualize and save annotated images
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_filename)[0]}_annotated.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_filename)[0]}_annotated_with_mask.jpg"), annotated_frame)

        # Save results in JSON format
        if DUMP_JSON_RESULTS:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            input_boxes = input_boxes.tolist()
            scores = scores.tolist()
            results = {
                "image_path": img_path,
                "annotations": [
                    {"class_name": class_name, "bbox": box, "segmentation": mask_rle, "score": score}
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
            with open(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_filename)[0]}_results.json"), "w") as f:
                json.dump(results, f, indent=4)
