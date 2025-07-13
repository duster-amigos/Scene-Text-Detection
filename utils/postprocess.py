import cv2
import numpy as np
import torch
import pyclipper
from shapely.geometry import Polygon

def get_mini_boxes(contour):
    """Get minimum bounding box for a contour."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box)

def unclip(box, unclip_ratio=1.5):
    """Expand a box by a certain ratio."""
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance)[0])
    return expanded

def get_boxes_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=1000, 
                         min_size=3, box_thresh=0.7, unclip_ratio=1.5):
    """
    Get boxes from bitmap using contour detection and post-processing.
    Args:
        pred: The raw predictions from the model
        bitmap: The binary map after thresholding
        dest_width: Original image width
        dest_height: Original image height
        max_candidates: Maximum number of box candidates to consider
        min_size: Minimum box size to keep
        box_thresh: Confidence threshold for boxes
        unclip_ratio: Ratio to expand boxes
    Returns:
        boxes: List of detected text boxes
        scores: Confidence scores for each box
    """
    height, width = bitmap.shape
    boxes = []
    scores = []

    try:
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), 
                                     cv2.RETR_LIST, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    except:
        return [], []

    for contour in contours[:max_candidates]:
        # Filter small boxes
        if cv2.contourArea(contour) < min_size:
            continue

        # Get minimum bounding box
        points = get_mini_boxes(contour)
        if points is None:
            continue

        # Calculate score
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        # Expand box
        points = unclip(points, unclip_ratio)
        if points is None or len(points) == 0:
            continue

        # Get minimum bounding box again
        points = get_mini_boxes(points.astype(np.int32))
        if points is None:
            continue

        # Rescale points to original image size
        points = points.reshape((-1, 2))
        points[:, 0] = points[:, 0] * (dest_width / width)
        points[:, 1] = points[:, 1] * (dest_height / height)
        boxes.append(points)
        scores.append(score)

    return boxes, scores

def box_score_fast(bitmap, _box):
    """
    Calculate score for a text box.
    Args:
        bitmap: Probability map
        _box: Box coordinates
    Returns:
        score: Mean probability within the box
    """
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    score = cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    return score

def boxes_from_batch(batch_pred, batch_size, min_size=3, box_thresh=0.7, unclip_ratio=1.5):
    """
    Process batch predictions to get text boxes.
    Args:
        batch_pred: Batch predictions from model (N, 2, H, W)
        batch_size: Batch size
        min_size: Minimum box size
        box_thresh: Box confidence threshold
        unclip_ratio: Box expansion ratio
    Returns:
        batch_boxes: List of boxes for each image
        batch_scores: List of scores for each image
    """
    batch_boxes = []
    batch_scores = []

    for i in range(batch_size):
        pred = batch_pred[i]
        pred = torch.sigmoid(pred)
        
        # Get binary map
        bitmap = pred[0] > box_thresh
        
        height, width = bitmap.shape
        boxes, scores = get_boxes_from_bitmap(pred[0].cpu().numpy(),
                                            bitmap.cpu().numpy(),
                                            width, height,
                                            min_size=min_size,
                                            box_thresh=box_thresh,
                                            unclip_ratio=unclip_ratio)
        batch_boxes.append(boxes)
        batch_scores.append(scores)

    return batch_boxes, batch_scores 

def process_predictions(pred, min_size=3, box_thresh=0.7, unclip_ratio=1.5):
    """
    Process model predictions to get text boxes and scores.
    Args:
        pred: Model prediction tensor (2, H, W)
        min_size: Minimum box size
        box_thresh: Box confidence threshold
        unclip_ratio: Box expansion ratio
    Returns:
        boxes: List of detected text boxes
        scores: List of confidence scores
    """
    pred = torch.sigmoid(pred)
    
    # Get binary map
    bitmap = pred[0] > box_thresh
    
    height, width = bitmap.shape
    boxes, scores = get_boxes_from_bitmap(pred[0].cpu().numpy(),
                                        bitmap.cpu().numpy(),
                                        width, height,
                                        min_size=min_size,
                                        box_thresh=box_thresh,
                                        unclip_ratio=unclip_ratio)
    
    return boxes, scores 