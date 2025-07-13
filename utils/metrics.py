import numpy as np
from shapely.geometry import Polygon
from collections import namedtuple

Detection = namedtuple('Detection', ['points', 'score'])

def polygon_from_points(points):
    """Convert points to shapely polygon."""
    try:
        return Polygon(points)
    except:
        return None

def calculate_iou(poly1, poly2):
    """Calculate IoU between two polygons."""
    try:
        if not poly1.is_valid or not poly2.is_valid:
            return 0
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0
    except:
        return 0

def evaluate_detections(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """
    Evaluate detection results.
    Args:
        gt_boxes: List of ground truth boxes (N x 4 x 2)
        pred_boxes: List of predicted boxes (M x 4 x 2)
        pred_scores: List of prediction scores (M,)
        iou_threshold: IoU threshold for considering a match
    Returns:
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        pred_matched: List of matched prediction indices
        gt_matched: List of matched ground truth indices
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0, 0, 0, [], []

    # Convert boxes to polygons
    gt_polys = [polygon_from_points(box) for box in gt_boxes]
    pred_polys = [polygon_from_points(box) for box in pred_boxes]

    # Remove invalid polygons
    valid_gt = [(i, poly) for i, poly in enumerate(gt_polys) if poly is not None]
    valid_pred = [(i, poly, score) for i, (poly, score) in enumerate(zip(pred_polys, pred_scores)) 
                 if poly is not None]

    # Sort predictions by confidence
    valid_pred.sort(key=lambda x: x[2], reverse=True)

    gt_matched = []
    pred_matched = []
    
    # Match predictions to ground truth
    for pred_idx, pred_poly, _ in valid_pred:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_poly in valid_gt:
            if gt_idx in gt_matched:
                continue
                
            iou = calculate_iou(pred_poly, gt_poly)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            pred_matched.append(pred_idx)
            gt_matched.append(best_gt_idx)

    # Calculate metrics
    num_gt = len(valid_gt)
    num_pred = len(valid_pred)
    num_correct = len(pred_matched)

    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_gt if num_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, pred_matched, gt_matched

def compute_batch_metrics(batch_gt_boxes, batch_pred_boxes, batch_pred_scores, iou_threshold=0.5):
    """
    Compute metrics for a batch of predictions.
    Args:
        batch_gt_boxes: List of ground truth boxes for each image
        batch_pred_boxes: List of predicted boxes for each image
        batch_pred_scores: List of prediction scores for each image
        iou_threshold: IoU threshold for considering a match
    Returns:
        mean_precision: Mean precision across batch
        mean_recall: Mean recall across batch
        mean_f1: Mean F1 score across batch
    """
    batch_metrics = []
    for gt_boxes, pred_boxes, pred_scores in zip(batch_gt_boxes, batch_pred_boxes, batch_pred_scores):
        precision, recall, f1, _, _ = evaluate_detections(gt_boxes, pred_boxes, pred_scores, iou_threshold)
        batch_metrics.append((precision, recall, f1))
    
    batch_metrics = np.array(batch_metrics)
    mean_precision = np.mean(batch_metrics[:, 0])
    mean_recall = np.mean(batch_metrics[:, 1])
    mean_f1 = np.mean(batch_metrics[:, 2])
    
    return mean_precision, mean_recall, mean_f1 