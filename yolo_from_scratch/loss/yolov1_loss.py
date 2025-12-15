import torch
import torch.nn as nn


def get_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    box1: tensor of shape (..., 4)  (x1,y1,x2,y2)
    box2: tensor of shape (..., 4)  (x1,y1,x2,y2)
    returns: tensor of shape (...) IoU values
    """
    # Intersection coordinates
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    # Intersection area
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Areas of the boxes
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # Union area
    union_area = box1_area.clamp(min= 0) + box2_area.clamp(min= 0) - inter_area + 1e-6  # avoid division by zero

    # IoU
    iou = inter_area / union_area

    return iou

class YOLOV1Loss(nn.Module):
    r"""
    Loss module for YoloV1 which caters to the following components:
    1. Localization Loss for responsible predictor boxes
    2. Objectness Loss for responsible predictor boxes
    2. Objectness Loss for non-responsible predictor boxes of cells assigned with objects
    2. Objectness Loss for ALL predictor boxes of cells not assigned with objects
    3. Classification Loss
    """

    def __init__(self, S=7, B=2, C=20):
        super(YOLOV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5


    def forward(self, preds, targets, use_sigmoid=False):
        r"""
        Main method of loss computation
        :param preds: (Batch, S*S*(5B+C)) tensor
        :param targets: (Batch, S, S, (5B+C)) tensor.
            Target element for each cell has been duplicated 5B times(done in VOCDataset)
        :param use_sigmoid: Whether to use sigmoid activation for box predicitons or not
        """
         
        batch_size = preds.size(0)
        # Reshape preds to (B, S, S, B*5 + C)
        preds = preds.reshape(batch_size, self.S, self.S, self.B * 5 + self.C)

        if use_sigmoid:
            preds[..., 0:5*self.B] = torch.sigmoid(preds[..., 0:5*self.B])


        # Shifts for all grid cell locations.
        # Will use these for converting x_center_offset/y_center_offset
        # values to x1/y1/x2/y2(normalized 0-1)
        # S cells = 1 => each cell adds 1/S pixels of shift

        shifts_x = torch.arange(0, self.S,
                               dtype = torch.int32,
                               device = preds.device)* 1/float(self.S)  
        shifts_y = torch.arange(0, self.S,
                               dtype = torch.int32,
                               device = preds.device)* 1/float(self.S)
        
        shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing='ij')  # (S,S)
        
        # shifts -> (1, S, S, B)
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # pred_boxes -> (Batch_size, S, S, B, 5)
        pred_boxes = preds[..., :5*self.B].reshape(batch_size,
                                                   self.S,
                                                   self.S,
                                                   self.B,
                                                   -1)
        
        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)
        # x_center = (xc_offset / S + shift_x)
        # x1 = x_center - 0.5 * w
        # x2 = x_center + 0.5 * w
        pred_boxes_x1 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         - 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x1 = pred_boxes_x1[..., None]
        pred_boxes_y1 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         - 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y1 = pred_boxes_y1[..., None]
        pred_boxes_x2 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         + 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x2 = pred_boxes_x2[..., None]
        pred_boxes_y2 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         + 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y2 = pred_boxes_y2[..., None]
        pred_boxes_x1y1x2y2 = torch.cat([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2], dim=-1)

        # target_boxes -> (Batch_size, S, S, B, 5)
        target_boxes = targets[..., :5*self.B].reshape(batch_size,
                                                       self.S,
                                                       self.S,
                                                       self.B,
                                                       -1)
        target_boxes_x1 = ((target_boxes[..., 0] / self.S + shifts_x)
                           - 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x1 = target_boxes_x1[..., None]
        target_boxes_y1 = ((target_boxes[..., 1] / self.S + shifts_y)
                           - 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y1 = target_boxes_y1[..., None]
        target_boxes_x2 = ((target_boxes[..., 0] / self.S + shifts_x)
                           + 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x2 = target_boxes_x2[..., None]
        target_boxes_y2 = ((target_boxes[..., 1] / self.S + shifts_y)
                           + 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y2 = target_boxes_y2[..., None]
        target_boxes_x1y1x2y2 = torch.cat([
            target_boxes_x1,
            target_boxes_y1,
            target_boxes_x2,
            target_boxes_y2
        ], dim=-1)

        




