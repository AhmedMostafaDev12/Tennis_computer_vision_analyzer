import os 
import albumentations as albu
import cv2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import torch


def load_image_and_anns(im_sets, label2idx, ann_fname, split):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: ['data/VOC2007', 'data/VOC2012']
    :param label2idx: Class Name to index mapping for dataset {'person': 0, 'dog': 1, 'cat': 2, ...}
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :return:

    A list of dictionaries, one per image, containing:
        Image filename
        Image dimensions
        List of objects (bounding boxes + labels)
    """

    im_infos = []
    ims = []
    ## Loop through datasets: im_sets = ['data/VOC2007', 'data/VOC2012']
    for im_set in im_sets:
        im_names = []
        # Read image names listed in the txt file data/VOC2007/ImageSets/Main/trainval.txt
        for line in open(os.path.join(
            im_set, 'ImageSets','Main',f'{ann_fname}.txt')):
            im_names.append(line.strip())

        ann_dir = os.path.join(im_set, 'Annotations') #  XML files with bounding boxes
        img_dir = os.path.join(im_set, 'JPEGImages') # Image files
        for im_name in im_names:
            """
            ├── Annotations/             
            │   ├── 000001.xml
            │   ├── 000002.xm
            │   └── ...
            """
            ann_file = os.path.join(ann_dir, f'{im_name}.xml')
            im_info = {}

            ## Parse XML file:
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot() # Get root <annotation> element

            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Get image ID and filename
            im_info['img_id'] = os.path.basename(ann_file).spilt('.xml')[0]
            im_info['filename'] = os.path.join(
                img_dir, f'{im_info["img_id"]}.jpg')
            im_info['width'] = width
            im_info['height'] = height
            detections = []


            # Loop through all object elements in the XML
            any_valid_object = False
            for obj in ann_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bndbox = obj.find('bndbox')
                bbox = [
                    int(float(bndbox.find('xmin').text)) - 1,
                    int(float(bndbox.find('ymin').text)) - 1,
                    int(float(bndbox.find('xmax').text)) - 1,
                    int(float(bndbox.find('ymax').text)) - 1
                ]
                det['label'] = label
                det['bbox'] = bbox
                det['difficult'] = difficult
                # Ignore difficult rois during training
                # At test time, we consider all rois for evaluation
                if split == 'train' and difficult == 0:
                    detections.append(det)
                    any_valid_object = True

            # Only add images with existing objects
            if any_valid_object:
                im_info['detections'] = detections
                im_infos.append(im_info)
    print("Loaded {} {} images".format(len(im_infos)))
    return im_infos


class VOCDataset(Dataset):
    """
    Knows where all images are (using the load_image_and_anns() function)
    Loads and processes one image at a time when asked
    Transforms images (resize, augment, normalize)
    Converts bounding boxes from Pascal VOC format → YOLO format
    Returns ready-to-train data to your model
    """
    def __init__(self, split, im_sets, im_size=448, S=7, B=2, C=20):
        self.split = split
        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        self.S = S  # grid size
        self.B = B  # number of boxes per grid cell
        self.C = C  # number of classes

        # train/test transforms
        # dictionary of transforms for train and test
        self.transform = {
            'train' : albu.Compose([
                albu.HorizontalFlip(p=0.5),
                ## zoom in/out from (.8 to 1.2) and translate by -20% to +20%
                albu.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), 
                            always_apply=True),

                
                albu.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue = (-0.2, 0.2),
                    always_apply=None,
                    p=0.5
                ),
                albu.Resize(self.im_size,self.im_size)],

                # tell albumentations that we are using pascal_voc bboxes 
                # so it also transforms the bounding boxes accordingly
                bbox_params=albu.BboxParams(
                    format='pascal_voc',
                    label_fields=['labels']
                )),
                'test' : albu.Compose([
                    albu.Resize(self.im_size,self.im_size)],

                    bbox_params = albu.BboxParams(
                        format='pascal_voc',
                        label_fields=['labels']
                    ))
        }

        # Class name to index mapping
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        classes = sorted(classes)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx : classes[idx] for idx in range(len(classes))}
        self.im_infos = load_image_and_anns(
            self.im_sets, self.label2idx, self.fname, self.split)
        
    def __len__(self):
        return len(self.im_infos)
    
    def __getitem__(self, index):
        im_info = self.im_info[index]
        im = cv2.imread(im_info['filename'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Get annotations for this image 
        bboxes = [detection['bbox'] for detection in im_info['detections']]
        labels = [detection['label'] for detection in im_info['detections']]
        difficult = [detection['difficult'] for detection in im_info['detections']]

        # transform Image and ann according to augmentations list 
        transformed_info = self.transform[self.split](
            image=im,
            bboxes=bboxes,
            labels=labels
        )

        im = transformed_info['image']
        # get transformed bboxes and labels
        bboxes = torch.as_tensor(transformed_info['bboxes'])
        labels = torch.as_tensor(transformed_info['labels'])
        difficult = torch.as_tensor(difficult)

        """
        Divide by 255:
            im / 255.  # Convert from [0, 255] to [0, 1]

        Convert to tensor and permute:
            im_tensor = torch.from_numpy(im/255.).permute((2,0,1))
            # Before permute: (448, 448, 3) - H, W, C
            # After permute:  (3, 448, 448) - C, H, W (PyTorch format!)
        Normalize each channel with ImageNet statistics:

            # ImageNet mean and std (standard for pretrained models)
            mean = [0.485, 0.456, 0.406]  # RGB
            std = [0.229, 0.224, 0.225]   # RGB

            # For each channel: (value - mean) / std
            R_channel = (R - 0.485) / 0.229
            G_channel = (G - 0.456) / 0.224
            B_channel = (B - 0.406) / 0.225
        """
        im_tensor = torch.from_numpy(im/255.).permute((2,0,1)).float()
        im_tensor_channel_0 = (torch.unsqueeze(im_tensor[0],0)- 0.485)/0.229
        im_tensor_channel_1 = (torch.unsqueeze(im_tensor[1],0)- 0.456)/0.224
        im_tensor_channel_2 = (torch.unsqueeze(im_tensor[2],0)- 0.406)/0.225
        im_tensor = torch.cat(
            (im_tensor_channel_0, im_tensor_channel_1, im_tensor_channel_2),0)
        
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)

        # Prepare target tensor of shape (S, S, B*5 + C) for yolo
        target_dim = self.B * 5 + self.C
        h, w = im.shape[:2]
        yolo_target = torch.zeros(self.S, self.S, target_dim)

        # Heigh and width of each grid cell as it is square
        cell_pixels = h // self.S

        if len(bboxes) > 0:
            # convert x1y1x2y2 to xywh 
            box_widths = bboxes_tensor[:,2] - bboxes_tensor[:,0]
            box_heights = bboxes_tensor[:,3] - bboxes_tensor[:,1]
            box_xc = bboxes_tensor[:,0] + 0.5 * box_widths
            box_yc = bboxes_tensor[:,1] + 0.5 * box_heights

            ## Get cell indexes for each bbox
            box_i = torch.floor(box_xc / cell_pixels).long()
            box_j = torch.floor(box_yc / cell_pixels).long()

            # xc offset and yc offset from top-left corner of cell
            box_xc_offsets = (box_xc - (box_i * cell_pixels)) / cell_pixels
            box_yc_offsets = (box_yc - (box_j * cell_pixels)) / cell_pixels

            box_w_label = box_widths / w
            box_h_label = box_heights / h

            # Update the target array for all bboxes 

            """
            # Box 1 (indices 0-4):
                yolo_target[3, 2, 0] = 0.734  # x offset
                yolo_target[3, 2, 1] = 0.516  # y offset
                yolo_target[3, 2, 2] = sqrt(0.335) = 0.579  # sqrt(w)
                yolo_target[3, 2, 3] = sqrt(0.335) = 0.579  # sqrt(h)
                yolo_target[3, 2, 4] = 1.0  # confidence (object exists!)

            # Box 2 (indices 5-9): Same values
                yolo_target[3, 2, 5] = 0.734
                yolo_target[3, 2, 6] = 0.516
                yolo_target[3, 2, 7] = 0.579
                yolo_target[3, 2, 8] = 0.579
                yolo_target[3, 2, 9] = 1.0
            """
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                for k in range(self.B):
                    s = 5 * k
                    yolo_target[box_j[idx], box_i[idx], s] = box_xc_offsets[idx]
                    yolo_target[box_j[idx], box_i[idx], s+1] = box_yc_offsets[idx]
                    yolo_target[box_j[idx], box_i[idx], s+2] = box_w_label[idx].sqrt()
                    yolo_target[box_j[idx], box_i[idx], s+3] = box_h_label[idx].sqrt()
                    yolo_target[box_j[idx], box_i[idx], s+4] = 1.0  # confidence score

                # One-hot encoding for class label
                label = int(labels_tensor[idx])
                cls_target = torch.zeros(self.C)
                cls_target[label] = 1.0
                yolo_target[box_j[idx], box_i[idx], 5 * self.B:] = cls_target

        # For training, we use yolo_targets(xoffset, yoffset, sqrt(w), sqrt(h))
        # For evaluation we use bboxes_tensor (x1, y1, x2, y2)
        # Below we normalize bboxes tensor to be between 0-1
        # as thats what evaluation script expects so (x1/w, y1/h, x2/w, y2/h)
        if len(bboxes) > 0:
            bboxes_tensor /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes_tensor)
        targets = {
            'bboxes': bboxes_tensor,  ## for evaluation
            'labels': labels_tensor,
            'yolo_targets': yolo_target,
            'difficult': difficult,
        }
        return im_tensor, targets, im_info['filename']
                        
                        


