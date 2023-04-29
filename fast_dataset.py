import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as T
import numpy as np
class FastRCNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, classes, im_size=None, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.im_size = im_size
        # load all image files, sorting them to ensure that they are aligned
        self.images = df["image_path"].tolist()
        self.annot_paths = df["annot_path"].tolist()
        self.labels = df['class'].tolist()
        self.classes = classes
        self.labels_dict = {c: i+1 for i, c in enumerate(classes)}

    def extract_boxes_from_xml(self, filename):
        tree = ET.parse(filename) # load and parse the file
        root = tree.getroot() # get the root of the document

        # extract each bounding box
        boxes,labels = [],[]
        if len(root.findall('.//bndbox')) == 0:
            labels.append(0)
        else:
            for box in root.findall('object'):
                xmin = int(box.find('bndbox').find('xmin').text)
                ymin = int(box.find('bndbox').find('ymin').text)
                xmax = int(box.find('bndbox').find('xmax').text)
                ymax = int(box.find('bndbox').find('ymax').text)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.classes.index(box.find('name').text))
        return boxes,labels

    def __getitem__(self, idx):
        image_path = self.images[idx]

        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        org_h,org_w = img.shape[:2] # Original image height,width
        if self.im_size: img = cv2.resize(img.copy(),self.im_size)
        res_h,res_w = img.shape[:2] # Resized image height,width
        temp_boxes,labels = self.extract_boxes_from_xml(self.annot_paths[idx]) # get bboxes from xml
        boxes = []

        for i,box in enumerate(temp_boxes):
            if len(box) == 0:
                pass
            else:
                if self.im_size:
                    box[0] = int(box[0]/org_w*res_w)
                    box[2] = int(box[2]/org_w*res_w)
                    box[1] = int(box[1]/org_h*res_h)
                    box[3] = int(box[3]/org_h*res_h)
                    boxes.append([box[0],box[1],box[2],box[3]])
                else:
                    boxes.append([int(box[0]),int(box[1]),int(box[2]),int(box[3])])
        
        if self.transforms:
            aug = self.transforms(image = img, bboxes = [[0,0,20,20]] if len(boxes)==0 else boxes, category_ids = labels)
            img = aug['image']
            labels = [0] if len(aug['category_ids']) == 0 else aug['category_ids']
            if len(boxes)==0 or len(aug['bboxes'])==0:
                pass
            else:
                boxes = aug['bboxes']

        # convert boxes into a Torch Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes)==0:
            target = {'boxes': torch.zeros((0, 4), dtype=torch.float32),
                      'labels': torch.zeros((0,), dtype=torch.int64),
                      'image_id': torch.tensor([idx]),
                      'area': torch.zeros((0,), dtype=torch.float32),
                      'iscrowd': torch.zeros((0,), dtype=torch.int64)}
        else:
            target = {'boxes': boxes, 'labels': labels, 'image_id':torch.tensor([idx]), 
                      'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), 
                      'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64)}

        return T.Compose([T.ToTensor()])(img), target

    def __len__(self):
        return len(self.images)