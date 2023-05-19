import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision
import numpy as np
from PIL import Image

class FastRCNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, im_size=None, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.im_size = im_size
        # load all image files, sorting them to ensure that they are aligned
        self.im_paths = df["image_path"].tolist()
        self.annot_paths = df["annot_path"].tolist()
        self.labels = df['class'].tolist()
        self.classes = ['BG'] + sorted(list(set(value for sublist in df['label'] for value in sublist)))
        self.labels_dict = {c: i+1 for i, c in enumerate(self.classes)}

    def extract_boxes_from_xml(self, filename):
        try:
            tree = ET.parse(filename) # load and parse the file
        except:
            return [],[0] # return emptry bbox and BG class if no xml exist
        root = tree.getroot() # get the root of the document

        # extract each bounding box
        boxes,labels = [],[]
        if len(root.findall('.//bndbox')) == 0:
            labels.append(0)
        else:
            for box in root.findall('object'):
                if box.find('name').text == 'OK DS NUT': continue # Skip some class while loading
                xmin = int(box.find('bndbox').find('xmin').text)
                ymin = int(box.find('bndbox').find('ymin').text)
                xmax = int(box.find('bndbox').find('xmax').text)
                ymax = int(box.find('bndbox').find('ymax').text)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.classes.index(box.find('name').text))
        return boxes,labels

    def __getitem__(self, idx):
        img = Image.open(self.im_paths[idx]).convert("RGB")
        org_w, org_h = img.size # Original image height,width
        if self.im_size: img = img.resize(self.im_size)
        res_w, res_h = img.size # Resized image height,width
        img = np.array(img, dtype=np.float32) / 255.0
        
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

        return torchvision.transforms.ToTensor()(img), target

    def __len__(self):
        return len(self.im_paths)