import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.eyeDataset import dataloader_dual

from utils.metrics import evaluate_segmentation
from utils.trainer_utils import Performance_Metrics
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 4 #19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'snapshots/modelSS_60000.pth' # 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    # if args.restore_from[:4] == 'http' :
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)

    model.eval()

    # testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
    #                                 batch_size=1, shuffle=False, pin_memory=True)
    testset_openeds, testset_calipso, testloader_openeds, testloader = \
        dataloader_dual(
            openeds_root="/home/yirus/Datasets/OpenEDS_SS_TL/",
            calipso_root="/home/yirus/Datasets/Calipso/GT_0.25/",
            batch_size=1,
            type="test",
        )

    interp = nn.Upsample(size=(184, 184), mode='bilinear', align_corners=True)

    Global_Accuracy = []
    Class_Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    IOU = []
    pm = Performance_Metrics(
        Global_Accuracy,
        Class_Accuracy,
        Precision,
        Recall,
        F1,
        IOU,
    )

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, label = batch
        image = image.to(device)

        with torch.set_grad_enabled(False):
            if args.model == 'DeeplabMulti':
                output1, output2 = model(image)
                output = interp(output2).cpu().data[0].numpy()
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(image)
                output = interp(output).cpu().data[0].numpy()

        image = image.detach().cpu().numpy()
        label = np.squeeze(label.detach().cpu().numpy(), axis=0)
        pred = np.argmax(output, axis=0)

        if index == 0:
            collate_image = image.copy()
            collate_preds = pred.copy()
            collate_labels = label.copy()
        else:
            collate_image = np.vstack([collate_image, image.copy()])
            collate_preds = np.vstack([collate_preds, pred.copy()])
            collate_labels = np.vstack([collate_labels, label.copy()])

        for idx in np.arange(pred.shape[0]):
            ga, ca, prec, rec, f1, iou = evaluate_segmentation(
                pred[idx, :],
                label[idx, :],
                NUM_CLASSES,
            )
            pm.GA.append(ga)
            pm.CA.append(ca)
            pm.Precision.append(prec)
            pm.Recall.append(rec)
            pm.F1.append(f1)
            pm.IOU.append(iou)

    if True:
        return pm, collate_image, collate_preds, collate_labels
    else:
        return pm

        # output = output.transpose(1,2,0)
        # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        #
        # output_col = colorize_mask(output)
        # output = Image.fromarray(output)
        #
        # # name = name[0].split('/')[-1]
        # name = "pred_image_{}".format(index)
        # output.save('%s/%s' % (args.save, name))
        # output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))


if __name__ == '__main__':
    pm_test, data_test, preds_test, acts_test = main()
    print('Global Mean Accuracy:', np.array(pm_test.GA).mean())
    print('Mean IOU:', np.array(pm_test.IOU).mean())
    print('Mean Recall:', np.array(pm_test.Recall).mean())
    print('Mean Precision:', np.array(pm_test.Precision).mean())
    print('Mean F1:', np.array(pm_test.F1).mean())