#This script calulates accuracy and other performance merics for two binary images
import numpy as np
import os
import cv2 as cv
from os import listdir
from os.path import isfile, join
import sys, getopt

def calulate_metrics(gt_dir, test_dir):
    gt_images = [f for f in listdir(gt_dir) if isfile(join(gt_dir,f))]
    gt_images_path = [os.path.join(gt_dir, f) for f in gt_images]
    test_images = [f for f in listdir(test_dir) if isfile(join(test_dir,f))]
    test_images_path = [os.path.join(test_dir,f) for f in test_images]

    combined_paths =  list(zip(gt_images_path, test_images_path))
    total_images = len(combined_paths)
    FN = 0
    FP = 0
    TP = 0
    TN = 0

    for i in combined_paths:
        image_gt = cv.imread(i[0],0)
        image_predicted = cv.imread(i[1], 0)

        ground_truth = image_gt > 0
        test_data = image_predicted > 0

        #Calculate False Positve for single image
        false_positives = ~ground_truth & test_data
        sum_false_positives = np.sum(false_positives)

        #Calculate False Negetives for single image
        false_negetives =  ground_truth & ~test_data
        sum_false_negetives = np.sum(false_negetives)

        #Calculate True Positives for single image
        true_positives = ground_truth & test_data
        sum_true_positives = np.sum(true_positives)

        #Calculate True Negetives for single image
        true_negetives = ~ground_truth & ~ test_data
        sum_true_negetives = np.sum(true_negetives)

        FP += sum_false_positives
        FN += sum_false_negetives
        TP += sum_true_positives
        TN += sum_true_negetives

    FP /= total_images
    FN /= total_images
    TP /= total_images
    TN /= total_images
    accuracy  = (TP + TN)/(TP + TN + FP + FN)
    # precision  = (TP)/(TP + FP)
    # recall = TP/(TP + FN)
    # f_measure = (2*precision*recall)/(precision +  recall)
    # print(precision)
    # print(recall)
    # print(f_measure)
    print(accuracy)


def main(argv):
    ground_truth_dir = ""
    test_image_dir = ""
    try:
        opts, args = getopt.getopt(argv, "hg:t:", ["ground_truth_dir=","predicted_image_dir="])
    except getopt.GetoptError:
        print("python3  calc_metrics.py -g <ground_truth_dir>  -t <predicted_img_dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 calc_metrics.py -gt <ground_truth_dir> -test <predicted_img_dir>")
            sys.exit()
        elif opt in ("-g", "--ground_truth_dir"):
            ground_truth_dir = arg
        elif  opt in ("-t", "--predicted_image_dir"):
            test_image_dir = arg
    calulate_metrics(ground_truth_dir, test_image_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

