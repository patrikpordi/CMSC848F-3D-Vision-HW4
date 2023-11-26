import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_x, rotate_y, rotate_z
import os
import csv


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Rotation arguments
    parser.add_argument('--Rotation_XYZ', type=float, nargs=3, default=None, help='Amount of rotation around x, y, and z axes')
    
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    # Directories and checkpoint/sample iterations
    parser.add_argument('--num_workers', type=int, default=12, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

def rotate(data_b, args):
    rotated_data = data_b.clone()
    x_angle, y_angle, z_angle = args.Rotation_XYZ
    rotated_data = rotate_x(rotated_data, x_angle, args)
    rotated_data = rotate_y(rotated_data, y_angle, args)
    rotated_data = rotate_z(rotated_data, z_angle, args)

    return rotated_data


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    directory = args.output_dir+'/seg'
    create_dir(directory)
    directory = directory + '/'
    # print(args.num_points)
    if args.num_points != 10000:
        directory = directory+f'num_points_{args.num_points}'
        create_dir(directory)
        directory = directory + '/'
        num_points = args.num_points
    else:
        num_points = 10000
    

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model()
    model.to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval().to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))


    dataloader_seg = get_data_loader(args=args, train=False)

    correct = 0
    num_obj = 0
    accuracies = []
    rotation = args.Rotation_XYZ

    if rotation:
        print("Rotated data")
    else:
        print("Normal data")

    for i, batch in enumerate(dataloader_seg):
        data_b, data_b_label = batch
        data_b = data_b.to(args.device)
        data_b_label = data_b_label.to(args.device)

        selected_indices = torch.randint(0, data_b.shape[1], (num_points,))
        data_b = data_b[:, selected_indices, :]
        data_b_label = data_b_label[:, selected_indices]

        if rotation:
            # ------ TO DO: Rotate the input point cloud ------
            
            output_rot_dir = f'Rotation_XYZ_{args.Rotation_XYZ[0]}_{args.Rotation_XYZ[1]}_{args.Rotation_XYZ[2]}/'
            rotated_data = rotate(data_b, args)
            create_dir(directory + output_rot_dir)
            
            # ------ TO DO: Make Prediction ------
            with torch.no_grad():
                pred_label = model(rotated_data.to(args.device))
                pred_label = pred_label.max(dim=-1)[1]
            correct += pred_label.eq(data_b_label.data).cpu().sum().item()
            num_obj += data_b_label.view([-1,1]).size()[0]
            accuracy = pred_label.eq(data_b_label.data).float().mean(dim=1)
            accuracies.extend(accuracy.cpu().numpy())

            for idx, acc in enumerate(accuracy):
                # print(f'Object {i * args.batch_size + idx}: Accuracy {acc.item()}')
                # Visualizations for objects with accuracy >= 50%
                if acc.item() >= 0.5:
                    if rotation:
                        success_dir = directory + output_rot_dir + 'success/'
                    else:
                        success_dir = directory  + 'success/'
                    create_dir(success_dir)
                    viz_seg(rotated_data[idx], pred_label[idx], os.path.join(success_dir, f"pred_{i}_{idx}.gif"), args.device, args)
                    viz_seg(rotated_data[idx], data_b_label[idx], os.path.join(success_dir, f"gt_{i}_{idx}.gif"), args.device, args)
                # Visualizations for objects with accuracy < 50%
                else:
                    if rotation:
                        fail_dir = directory + output_rot_dir + 'fail/'
                    else:
                        fail_dir = directory +  'fail/'
                    create_dir(fail_dir)
                    viz_seg(rotated_data[idx], pred_label[idx], os.path.join(fail_dir, f"pred_{i}_{idx}.gif"), args.device, args)
                    viz_seg(rotated_data[idx], data_b_label[idx], os.path.join(fail_dir, f"gt_{i}_{idx}.gif"), args.device, args)
                    
        else:
            with torch.no_grad():
                pred_label = model(data_b.to(args.device))
                pred_label = pred_label.max(dim=-1)[1]
            correct += pred_label.eq(data_b_label.data).cpu().sum().item()

            num_obj += data_b_label.view([-1, 1]).size()[0]

            # Calculate and print accuracy for each object
            accuracy = pred_label.eq(data_b_label.data).float().mean(dim=1)
            accuracies.extend(accuracy.cpu().numpy())
            for idx, acc in enumerate(accuracy):
                # print(f'Object {i * args.batch_size + idx}: Accuracy {acc.item()}')
                # Visualizations for objects with accuracy > 50%
                if acc.item() >= 0.5:
                    if rotation:
                        success_dir = directory + output_rot_dir + 'success/'
                    else:
                    # Visualizations for objects with accuracy > 50%
                        success_dir = directory  + 'success/'
                    create_dir(success_dir)
                    viz_seg(data_b[idx], pred_label[idx], success_dir+ f"pred_{i}_{idx}.gif", args.device, args)
                    viz_seg(data_b[idx], data_b_label[idx], success_dir+ f"gt_{i}_{idx}.gif", args.device, args)
                else:
                    if rotation:
                        fail_dir = directory + output_rot_dir + 'fail/'
                    else:
                        fail_dir = directory + 'fail/'
                    create_dir(fail_dir)
                    
                    viz_seg(data_b[idx], pred_label[idx], fail_dir+ f"pred_{i}_{idx}.gif", args.device, args)
                    viz_seg(data_b[idx], data_b_label[idx], fail_dir+ f"gt_{i}_{idx}.gif", args.device, args)


    test_accuracy = correct / num_obj
    print ("Test accuracy: {}".format(test_accuracy))
    if rotation:
        csv_file_path = directory + output_rot_dir +'acc.csv'
    else:
        csv_file_path = directory  +'acc.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Object ID', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, acc in enumerate(accuracies):
            writer.writerow({'Object ID': i, 'Accuracy': acc})

    print(f'Accuracies saved to {csv_file_path}')
