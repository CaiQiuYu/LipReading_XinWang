import sys
import glob
import os
from tqdm import tqdm
import argparse
from yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo):
    base_dir = 'lip_test'
    file_list = os.listdir(base_dir)
    for fi in tqdm(file_list):
        name_list = glob.glob(base_dir + '/' + fi + '/*.png')
        # img = input('Input image filename:')
        for name in name_list:
            try:
                image = Image.open(name)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, name)


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, default='logs/ep079-loss4.391-val_loss4.269.h5',
        help='path to model weight file, default '
    )

    parser.add_argument(
        '--anchors', type=str, default='model_data/yolo_anchors.txt',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, default='model_data/classes.txt',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int, default=1,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
