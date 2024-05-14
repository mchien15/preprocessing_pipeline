import tensorflow as tf
import cv2
import numpy as np
import os
import time
import subprocess
import argparse
from utils import rotate_small_angle, resizeAndPad

pdf_photo_model = tf.saved_model.load('pdf_vs_photo_v1/exported_pdf_vs_photo')

sar_model = tf.saved_model.load('small_rotation_v4/epoch-5-small-rotation/exported_small_rotation')

if not os.path.exists('unwarping_output'):
    os.makedirs('unwarping_output')
    # subprocess.run(['touch', 'unwarping_output/.gitkeep'])
    open('unwarping_output/.gitkeep', 'a').close()

unwarping_output = os.path.join(os.getcwd(), 'unwarping_output')
print(unwarping_output)

def main(input_path, output_path, cleanup):
    for image in os.listdir(input_path):
        try:
            if image == '.gitkeep':
                continue

            start_time = time.time()

            img_path = os.path.join(input_path, image)

            print('Processing image: ' + img_path)

            img = cv2.imread(img_path)

            classes = ['pdf', 'photo']

            img_to_process = cv2.resize(img, (480, 480))
            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)

            input_tensor = tf.convert_to_tensor(np.array([img_to_process]), dtype=tf.float32)

            result = pdf_photo_model.signatures['serving_default'](input_tensor)['dense_1'][0]

            print(result)
            print('Classified as class: ' + classes[np.argmax(result)])

            end_time = time.time()

            pdf_photo_time = end_time - start_time

            print('Classification time: ', pdf_photo_time)


            if classes[np.argmax(result)] == 'pdf':
                cv2.imwrite(os.path.join(output_path, image), img)
                print('-' * 50)
                continue
            else:
                # check if current OS is Windows
                if os.name == 'nt':
                    start_time = time.time()
                    if os.access(os.path.abspath("test.exe"), os.X_OK) == False:
                        print('test.exe is not executable')
                    args = f'{os.path.abspath("test.exe")} -idir={img_path} -odir={unwarping_output}'
                    subprocess.call(args, shell=True)

                    end_time = time.time()

                    unwarping_time = end_time - start_time

                    print('Unwarping time: ', unwarping_time)
                else:
                    start_time = time.time()

                    args = f'./test_fix_path -idir={img_path} -odir={unwarping_output}'
                    subprocess.call(args, shell=True)

                    end_time = time.time()

                    unwarping_time = end_time - start_time

                    print('Unwarping time: ', unwarping_time)

                # unwarped_img = cv2.imread(unwarping_output + '/' + image.split(".")[0] + '_remap.png')

                temp_unwarped_img_path = os.path.join(unwarping_output, image.split(".")[0]) + '_remap.png'
                # print(temp_unwarped_img_path)

                unwarped_img = cv2.imread(temp_unwarped_img_path)

                if cleanup:
                    os.remove(temp_unwarped_img_path)

                start_time = time.time()

                rs_img = resizeAndPad(unwarped_img, (480, 480))
                rs_img = cv2.cvtColor(rs_img, cv2.COLOR_BGR2RGB)
                rs_img = rs_img / 255.0
                print(rs_img.shape)
                input_tensor = tf.convert_to_tensor(np.array([rs_img]), dtype=tf.float32)

                result = sar_model.signatures['serving_default'](input_tensor)['output_0'][0]

                rotated = rotate_small_angle(unwarped_img, result[0])

                rotate_time = time.time() - start_time

                print('Rotatingtime: ', rotate_time)

                cv2.imwrite(os.path.join(output_path, image), rotated)
                    
                print('-' * 50)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script with input and output paths as arguments")
    parser.add_argument("--input_path", type=str, nargs='?', help="Path to the directory containing input images", default='/home/asus/stuDYING/IT/Thesis/from_server/image1')
    parser.add_argument("--output_path", type=str, nargs='?', help="Path to the directory where output images will be saved", default='/home/asus/stuDYING/IT/Thesis/preprocessing_pipeline/out_file/image1')
    parser.add_argument("--cleanup", action='store_true', help="Delete the contents of the unwarping_output folder")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.cleanup)
