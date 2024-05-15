import onnxruntime as ort
import cv2
import numpy as np
import os
import time
import subprocess
import argparse
from utils import rotate_small_angle, resizeAndPad

pdf_photo_model = ort.InferenceSession('pdf_photo.onnx')
pdf_photo_model_input_name = pdf_photo_model.get_inputs()[0].name
pdf_photo_model_output_name = pdf_photo_model.get_outputs()[0].name
print(pdf_photo_model_input_name, pdf_photo_model_output_name)

sar_model = ort.InferenceSession('sar.onnx')
sar_model_input_name = sar_model.get_inputs()[0].name
sar_model_output_name = sar_model.get_outputs()[0].name
print(sar_model_input_name, sar_model_output_name)

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

            result = pdf_photo_model.run([pdf_photo_model_output_name], {pdf_photo_model_input_name: np.array([img_to_process], dtype=np.float32)})

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

                    args = f'./test.exe -idir={img_path} -odir={unwarping_output}'
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
                result = sar_model.run([sar_model_output_name], {sar_model_input_name: np.array([rs_img], dtype=np.float32)})

                rotated = rotate_small_angle(unwarped_img, result[0])

                rotate_time = time.time() - start_time

                print('Rotatingtime: ', rotate_time)

                cv2.imwrite(os.path.join(output_path, image), rotated)
                    
                print('-' * 50)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script with input and output paths as arguments")
    parser.add_argument("--input_path", type=str, nargs='?', help="Path to the directory containing input images", default='example_input')
    parser.add_argument("--output_path", type=str, nargs='?', help="Path to the directory where output images will be saved", default='example_output')
    parser.add_argument("--cleanup", action='store_true', help="Delete the contents of the unwarping_output folder")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.cleanup)
