# Preprocessing Module (skip pdf images + unwarping + small rotation)

## How to use
### Install required packages
```
pip install -r requirements.txt
```
### Preprocessing multiple images in a folder
<!-- 
#### Option 1: Use `inference.ipynb` file -->

#### Run `inference.py` file with example images:
```
python inference.py
```
Output images will be stored in `example_output` folder

#### Or you can test with your own images by using:
```
python inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder
```

The unwarped images will be stored in `unwarping_output` folder \
The final output images will be stored in your custom output folder

---
**NOTE**

Running the program multiple times will quickly fill up the `unwarping_output` folder with images. If you don't wish to keep the newly unwarped image, you can run `inference.py` file with `--cleanup` flag. For example:
```
python inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder --cleanup
```

---