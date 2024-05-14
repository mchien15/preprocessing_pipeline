# Preprocessing Module (skip pdf images + unwarping + small rotation)

Spam check and corner detection will be updated soon

## How to use
### Install required packages
```
pip install -r requirements.txt
```
### Preprocessing multiple images in a folder
<!-- 
#### Option 1: Use `inference.ipynb` file -->

#### Run `inference.py` file using this command:

```
python inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder
```

Running this module using Python script is actually a lot faster compared with using the IPython Notebook

The unwarped image will be stored in `unwarping_output` folder \
The final output image will be stored in your custom output folder

---
**NOTE**

If you run the program multiple times, the `unwarping_output` folder will quickly fill up with images. If you don't wish to keep the newly unwarped image, you can run `inference.py` file with `--cleanup` flag. For example:
```
python inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder --cleanup
```

---