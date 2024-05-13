# Preprocessing Module (skip pdf images + unwarping + small rotation)

Spam check and corner detection will be updated soon

## How to use
### Install required packages
```
pip install -r requirements.txt
```
### Preprocessing multiple images in a folder

#### Option 1: Use `inference.ipynb` file

#### Option 2: Run `inference.py` file using this command:

```
python inference.py /path/to/your/input/folder /path/to/your/output/folder
```

Running this module using Python script is actually a lot faster compared with using the IPython Notebook

The unwarped image will be stored in `unwarping_output` folder \
The final output image will be stored in your custom output folder