# TextExtraction_From_Image

In this repository, I am using two approaches to extract text from images. The methods employed are:

1. **Detectron2 Layout Model**: Utilizes Faster R-CNN R-50 FPN on PubLayNet to get layout information.
2. **LayoutLMV3**: Another model used to acquire layout information about the text in the image.

After obtaining the layout information, the text is extracted using PyTesseract.
