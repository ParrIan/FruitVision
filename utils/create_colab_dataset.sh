#!/bin/bash
# Create zip file of YOLO dataset for Google Colab

echo "Creating yolo_dataset.zip for Google Colab..."

cd /Users/ianparr/Personal\ Projects/Fruits/data

# Create zip file (this will take a few minutes)
zip -r yolo_dataset.zip yolo_dataset/

echo ""
echo "✓ Dataset zip created!"
echo "File location: $(pwd)/yolo_dataset.zip"
echo ""
echo "Next steps:"
echo "1. Upload yolo_dataset.zip to your Google Drive"
echo "2. Open train_yolo_colab.ipynb in Google Colab"
echo "3. Follow the notebook instructions"
