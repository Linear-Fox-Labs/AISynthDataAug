# AISynthDataAug
AI-Driven Image Synthesis for Data Augmentation - research.linearfox.com

This project aims to develop an AI-driven image synthesis tool that leverages deep learning techniques to generate synthetic images for data augmentation, enhancing the performance of existing AI models in computer vision tasks.

## Installation
```
pip install .
```
"setup.py"

## Usage 

To generate synthetic images using the trained GAN model, run:

```
python scripts/generate_synthetic_images.py --model_path models/GAN/model.h5 --output_dir data/synthetic_dataset/images --num_images 1000
```

To augment the original dataset with the generated synthetic images, run:

```
python scripts/augment_dataset.py --original_dir data/original_dataset --synthetic_dir data/synthetic_dataset --output_dir data/augmented_dataset --num_augmentations 10
```

To train the GAN model on your own dataset, run:

```
python models/GAN/train.py --data_dir path/to/dataset --model_dir models/GAN --epochs 100 --batch_size 32
```

To train a CNN model on the augmented dataset, run:

```
python models/CNN/train.py --data_dir data/augmented_dataset --model_dir models/CNN --epochs 50 --batch_size 32
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.