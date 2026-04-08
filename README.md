# TypoChroma

Typo-Chroma is a dataset for studying object-level color recognition in vision-language models under conflicting multimodal cues. It consists of color-verified object instances with segmentation masks, paired with typographic attack images that introduce contradictory textual signals.

## Overview

- 4,360 color-verified object instances
- Instance-level segmentation masks (from LVIS)
- 4,333 attack images generated via a diffusion-based pipeline
- 4,170 attack images verified to preserve original object color
- 2,225 high-quality attack images with perfect typography (OCR score = 1.0)

Each instance is annotated with:
- object category
- segmentation mask
- verified object color
- attack metadata

# Data Format
This repository provides **attack images only**. The corresponding clean images are **not included** and must be obtained separately from the official COCO and LVIS releases. [train2017 partition,https://cocodataset.org/#download] [LVIS v1.0 train partition, https://www.lvisdataset.org/dataset]

Each instance can be matched to its original image using the provided `image_id` and `annotation_id` fields.

# Data Usage Notice

This dataset is constructed from images in the LVIS v1.0 dataset, which is built on the COCO dataset.

Due to licensing restrictions, we do **not** redistribute the original images. Attack images provided in this repository are **derived from COCO/LVIS images**.

- These images are provided for research purposes only.
- They may retain substantial similarity to the original images.
- Users are responsible for ensuring compliance with the licenses of COCO and LVIS.

We do not claim ownership of the original image content.

If you are a rights holder and have concerns, please contact us.
