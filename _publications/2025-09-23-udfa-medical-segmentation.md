---
title: "U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation"
collection: publications
permalink: /publication/2025-09-23-udfa-medical-segmentation
excerpt: 'We propose U-DFA, a unified DINOv2-Unet architecture integrating a novel Local-Global Fusion Adapter (LGFA) to effectively combine spatial CNN features with foundation model vision transformers for multi-dataset medical image segmentation.'
date: 2025-09-23
venue: 'International Workshop on Machine Learning in Medical Imaging (MLMI / MICCAI 2025), Springer Nature'
paperurl: 'https://link.springer.com/chapter/10.1007/978-3-032-09513-8_19'
citation: 'Zulkaif Sajjad, Furqan Shaukat, and Junaid Mir. "U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation." In <i>International Workshop on Machine Learning in Medical Imaging (MLMI 2025)</i>, Lecture Notes in Computer Science, vol 16008, pp. 192-201. Springer Nature Switzerland, 2025.'
---

[Download / View Paper Here](https://link.springer.com/chapter/10.1007/978-3-032-09513-8_19)

## Abstract
Accurate medical image segmentation plays a crucial role in overall diagnosis and is one of the most essential tasks in the diagnostic pipeline. CNN-based models, despite their extensive use, suffer from a local receptive field and fail to capture the global context. A common approach that combines CNNs with transformers attempts to bridge this gap but fails to effectively fuse the local and global features. With the recent emergence of VLMs and foundation models, they have been adapted for downstream medical imaging tasks; however, they suffer from an inherent domain gap and high computational cost. 

To this end, we propose U-DFA, a unified DINOv2-Unet encoder-decoder architecture that integrates a novel Local-Global Fusion Adapter (LGFA) to enhance segmentation performance. LGFA modules inject spatial features from a CNN-based Spatial Pattern Adapter (SPA) module into frozen DINOv2 blocks at multiple stages, enabling effective fusion of high-level semantic and spatial features. Our method achieves state-of-the-art performance on the Synapse and ACDC datasets with only 33% of the trainable model parameters.

Recommended citation: Zulkaif Sajjad, Furqan Shaukat, and Junaid Mir. "U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation." In <i>International Workshop on Machine Learning in Medical Imaging (MLMI 2025)</i>, Lecture Notes in Computer Science, vol 16008, pp. 192-201. Springer Nature Switzerland, 2025.


[Project Code / Repository](https://github.com/zulkaifsajjad/U-DFA)