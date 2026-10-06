---
title: "LGSFNet: A Local–Global Semantic Fusion Network for Robust Cross-Domain Deepfake Detection"
collection: publications
permalink: /publication/2025-11-25-lgsfnet-deepfake-detection
excerpt: 'LGSFNet introduces a dual-path architecture combining Spatial Resolution Adapters (SRA) and Local Semantic Fusion Adapters (LSFA) with DINOv3 transformer backbones for robust cross-domain facial deepfake detection.'
date: 2025-11-25
venue: 'BMVC Workshops (MAAAI 2025), British Machine Vision Association'
paperurl: 'https://bmva-archive.org.uk/bmvc/2025/assets/workshops/MAAAI/Paper_11/paper.pdf'
citation: 'Zulkaif Sajjad, Junaid Mir, Shah Nawaz, Furqan Shaukat, and Muhammad Haroon Yousaf. "LGSFNet: A Local–Global Semantic Fusion Network for Robust Cross-Domain Deepfake Detection." In <i>BMVC Workshops (MAAAI)</i>, 2025.'
---

[Download / View Paper Here](https://bmva-archive.org.uk/bmvc/2025/assets/workshops/MAAAI/Paper_11/paper.pdf)

## Abstract
Deepfakes, enabled by recent advances in generative models, pose significant ethical, societal, and security risks. Although many detection methods achieve strong intra-dataset performance, they often degrade on low-quality or cross-domain data due to compression artifacts and unseen manipulations. 

To address this, we introduce LGSFNet, a robust deepfake detection framework that fuses local and global forgery semantics in a dual-path architecture. The design integrates a Spatial Resolution Adapter (SRA) to extract local low-level features and a novel Local Semantic Fusion Adapter (LSFA) to inject these cues into the DINOv3 transformer backbone for multi-stage feature fusion with parameter-efficient training. Experiments on FaceForensics++ demonstrate state-of-the-art results across all four manipulation types, achieving up to 99.98% AUC. Cross-corpora evaluations on Celeb-DF, DFD, and DFDC further highlight strong generalization.

Recommended citation: Zulkaif Sajjad, Junaid Mir, Shah Nawaz, Furqan Shaukat, and Muhammad Haroon Yousaf. "LGSFNet: A Local–Global Semantic Fusion Network for Robust Cross-Domain Deepfake Detection." In <i>BMVC Workshops (MAAAI)</i>, 2025.


[Project Code / Repository](https://github.com/zulkaifsajjad/LGSFNet)