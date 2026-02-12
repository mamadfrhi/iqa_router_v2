# NR-IQA Valuable Papers (Top 30)

## 1. Decoupling Perception and Calibration: Label-Efficient Image Quality Assessment Framework

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; AVA
- Best metrics (auto-parsed): SRCC 0.841, PLCC 0.899
- arXiv: https://arxiv.org/abs/2601.20689
- PDF: https://arxiv.org/pdf/2601.20689.pdf
- Summary: Recent multimodal large language models (MLLMs) have demonstrated strong capabilities in image quality assessment (IQA) tasks. However, adapting such large-scale models is computation- ally expensive and still relies on substantial Mean Opinion Score (MOS) annotations. We argue that for MLLM-based IQA, the core bottleneck lies not in the quality perception capacity of MLLMs, but in MOS scale calibration.

## 2. Q-Probe: Scaling Image Quality Assessment to High Resolution via Context-Aware Agentic Probing

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; KADID-10k
- Best metrics (auto-parsed): SRCC 0.728, PLCC 0.776
- arXiv: https://arxiv.org/abs/2601.15356
- PDF: https://arxiv.org/pdf/2601.15356.pdf
- Summary: Reinforcement Learning (RL) has empowered Multimodal Large Language Models (MLLMs) to achieve superior human preference alignment in Image Quality Assessment (IQA). However, existing RL-based IQA models typically rely on coarse-grained global views, failing to cap- ture subtle local degradations in high-resolution scenarios. While emerging “Thinking with Im- ages” paradigms enable multi-scale visual per- 1University of Science and Technology of China, Hefei, China 2Hefei University of Technology, Hefei, China3Institute of Intelli- gent Machines, Chinese Academy of Sciences, Hefei, China.

## 3. Understanding Pure Textual Reasoning for Blind Image Quality Assessment

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; AVA
- Best metrics (auto-parsed): SRCC 0.03
- arXiv: https://arxiv.org/abs/2601.02441
- PDF: https://arxiv.org/pdf/2601.02441.pdf
- Summary: —Textual reasoning has recently been widely adopted in Blind Image Quality Assessment (BIQA). However, it remains unclear how textual information contributes to quality prediction and to what extent text can represent the score-related image con- tents. This work addresses these questions from an information- flow perspective by comparing existing BIQA models with three paradigms designed to learn the image–text–score relationship: Chain-of-Thought, Self-Consistency, and Autoencoder.

## 4. Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models

- Venue/Year: ICCV 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; TID2013; AVA
- Best metrics (auto-parsed): PLCC 0.858
- CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.html
- PDF: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.pdf
- Summary: Image Quality Assessment (IQA) remains an unresolved challenge in computer vision due to complex distortions, di- verse image content, and limited data availability. Existing Blind IQA (BIQA) methods largely rely on extensive human annotations, which are labor-intensive and costly due to the demanding nature of creating IQA datasets. To reduce this dependency, we propose the Gradient-Regulated Meta- Prompt IQA Framework (GRMP-IQA), designed to effi- ciently adapt the visual-language pre-trained model, CLIP , to IQA tasks, achieving high accuracy even with limited data.

## 5. 3DGS-IEval-15K: A Large-scale Image Quality Evaluation Database for 3D Gaussian-Splatting

- Venue/Year: arXiv 2025
- Datasets: LIVE; AVA
- Best metrics (auto-parsed): SRCC 0.93
- arXiv: https://arxiv.org/abs/2506.14642
- PDF: https://arxiv.org/pdf/2506.14642.pdf
- Summary: 3D Gaussian Splatting (3DGS) has emerged as a promising ap- proach for novel view synthesis, offering real-time rendering with high visual fidelity. However, its substantial storage requirements present significant challenges for practical applications. While re- cent state-of-the-art (SOTA) 3DGS methods increasingly incorpo- rate dedicated compression modules, there is a lack of a comprehen- sive framework to evaluate their perceptual impact.

## 6. BELE: Blur Equivalent Linearized Estimator

- Venue/Year: arXiv 2025
- Datasets: KADID-10k; LIVE; CSIQ; TID2013
- Best metrics (auto-parsed): SRCC 0.9548, PLCC 0.9513
- arXiv: https://arxiv.org/abs/2503.00503
- PDF: https://arxiv.org/pdf/2503.00503.pdf
- Summary: —In the Full-Reference Image Quality Assessment context, Mean Opinion Score values represent subjective eval- uations based on retinal perception, while objective metrics assess the reproduced image on the display. Bridging these subjective and objective domains requires parametric mapping functions, which are sensitive to the observer’s viewing distance. This paper introduces a novel parametric model that separates perceptual effects due to strong edge degradations from those caused by texture distortions.

## 7. BPCLIP: A Bottom-up Image Quality Assessment from Distortion to Semantics Based on CLIP

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; CLIVE; LIVE; CSIQ; TID2013; AVA
- Best metrics (auto-parsed): SRCC 0.865, PLCC 0.953
- arXiv: https://arxiv.org/abs/2506.17969
- PDF: https://arxiv.org/pdf/2506.17969.pdf
- Summary: —Image Quality Assessment (IQA) aims to evaluate the perceptual quality of images based on human subjective per- ception. Existing methods generally combine multiscale features to achieve high performance, but most rely on straightforward linear fusion of these features, which may not adequately capture the impact of distortions on semantic content. To address this, we propose a bottom-up image quality assessment approach based on the Contrastive Language-Image Pre-training (CLIP, a recently proposed model that aligns images and text in a shared feature space), named BPCLIP, which progressively ex- tracts the impact of low-level distortions on high-level semantics.

## 8. Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; CLIVE; LIVE; CSIQ; TID2013; AVA
- Best metrics (auto-parsed): SRCC 0.765, PLCC 0.706
- arXiv: https://arxiv.org/abs/2511.09948
- PDF: https://arxiv.org/pdf/2511.09948.pdf
- Summary: Recent efforts have repurposed the Contrastive Language- Image Pre-training (CLIP) model for No-Reference Image Quality Assessment (NR-IQA) by measuring the cosine sim- ilarity between the image embedding and textual prompts such as “a good photo” or “a bad photo.” However, this se- mantic similarity overlooks a critical yet underexplored cue: the magnitude of the CLIP image features, which we em- pirically find to exhibit a strong correlation with percep- tual quality.In this work, we introduce a novel adaptive fu- sion framework that complements cosine similarity with a magnitude-aware quality cue. Specifically, we first extract the absolute CLIP image features and apply a Box-Cox transfor- mation to statistically normalize the feature distribution and mitigate semantic sensitivity. The 

## 9. Burst Image Quality Assessment: A New Benchmark and Unified Framework for Multiple Downstream Tasks

- Venue/Year: arXiv 2025
- Datasets: SPAQ
- Best metrics (auto-parsed): SRCC 0.343
- arXiv: https://arxiv.org/abs/2511.07958
- PDF: https://arxiv.org/pdf/2511.07958.pdf
- Summary: In recent years, the development of burst imaging technology has improved the capture and processing capabilities of visual data, enabling a wide range of applications. However, the re- dundancy in burst images leads to the increased storage and transmission demands, as well as reduced efficiency of down- stream tasks. To address this, we propose a new task of Burst Image Quality Assessment (BuIQA), to evaluate the task- driven quality of each frame within a burst sequence, provid- ing reasonable cues for burst image selection.

## 10. DEFNet: Multitasks-based Deep Evidential Fusion Network for Blind Image Quality Assessment

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; TID2013
- Best metrics (auto-parsed): SRCC 0.828
- arXiv: https://arxiv.org/abs/2507.19418
- PDF: https://arxiv.org/pdf/2507.19418.pdf
- Summary: Blind image quality assessment (BIQA) methods often in- corporate auxiliary tasks to improve performance. How- ever, existing approaches face limitations due to insuffi- cient integration and a lack of flexible uncertainty estima- tion, leading to suboptimal performance. To address these challenges, we propose a multitasks-based DeepEvidential Fusion Network (DEFNet) for BIQA, which performs multi- task optimization with the assistance of scene and distortion type classification tasks.

## 11. Describe-to-Score: Text-Guided Efficient Image Complexity Assessment

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; KADID-10k; LIVE; TID2013; AVA
- Best metrics (auto-parsed): SRCC 0.959, PLCC 0.9576
- arXiv: https://arxiv.org/abs/2509.16609
- PDF: https://arxiv.org/pdf/2509.16609.pdf
- Summary: Accurately assessing image complexity (IC) is critical for computer vision, yet most existing methods rely solely on visual features and often neglect high-level semantic information, limiting their accuracy and generalization. We introduce vision-text fusion for IC modeling. This approach integrates visual and textual semantic features, increasing representational diversity.

## 12. DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; CLIVE; LIVE; CSIQ; AVA
- Best metrics (auto-parsed): SRCC 0.9086, PLCC 0.8999
- arXiv: https://arxiv.org/abs/2509.17012
- PDF: https://arxiv.org/pdf/2509.17012.pdf
- Summary: —Document image quality assessment (DIQA) is an important component for various applications, including optical character recognition (OCR), document restoration, and the evaluation of document image processing systems. In this paper, we introduce a subjective DIQA dataset DIQA-5000. The DIQA- 5000 dataset comprises 5,000 document images, generated by applying multiple document enhancement techniques to 500 real- world images with diverse distortions.

## 13. Don't Judge Before You CLIP: A Unified Approach for Perceptual Tasks

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; TID2013
- Best metrics (auto-parsed): SRCC 0.943
- arXiv: https://arxiv.org/abs/2503.13260
- PDF: https://arxiv.org/pdf/2503.13260.pdf
- Summary: Visual perceptual tasks aim to predict human judgment of images (e.g., emotions invoked by images, image qual- ity assessment). Unlike objective tasks such as object/scene recognition, perceptual tasks rely on subjective human as- sessments, making its data-labeling difficult. The scarcity of such human-annotated data results in small datasets lead- ing to poor generalization.

## 14. Fine-Grained HDR Image Quality Assessment From Noticeably Distorted to Very High Fidelity

- Venue/Year: arXiv 2025
- Datasets: LIVE; AVA
- Best metrics (auto-parsed): PLCC 0.587
- arXiv: https://arxiv.org/abs/2506.12505
- PDF: https://arxiv.org/pdf/2506.12505.pdf
- Summary: —High dynamic range (HDR) and wide color gamut (WCG) technologies significantly improve color reproduction compared to standard dynamic range (SDR) and standard color gamuts, resulting in more accurate, richer, and more immersive images. However, HDR increases data demands, posing challenges for bandwidth efficiency and compression techniques. Advances in compression and display technologies require more precise image quality assessment, particularly in the high-fidelity range where perceptual differences are subtle.

## 15. Fine-grained Image Quality Assessment for Perceptual Image Restoration

- Venue/Year: arXiv 2025
- Datasets: KADID-10k; AVA
- Best metrics (auto-parsed): SRCC 0.323, PLCC 0.280
- arXiv: https://arxiv.org/abs/2508.14475
- PDF: https://arxiv.org/pdf/2508.14475.pdf
- Summary: Recent years have witnessed remarkable achievements in perceptual image restoration (IR), creating an urgent de- mand for accurate image quality assessment (IQA), which is essential for both performance comparison and algorithm optimization. Unfortunately, the existing IQA metrics ex- hibit inherent weakness for IR task, particularly when dis- tinguishing fine-grained quality differences among restored images. To address this dilemma, we contribute the first-of- its-kind fine-grained image quality assessment dataset for im- age restoration, termedFGRestore, comprising 18,408 re- stored images across six common IR tasks.

## 16. Guiding Perception-Reasoning Closer to Human in Blind Image Quality Assessment

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; AVA
- Best metrics (auto-parsed): PLCC 0.041
- arXiv: https://arxiv.org/abs/2512.16484
- PDF: https://arxiv.org/pdf/2512.16484.pdf
- Summary: Humans assess image quality through a perception- reasoning cascade, integrating sensory cues with implicit reasoning to form self-consistent judgments. In this work, we investigate how a model can acquire both human-like and self-consistent reasoning capability for blind image quality assessment (BIQA). We first collect human evalua- tion data that capture several aspects of human perception- reasoning pipeline.

## 17. Bridging the Synthetic-to-Authentic Gap: Distortion-Guided Unsupervised Domain Adaptation for Blind Image Quality Assessment

- Venue/Year: CVPR 2024
- Datasets: KonIQ-10k; KADID-10k; LIVE; AVA
- Best metrics (auto-parsed): SRCC 0.5646, PLCC 0.6958
- CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Bridging_the_Synthetic-to-Authentic_Gap_Distortion-Guided_Unsupervised_Domain_Adaptation_for_Blind_CVPR_2024_paper.html
- PDF: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Bridging_the_Synthetic-to-Authentic_Gap_Distortion-Guided_Unsupervised_Domain_Adaptation_for_Blind_CVPR_2024_paper.pdf
- Summary: The annotation of blind image quality assessment (BIQA) is labor-intensive and time-consuming, especially for authentic images. Training on synthetic data is expected to be beneficial, but synthetically trained models often suf- fer from poor generalization in real domains due to do- main gaps. In this work, we make a key observation that introducing more distortion types in the synthetic dataset may not improve or even be harmful to generalizing au- thentic image quality assessment.

## 18. Application of Ideal Observer for Thresholded Data in Search Task

- Venue/Year: arXiv 2026
- Datasets: AVA
- arXiv: https://arxiv.org/abs/2601.07976
- PDF: https://arxiv.org/pdf/2601.07976.pdf
- Summary: This study advances task-based image quality assessment by developing an anthropomorphic thresh- olded visual-search model observer. The model is an ideal observer for thresholded data inspired by the human visual system, allowing selective processing of high-salience features to improve discrimination performance. By filtering out irrelevant variability, the model enhances diagnostic accuracy and computational efficiency.

## 19. ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration

- Venue/Year: arXiv 2026
- Datasets: AVA
- arXiv: https://arxiv.org/abs/2601.02763
- PDF: https://arxiv.org/pdf/2601.02763.pdf
- Summary: All-in-One Image Restoration (AiOIR) has advanced signif- icantly, offering promising solutions for complex real-world degradations. However, most existing approaches rely heav- ily on degradation-specific representations, often resulting in oversmoothing and artifacts. To address this, we pro- pose ClearAIR, a novel AiOIR framework inspired by Hu- man Visual Perception (HVP) and designed with a hier- archical, coarse-to-fine restoration strategy.

## 20. Enhancing Image Quality Assessment Ability of LMMs via Retrieval-Augmented Generation

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; KADID-10k; CLIVE; LIVE
- arXiv: https://arxiv.org/abs/2601.08311
- PDF: https://arxiv.org/pdf/2601.08311.pdf
- Summary: —Large Multimodal Models (LMMs) have recently shown remarkable promise in low-level visual perception tasks, particularly in Image Quality Assessment (IQA), demonstrat- ing strong zero-shot capability. However, achieving state-of-the- art performance often requires computationally expensive fine- tuning methods, which aim to align the distribution of quality- related token in output with image quality levels. Inspired by recent training-free works for LMM, we introduce IQARAG, a novel, training-free framework that enhances LMMs’ IQA ability.

## 21. Q-Hawkeye: Reliable Visual Policy Optimization for Image Quality Assessment

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; AVA
- arXiv: https://arxiv.org/abs/2601.22920
- PDF: https://arxiv.org/pdf/2601.22920.pdf
- Summary: Image Quality Assessment (IQA) predicts percep- tual quality scores consistent with human judg- ments. Recent RL-based IQA methods built on MLLMs focus on generating visual quality de- scriptions and scores, ignoring two key reliability limitations: (i) although the model’s prediction stability varies significantly across training sam- ples, existing GRPO-based methods apply uni- form advantage weighting, thereby amplifying noisy signals from unstable samples in gradient updates; (ii) most works emphasize text-grounded reasoning over images while overlooking the model’s visual perception ability of image content. In this paper, we propose Q-Hawkeye, an RL- based reliable visual policy optimization frame- work that redesigns the learning signal through unified Uncertainty-Aware Dynamic Opti

## 22. VTONQA: A Multi-Dimensional Quality Assessment Dataset for Virtual Try-on

- Venue/Year: arXiv 2026
- Datasets: LIVE
- arXiv: https://arxiv.org/abs/2601.02945
- PDF: https://arxiv.org/pdf/2601.02945.pdf
- Summary: —With the rapid development of e-commerce and digital fashion, image-based virtual try-on (VTON) has attracted increasing attention. However, existing VTON models often suffer from artifacts such as garment distortion and body inconsistency, highlighting the need for reliable quality evaluation of VTON- generated images. To this end, we construct VTONQA, the first multi-dimensional quality assessment dataset specifically designed for VTON, which contains 8,132 images generated by 11 representative VTON models, along with 24,396 mean opinion scores (MOSs) across three evaluation dimensions (i.e., clothing fit, body compatibility, and overall quality).

## 23. Zoom-IQA: Image Quality Assessment with Reliable Region-Aware Reasoning

- Venue/Year: arXiv 2026
- Datasets: KonIQ-10k; SPAQ; KADID-10k; LIVE; CSIQ; AVA
- arXiv: https://arxiv.org/abs/2601.02918
- PDF: https://arxiv.org/pdf/2601.02918.pdf
- Summary: Image Quality Assessment (IQA) is a long-standing prob- lem in computer vision. Previous methods typically focus on predicting numerical scores without explanation or pro- viding low-level descriptions lacking precise scores. Re- cent reasoning-based vision language models (VLMs) have shown strong potential for IQA by jointly generating qual- *Corresponding authority descriptions and scores.

## 24. IQA-Adapter: Exploring Knowledge Transfer from Image Quality Assessment to Diffusion-based Generative Models

- Venue/Year: ICCV 2025
- Datasets: KonIQ-10k; KADID-10k; LIVE; AVA
- CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Abud_IQA-Adapter_Exploring_Knowledge_Transfer_from_Image_Quality_Assessment_to_Diffusion-based_ICCV_2025_paper.html
- PDF: https://openaccess.thecvf.com/content/ICCV2025/html/Abud_IQA-Adapter_Exploring_Knowledge_Transfer_from_Image_Quality_Assessment_to_Diffusion-based_ICCV_2025_paper.pdf
- Summary: Diffusion-based models have recently revolutionized im- age generation, achieving unprecedented levels of fidelity. However, consistent generation of high-quality images re- mains challenging partly due to the lack of conditioning mechanisms for perceptual quality. In this work, we pro- pose methods to integrate image quality assessment (IQA) models into diffusion-based generators, enabling quality- aware image generation.

## 25. 4KAgent: Agentic Any Image to 4K Super-Resolution

- Venue/Year: arXiv 2025
- Datasets: AVA
- arXiv: https://arxiv.org/abs/2507.07105
- PDF: https://arxiv.org/pdf/2507.07105.pdf
- Summary: We present 4KAgent , a unified agentic super-resolution generalist system designed to universally upscale any image to 4K resolution (and even higher, if applied iteratively). Our system can transform images from extremely low resolutions with severe degradations, for example, highly distorted inputs at 256×256, into crystal-clear, photorealistic 4K outputs. 4KAgent comprises three core components: (1)Profiling , a module that customizes the 4KAgent pipeline based on bespoke use cases; (2)APerception Agent , which leverages vision-language models alongside image quality assessment experts to analyze the input image and make a tailored restoration plan; and (3)ARestoration Agent , which executes the plan, following a recursive execution-reflection paradigm, guided by a quality-driven mixtur

## 26. A Causal Framework for Aligning Image Quality Metrics and Deep Neural Network Robustness

- Venue/Year: arXiv 2025
- Datasets: AVA
- arXiv: https://arxiv.org/abs/2503.02797
- PDF: https://arxiv.org/pdf/2503.02797.pdf
- Summary: Image quality plays an important role in the performance of deep neural networks (DNNs) that have been widely shown to exhibit sensitivity to changes in imaging conditions. Conventional image quality assessment (IQA) seeks to measure and align quality relative to human perceptual judgments, but we often need a metric that is not only sensitive to imaging conditions but also well-aligned with DNN sensitivities. We first ask whether conventional IQA metrics are also informative of DNN performance.

## 27. A Hybrid Approach for Unified Image Quality Assessment: Permutation Entropy-Based Features Fused with Random Forest for Natural-Scene and Screen-Content Images for Cross-Content Applications

- Venue/Year: arXiv 2025
- Datasets: LIVE; AVA
- arXiv: https://arxiv.org/abs/2508.17351
- PDF: https://arxiv.org/pdf/2508.17351.pdf
- Summary: Image Quality Assessment (IQA) plays a vital role in applications such as image compression, restoration, and multime- dia streaming. However, existing metrics often struggle to generalize across diverse image types—particularly between natural-scene images (NSIs) and screen-content images (SCIs)—due to their differing structural and perceptual char- acteristics. To address this limitation, we propose a novel full-reference IQA framework: Permutation Entropy-based Features Fused with Random Forest (PEFRF).

## 28. A Survey on Image Quality Assessment: Insights, Analysis, and Future Outlook

- Venue/Year: arXiv 2025
- Datasets: LIVE; AVA
- arXiv: https://arxiv.org/abs/2502.08540
- PDF: https://arxiv.org/pdf/2502.08540.pdf
- Summary: Image quality assessment (IQA) represents a piv- otal challenge in image-focused technologies, sig- nificantly influencing the advancement trajectory of image processing and computer vision. Recently, IQA has witnessed a notable surge in innovative research efforts, driven by the emergence of novel architectural paradigms and sophisticated compu- tational techniques. This survey delivers an exten- sive analysis of contemporary IQA methodologies, organized according to their application scenarios, serving as a beneficial reference for both begin- ners and experienced researchers.

## 29. AU-IQA: A Benchmark Dataset for Perceptual Quality Assessment of AI-Enhanced User-Generated Content

- Venue/Year: arXiv 2025
- Datasets: KonIQ-10k; KADID-10k; TID2013; AVA
- arXiv: https://arxiv.org/abs/2508.05016
- PDF: https://arxiv.org/pdf/2508.05016.pdf
- Summary: AI-based image enhancement techniques have been widely adopted in various visual applications, significantly improving the percep- tual quality of user-generated content (UGC). However, the lack of ∗Corresponding author. Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page.

## 30. Accelerate High-Quality Diffusion Models with Inner Loop Feedback

- Venue/Year: arXiv 2025
- Datasets: LIVE; AVA
- arXiv: https://arxiv.org/abs/2501.13107
- PDF: https://arxiv.org/pdf/2501.13107.pdf
- Summary: We propose InnerLoopFeedback ( ILF), a novel approach to accelerate diffusion models’ inference. ILF trains a lightweight module to predict future features in the denois- ing process by leveraging the outputs from a chosen diffu- sion backbone block at a given time step. This approach exploits two key intuitions; (1) the outputs of a given block at adjacent time steps are similar, and (2) performing par- tial computations for a step imposes a lower burden on the model than skipping the step entirely.
