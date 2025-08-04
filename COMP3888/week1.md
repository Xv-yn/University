# General Knowledge

## Project Client

- They will set the **requirements and scope**
- They will provide any **domain knowledge** needed
- They will provide any **technical resources** needed

## Liked Projects

1. AI Boardgame

```
Project ID: P2

Company/Department name: USyd School of Computer Scienc

Project Title: Creating a Board Game AI Advisor

    Project description and scope: Imagine you are playing a board game. What is the best move to make? Let's find out with AI! In this project, the team will build an AI advisor for famous board games.
    Expected outcomes/deliverables: Open source (1) Board game simulator, (2) multiple AI models for the game, (3) UI for using the AI in a real life game.

    Specific required knowledge, skills, and/or technology: Assumed/required knowledge and skills: COMP3308 / COMP3608, strong programming skills in Python, experience with board games!
    Related general fields/disciplines (if applicable): Artificial Intelligence;
    Can you provide dataset for this project?: Not applicable
    How many groups would you like to work with on this project?: 2 groups
```

2. Website/Search Engine Tailored for a Specific Dataset

```
 Project ID: P3

Company/Department name: Institute for Musculoskeletal Health

Project Title: Designing a searchable physical activity directory for people with disability in Australia

    Project description and scope: This project is part of two broader government-funded research initiative aimed at supporting people with moderate-to-severe traumatic brain injury (TBI) and people with intellectual disability (ID) to increase their physical activity levels. We are developing two online resource hubs tailored to individuals with TBI and ID and their support networks (e.g., health professionals, family members, support workers). A key feature of both hubs will be a searchable activity directory that enables users across Australia to find relevant physical activity opportunities near them. These may include exercise classes, adaptive sports, recreational groups…etc.  Computer Science Capstone students will contribute by designing and developing this user-friendly searchable activity directory, the details of which will be co-designed with students. For example, this may include: a chatbot or AI-powered search that uses non-identifiable user input to deliver tailored activity suggestions; and a dynamic backend system that allows easy updates to the activity directory by administrators. The directory should include: activity name, location(s), brief description, target audience (e.g., disability-specific or general), cost, link to activity/organisation, contact information.
    Expected outcomes/deliverables: (1) A functioning, searchable activity directory with the ability to be integrated into the online resource hubs. (2) Documentation and plain language user guide for future maintenance and updates. (3) Optional: AI-enhanced or chatbot functionality.
    Specific required knowledge, skills, and/or technology: Web development (frontend and backend), software development fundamentals, basic data and database design, structuring and management, artificial intelligence/machine learning, cybersecurity awareness, information systems and user experience design (good to have but not essential)
    Related general fields/disciplines (if applicable): Web Development;Software Development;Data Science/Analytics;Artificial Intelligence;Information Systems;
    Can you provide dataset for this project?: Yes, the project client will provide dataset.
    How many groups would you like to work with on this project?: 1 group
```

3. Use AI to detect "segmented nuclei or cells"

```
Project ID: P31

Company/Department name: School of Computer Science

Project Title: Cross-Stain Cell Segmentation in Histopathology Using Deep Learning

    Project Description and Scope:

    Cell segmentation is a critical step in computational pathology for analyzing tissue morphology, immune cell infiltration, and tumor microenvironments. However, histology slides come in different stains, primarily H&E (hematoxylin and eosin) and IHC (immunohistochemistry). Models trained on one stain type often fail on others due to color, texture, and intensity differences.

    This project challenges students to build a generalizable segmentation model that can detect and segment nuclei or cells in both IHC and H&E-stained images using publicly available datasets.

    The pipeline should include:

    Training on one stain type (e.g., IHC or H&E)

    Applying to the other without annotations (zero-shot testing)

    Optional domain adaptation or stain normalization
    Evaluation of segmentation robustness across stain domains
    Expected Outcomes / Deliverables:

        A segmentation model (U-Net, Cellpose, etc.) trained on IHC or H&E images

        Performance evaluation on test stain and generalization to the unseen stain

        Visual outputs: overlayed masks, density maps, before/after stain normalization
        A documented codebase and final report
    Suggested Public Datasets

        IHC-Stained Datasets

            DeepLIIF

            Links to an external site. – IHC images with cell/nuclei segmentation masks

            NuClick

            Links to an external site. – Lymphocyte segmentation in IHC-stained histology images

        H&E-Stained Datasets

            MoNuSeg

Links to an external site. – Nuclei segmentation dataset from H&E-stained WSIs (multiple organs)
PanNuke

            Links to an external site. – Large-scale multi-organ dataset with pixel-level nuclei labels and masks
    Specific Required Knowledge, Skills, and/or Technology:
    Essential: Python programming (NumPy, pandas, matplotlib), Basic understanding of convolutional neural networks (CNNs), Experience with PyTorch or TensorFlow, Image processing with OpenCV or scikit-image.
    Helpful (but can be learned during project): U-Net architecture or Cellpose tool, Evaluation metrics for segmentation (Dice, IoU), Data augmentation and patch extraction, Use of Jupyter notebooks for experimentation and reporting.
    Related General Fields / Disciplines: Medical Image Analysis, Artificial Intelligence in Healthcare, Biomedical Engineering, Computer Vision, Digital Pathology, Bioinformatics
    Suggested Student Workflow & Techniques

        Train a segmentation model on IHC images: use a well-annotated IHC dataset such as DeepLIIF, Model: U-Net, Cellpose, or similar

        Evaluate generalization: by testing the trained model on H&E-stained images (e.g., MoNuSeg), observe potential performance drop due to stain differences

        Apply stain normalization techniques to reduce domain shift, techniques: Macenko, Reinhard, or Vahadane, apply during preprocessing or as part of the pipeline

        Retrain or fine-tune the model using: Mixed-stain datasets (combine IHC and H&E samples), Color augmentation (brightness, hue, stain jittering)

        Evaluate the improved model for cross-stain generalization, use metrics like Dice coefficient, IoU, and visual inspection, compare performance before and after stain adaptation

        (Optional Goal):

        Use style transfer or domain adaptation (e.g., CycleGAN) to align IHC and H&E stain domains

        Explore self-supervised or pseudo-labeling techniques on unlabeled H&E/IHC patches

```

4. Use AI to format data ina way a segemntation model can analyze it

```
Project ID: P32

Company/Department name: School of Computer Science

Project Title: Improving Generalization of Cell Segmentation Across Histology Stains

Project Description and Scope Segmentation models trained on histopathology images stained with one technique, such as Hematoxylin & Eosin (H&E) or Immunohistochemistry (IHC), often perform poorly on images stained with the other due to significant domain shifts in color, texture, and appearance. This project focuses on developing and evaluating domain adaptation methods to enable segmentation models to generalize across stain types without requiring extensive labeled data for each domain.

Students will explore:

    Style transfer techniques (e.g., CycleGAN) to transform images from one stain domain to another, reducing domain discrepancy.

    Self-supervised learning and pseudo-labeling methods to leverage unlabeled or weakly labeled data for improving cross-domain segmentation.

    Training segmentation models on one stain type and testing on the other, with and without domain adaptation techniques.

    Comprehensive evaluation of segmentation performance and model robustness across staining domains.

This project provides practical experience with advanced deep learning methods in computational pathology and addresses a key challenge in clinical AI deployment.

Expected Outcomes / Deliverables

    Implementation of a style transfer model (CycleGAN or equivalent) to translate between IHC and H&E image domains.

    Segmentation model trained on one stain domain (IHC or H&E).

    Application of self-supervised learning or pseudo-labeling to improve segmentation on the unlabeled domain.

    Quantitative evaluation comparing baseline segmentation performance versus domain-adapted models.

    Visualizations including style-transferred images, segmentation masks, and performance metrics.

    Well-documented codebase and a final written report

Suggested Public Datasets

IHC-Stained Datasets

    DeepLIIF

    Links to an external site. – IHC images with cell/nuclei segmentation masks

    NuClick

    Links to an external site. – Lymphocyte segmentation in IHC-stained histology images

H&E-Stained Datasets

    MoNuSeg

    Links to an external site. – Nuclei segmentation dataset from H&E-stained WSIs (multiple organs)

    PanNuke

    Links to an external site. – Large-scale multi-organ dataset with pixel-level nuclei labels and masks

  Required Skills

    Essential: Python, PyTorch or TensorFlow, basic CNNs
    Helpful: GANs or CycleGAN, segmentation models, Self-supervised learning and pseudo-labeling techniques

Related General Fields / Disciplines: Medical Image Analysis, Artificial Intelligence in Healthcare, Biomedical Engineering, Computer Vision, Digital Pathology, Bioinformatics, Self-Supervised Learning

Suggested Student Workflow

    Familiarize with datasets and problem domain: download and explore IHC (e.g., DeepLIIF) and H&E (e.g., MoNuSeg) datasets, understand segmentation masks and evaluation metrics (Dice, IoU)

    Baseline segmentation model: train a segmentation model (e.g., U-Net) on the IHC dataset, evaluate baseline performance on both IHC test set and H&E test set to observe domain shift effects

    Implement stain style transfer: train or use a pretrained CycleGAN model to convert IHC images to H&E style or vice versa, visually inspect style-transferred images for quality

    Domain adaptation via style transfer: use style-transferred images to train or fine-tune the segmentation model, evaluate whether domain adaptation improves segmentation on the target stain dataset

    (Optional Goal) Explore self-supervised or pseudo-labeling techniques: apply self-supervised learning methods on unlabeled data to learn robust features, generate pseudo-labels on unlabeled H&E or IHC patches and use them to further train the model

    Evaluation and analysis: quantitatively compare segmentation performance before and after domain adaptation, visualize segmentation masks, failure cases, and domain adaptation effects

    Documentation and reporting: document code, experiments, and results, prepare a final report and presentation summarizing findings and future directions
```

5. Use AI to Summarize what the EULA and etc to determine waht data is taken from the user

```
Project ID: P38

Company/Department name: School of Computer Science

Project Title: PrivBot - Interactive Mobile App Privacy Compliance Checks

Project description and scope: Mobile applications (apps) hosted on platforms such as the Google Play Store and Apple App Store are required to include data privacy labels. These labels provide a concise summary of what data is collected from end-users, what is shared with third parties, and for what purposes. While these labels are designed to aid end-user comprehension and improve transparency, developers are also obligated to publish comprehensive privacy policies that explain their data practices in detail. However, it is widely acknowledged that most end-users do not engage with these lengthy privacy policies. Moreover, privacy policies tend to be written to meet regulatory compliance obligations—such as the General Data Protection Regulation (GDPR) in the EU or the Australian Privacy Principles (APP)—and often focus on broader legal requirements including user rights, cookie usage, and data retention policies, rather than offering clarity on mobile app-specific data behaviours.

Project Aim: This project proposes the development of Privacy-Bot, a system that leverages state-of-the-art Natural Language Processing (NLP) techniques to automatically extract and summarise data handling practices specifically related to mobile app behaviour from privacy policies. The extracted information will then be visualised through user-friendly dashboards. End-users benefit from a clear overview of data practices, enabling them to easily navigate to relevant sections of the privacy policy, while app developers gain insights to help verify and ensure compliance of their mobile app services with privacy regulations

Project Scope:

1.  Development of an Interactive User Interface: The Privacy-Bot will feature an online, interactive dashboard accessible to users. Given a pair of URLs—one pointing to a mobile application (e.g., from the App Store or Google Play) and the other to its corresponding privacy policy—the system will generate an overview of the app’s data collection, sharing, and compliance practices. Users can interact with this overview to navigate directly to relevant sections of the privacy policy, where key data items and their associated purposes are automatically highlighted by the bot.

2.  Back-End Architecture Using Language Models: The system’s back end will utilise encoder and / or decoder-based language models selected from established research. These models will operate in a plug-and-play fashion, supporting modular integration. This modularity ensures that individual components, such as NLP models, can be updated or replaced independently without requiring major changes to the overall system. This project does not involve training or designing language models; instead, it focuses on performing inference using pre-trained or fine-tuned models within Python environments, leveraging PyTorch and CUDA for efficient computation.

Local Hosting on University Resources: All NLP models employed in the system will be based on open-source implementations and will be hosted locally using the University of Sydney’s computing infrastructure.

Expected outcomes/deliverables: Interactive dashboard integrated to our exisitng backend models

Specific required knowledge, skills, and/or technology -Software/Web Development, LLMs and Machine Learning

Please specify related general fields/disciplines (if applicable): Web Development; Software Development; Artificial Intelligence; Security/Networks;

Can you provide dataset for this project?: Not applicable

How many postgraduate groups would you like to work with on this project? (each group contains 5-7 students): 1 group
```

6. Use AI to recognize images and identify "cell tracking"

```
Project ID: P40

Client: School of Computer Science

Project Title: AI Framework for Cell Tracking

Project Description and Scope:

Cell tracking is a well-established task in computer vision and biological imaging that involves tracking the migration and proliferation of cell(s) across a sequence of time frames. This task can typically be achieved through two independent processes: cell segmentation—detecting cell instances at each individual time frame—and cell linking—associating instances of the same cell across different time frames to construct cell trajectories. Despite efforts spanning over a decade, cell tracking remains an active area of research. The inherent challenge of this task lies in the facts that cell instances can be visually indistinguishable, cell movements are stochastic, and cells can undergo mitosis at any time. Like other fields in computer vision, deep learning has recently emerged as the state-of-the-art approach for cell tracking. However, its application mostly pertains to cell segmentation (using CNNs); many methods still rely on classical methods (such as discrete optimisation) for cell linking.

Objective. This project aims to design and implement an end-to-end cell tracking framework that utilises deep learning for both cell segmentation and cell linking. Such a framework can be inspired by or extend from the pioneering works in [1], [2], [3].

Expected experience. Students will be given the autonomy to design and implement deep learning algorithms, whether it be from scratch or existing code. The client will provide students with a starting point and sufficient guidance throughout the project. The outcome of this project is expected to be a meaningful contribution to the research community.

Timeline and Milestones:

    Mid-term milestone: complete design of cell tracking framework.
    Final milestone: code implementation of cell tracking framework (note that the initial design can be modified as appropriated during implementation).

Expected outcomes/deliverables:

    A functional code implementation of the cell tracking framework (if possible, made available on GitHub);
    A submission to the Cell Tracking Challenge (https://celltrackingchallenge.net

    Links to an external site.)for comprehensive evaluation;
    Demo on the AI framework for validation.

Specific required knowledge, skills, and/or technology:

    Interest and foundational knowledge in deep learning (specifically, computer vision);
    Proficiency in Python and deep learning libraries such as PyTorch, NumPy, etc.; and
    Ability to conduct independent research (with initial guidance).

Related general fields/disciplines:

Deep learning; Computer vision; Biological imaging.

Resources provided by the client:N/A

Resources need to be prepared by the group: N/A

References

[1]        T. He, H. Mao, J. Guo, and Z. Yi, “Cell tracking using deep neural networks with multi-task learning”.

[2]        T. Ben-Haim and T. R. Raviv, “Graph Neural Network for Cell Tracking in Microscopy Videos,” July 17, 2022, arXiv: arXiv:2202.04731. doi: 10.48550/arXiv.2202.04731.

[3]        B. Gallusser and M. Weigert, “Trackastra: Transformer-based cell tracking for live-cell microscopy,” July 24, 2024, arXiv: arXiv:2405.15700. doi: 10.48550/arXiv.2405.15700.
```

7. Use AI to look at pictures of pancrease and determine "pancreas segmentation"

```
Project ID: P41

Client: School of Computer Science

Project Title: PanSegAI-Deep Learning Model for Pancreas Segmentation from Medical Images

Project Description and Scope: Accurate pancreas segmentation is critical for diagnosing and monitoring pancreatic diseases, including cancer and diabetes. Manual segmentation is time-consuming and operator dependent. Rapid advances in deep learning offer promising solutions for automated, accurate and reproducible segmentation across different imaging modalities.

This project aims to (1) develop a robust deep learning model for pancreas segmentation from CT and MRI scans; (2) achieve benchmark/comparable performance against existing SOTA models; (3) evaluate generalizability across multi-center datasets; (4) Open-source tools contributing to the research communities

    Expected Outcomes/deliverables:
        A validated pancreas segmentation model with high accuracy across CT and MRI.
        Open-source code and trained models for reproducibility.
        Comparison of existing methods and analysis.
        Potential integration into radiology workflows for clinical use.

    Resources -
        nnU-Net: A self-configuring deep learning framework for biomedical image segmentation. It automatically adapts its architecture and training pipeline to any new dataset. (https://github.com/MIC-DKFZ/nnUNet)
        MONAI: Medical Open Network for AI, a PyTorch-based framework for deep learning in healthcare imaging, developed by NVIDIA and the community. (https://monai.io/)
        TotalSegmentator: An open-source tool for multi-organ segmentation in CT images, built on top of nnU-Net and trained on over 1,000 scans. (https://github.com/wasserth/TotalSegmentator)
    Specific required knowledge, skills, and/or technology:  (1) Python programming; (2) deep learning, computer vision, and image processing.
    Related general fields / disciplines: Computer Science
    Can you provide dataset for this project?: N/A

```

8. Facial Recognition

```
Project ID: P43

Company/Department name: School of Computer Science

Project Title: Synthetic Face Dataset Generation and Benchmarking for Face Recognition Models

Project Description and Scope:
This project will programmatically use Gemini and Midjourney to generate a large synthetic face dataset containing thousands of unique identities, each with multiple images varying in angles, lighting, resolution, and occlusions. We will evaluate the dataset using state-of-the-art face recognition models (e.g., ArcFace, FaceNet) and benchmark its quality using metrics such as Fréchet Inception Distance (FID), Inception Score (IS), and identity consistency measures. The outcome will be a high-quality synthetic dataset and performance benchmarks that highlight its potential for training and evaluating face recognition systems.

Expected Outcomes/Deliverables:
a) Synthetic faces dataset and all the prompts used to generate each image b) Performance results using state-of-the-art models

Required Knowledge, Skills, and/or Technology:
Computer Vision, Machine Learning

Related General Fields/Disciplines: Artificial Intelligence; Security/Networks

Dataset Provided: Not applicable

Number of Postgraduate Groups: 1 group
```
