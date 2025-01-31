# Archival_AI_DePOSIT

Archival AI Description Pipeline for Open-Source Image Tagging

Developed by Giulia Taurino, Ph.D. and Shoumik Majumdar

Visual Language Models (VLMs) are multimodal AI models that can be used for tasks like image tagging and captioning, visual question answering, and text transcription. This repository contains a pipeline that leverage open-source VLMs to automatically generate detailed descriptions of archival materials, such as historical photographs from journalism collections. Through “image re-prompting” (Kaduri et al. 2024), this pipeline allows to extract key information like authors, dates, locations, people portrayed, and events directly from the record itself, significantly improving the discoverability and accessibility of archival collections via automated metadata extraction.

Key components of AI DEPOSIT:

1. Preprocessing: cleaning and structuring the data by removing noise, file format conversion
2. Data ingestion: the initial step where digital archival materials are ingested into the VLMs system;
3. Textual feature extraction: utilizing VLMs to identify key elements within the archival materials, such as photographer's name, date, subject, location and other contextual information via a text transcription prompt for optical character recognition (OCR) for text within images;
4. Visual feature extraction and metadata generation: applying the trained model to new archival materials to automatically generate descriptive metadata fields like title and abstract via an image description prompt that includes the outputs from the text prompt;
5. Qualitative evaluation: a human archivist reviews the generated metadata to ensure accuracy, address potential errors, and suggest necessary adjustments. This feedback is used for A/B testing and paired with other evaluation metrics to assess model's accuracy;
6. Model training: training VLMs on large datasets of annotated archival materials to learn patterns and associations between visual or textual features and corresponding metadata. [extra step] 

Potential benefits of AI DEPOSIT:

1. Increased efficiency: automating metadata creation significantly reduces the time and labor required to catalog large collections;
2. Improved discoverability: Richer and more accurate metadata improves the ability of researchers to find relevant archival materials through search functions;
3. Enhanced access: Faster cataloging enables quicker public access to archival collections.  

Important considerations when implementing AI DEPOSIT:

1. Data quality: The quality of the image data heavily impacts the accuracy of the generated metadata;
2. Privacy concerns: care must be taken when dealing with sensitive personal information within archival records;
3. Human oversight: AI should be seen as a tool to assist archivists, not replace their expertise. 
