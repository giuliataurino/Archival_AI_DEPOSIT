How to ssh into the Discovery Cluster
ssh s.majumdar@login.discovery.neu.edu 

Ideally, we would run the Qwen model on the same set of images (we can use the Sarah's dataset of 100/200 img or the OCR dataset I had shared with you, which is preferable for evaluating text transcription, but maybe less relevant for this paper?) with 3 different prompting approaches:
1. single "describe the image and provide a title" prompt on the front (without additional textual inputs from the back);
2. double prompt: text transcription + image description prompts in two separate tasks (run twice loading two separate images: 1.  on the back and 2. on the front;
3. image re-prompting on the front:  text transcription prompt on the front of the image > without reloading the image, compress information from output > re-prompt to describe the image (cf. “image re-prompting” approach by Kaduri et al. 2024)   

We plan on using the evaluation metrics to compare these outputs. This would allow us to bypass the A/B testing with human archivists in case they don't have time to write manual descriptions. We can still proceed with further evaluation after the paper submission.