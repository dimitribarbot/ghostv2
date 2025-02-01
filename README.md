# Ghost V2

The goal of this project is to try to implement [Ghost](https://github.com/ai-forever/ghost) but with other face detection and recognition models than InsightFace RetinaFace and ArcFace to allow a more permissive licence than the InsightFace ones. It includes a full rewrite of the original [Ghost](https://github.com/ai-forever/ghost) repository code, by integrating Pytorch Lightning to boost training and using other datasets than VGGFace2.

## Image Swap Results 
![](/examples/results/training/source_target_grid.png)

## Reminder of GHOST Ethics

Here you can find the ethic chart written by the original authors of Ghost which still holds today:

"Deepfake stands for a face swapping algorithm where the source and target can be an image or a video. Researchers have investigated sophisticated generative adversarial networks (GAN), autoencoders, and other approaches to establish precise and robust algorithms for face swapping. However, the achieved results are far from perfect in terms of human and visual evaluation. In this study, we propose a new one-shot pipeline for image-to-image and image-to-video face swap solutions - GHOST (Generative High-fidelity One Shot Transfer).

Deep fake synthesis methods have been improved a lot in quality in recent years. The research solutions were wrapped in easy-to-use API, software and different plugins for people with a little technical knowledge. As a result, almost anyone is able to make a deepfake image or video by just doing a short list of simple operations. At the same time, a lot of people with malicious intent are able to use this technology in order to produce harmful content. High distribution of such a content over the web leads to caution, disfavor and other negative feedback to deepfake synthesis or face swap research.

As a group of researchers, we are not trying to denigrate celebrities and statesmen or to demean anyone. We are computer vision researchers, we are engineers, we are activists, we are hobbyists, we are human beings. To this end, we feel that it's time to come out with a standard statement of what this technology is and isn't as far as us researchers are concerned.
* GHOST is not for creating inappropriate content.
* GHOST is not for changing faces without consent or with the intent of hiding its use.
* GHOST is not for any illicit, unethical, or questionable purposes.
* GHOST exists to experiment and discover AI techniques, for social or political commentary, for movies, and for any number of ethical and reasonable uses.

We are very troubled by the fact that GHOST can be used for unethical and disreputable things. However, we support the development of tools and techniques that can be used ethically as well as provide education and experience in AI for anyone who wants to learn it hands-on. Now and further, we take a **zero-tolerance approach** and **total disregard** to anyone using this software for any unethical purposes and will actively discourage any such uses."

## Disclaimer

We understand the unethical potential of GhostV2 and are committed to protecting against such behavior. The repository has been modified to prevent the processing of inappropriate content, including nudity, graphic content, and sensitive content. Collaboration with websites that promote the use of unauthorized software is strictly prohibited. Those who intend to engage in such activities will be subject to repercussions, such as being reported to authorities for violating the law.

## Installation
  
1. Clone this repository
```bash
git clone https://github.com/dimitribarbot/ghostv2.git
cd ghostv2
```

2. Install dependent packages
```bash
pip install -r requirements.txt
```

3. Download weights

To only download the needed models for inference run this script from the root folder of the repository:
```bash
sh download_inference_models.sh
```

To download all the needed models for inference, dataset preprocessing and training, run this script from the root folder of the repository:
```bash
sh download_all_models.sh
```