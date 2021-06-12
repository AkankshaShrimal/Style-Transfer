# Style-Transfer

## Project Overview

Implemented [Image Style Transfer Using Convolutional Neural Networks by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). 

Rendering the semantic content of an image in different
styles is a difficult image processing task widely known as **Image Style Transfer**. In this project, I implemented Image Style Transfer using features extracted from pre-trained **VGG-19**. 
Image Style Transfer requires two images **Content Image** from which objects and arrangements are extracted and **Style Image** from which style, colors and textures are etracted. Using objects from Content Image and colors and textures from Style Image a new **Target Image** is generated.   

<div align="center"><img src="Images/output_images/plots1.jpg" height='200px'/></div>

## Detailed Steps and Explanation

<div align="center"><img src="plots/classes_imgs.jpg" height='200px'/></div>

Style transfer relies on separating the content and style of an image. Given one content image and one style image, we aim to create a new, target image which should contain our desired content and style components:
- objects and their arrangement are similar to that of the content image
- style, colors, and textures are similar to that of the style image
- Use pre-trained VGG19 Net to extract content or style features from a passed in image 


### Steps 
- Content and Style features are extracted using **Vgg-19** convolutional and pooling layers so vgg19.classifier layers are discarded.  
- According to [Gayts](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) the deeper layers of Convolutional Neural Networ capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction (per pixel colors, textures). In contrast, reconstructions from
the lower layers simply reproduce the exact pixel values of
the original image. 
Following above, **content features are extracted using conv4_2** for Vgg-19.  
- The style representation are computed using correlations between the different features in different layers of the CNN. Gram Matrix is used to capture these style representations for different CNN layers (detects how strongly features in one feature map relate to other feature map from same CNN layer eg. common color). 
Following above, **Style features are extracted using different Gram Matrix corresponding to each convolutional layer** for Vgg-19. 
With co-relations from multiple CNN layers a multi-scale style representation of input image is obtained (captures large and small style features).  
- The style representations are calculated as style image passes through the network at first convolutional layer in all 5 stacks i.e convx_1, where corelations at each layer are given using a gram matrix. 
- Steps to generate gram matrix at each convolutional layer stack. 
    - x,y  dimensions of feature maps are flattened thus (d,x,y) becomes (d,x*y)
    - The above matrix is multiplied by its transpose to obtain gram matrix of dimension (d,d) for current convolutional layer stack. (this step keeps non localized information) 

- For generating the target image using content and textual features, we change the target image until its content matches to content image and style matches to style image. Original target image is taken as the content image clone and trained until the loss is minimised. 
- Two different losses used : 
  - Content Loss :- Compares the content representation of content and target image using mean squared error. **Content representation is taken from conv4_2 layer of Vgg-19 for both images**
  Target  
  - Style Loss :- Computed using mean squared distance between gram matrices of style and target image at each layer (conv1_1,conv2_1,conv3_1,conv4_1,conv5_1) 
  Separate weights for each layers used, a is a constant for number of values in each layer 
  - Total Loss : Content Loss + Style Loss , It is used along with back propagation to iteratively change the target image to miniise the loss. 

- Constant weights alpha and beta are used to balance out total loss over both losses. **Often the style weights are kept much larger** Weights are expressed as a ration alpha/beta which implies the smaller the ratio the more stylistic effect visible.   


<div align="center"><img src="plots/classes_imgs.jpg" height='200px'/></div>


## Hyper Parameters 

- Style weights corresponding to each layer  
- Content weight (alpha) 1 
- Style weight (beta) 1e4
- Epochs 5000 


## Results

Follwing are the results of the project:

                                        Fig 1. Feature Visualization
   <div align="center"><img src="plots/Final_Feature_extraction.png" height='450px'/></div>                                         

                                Fig 2. Variance of PCA projected over min-max normalized data
<div align="center"><img src="plots/PCA_variance_graphs.png" height='500px'/></div>

                                Fig 3 Receiver Operating Characteristic (ROC) Curves
                                    a. ROC of PCA reduced data
                                    b. ROC of LDA reduced data
                                    c. ROC of LDA on PCA reduced data
<div align="center">
    <img src="plots/roc_over_pca.png" height='225px'/>
    <img src="plots/roc_over_lda.png" height='225px'/>
    <img src="plots/roc_over_lda_over_pca.png" height='225px'/>
</div>

                Fig 4. Optimal Parameters of classifiers after grid search
<div align="center"><img src="plots/optimal_param.png" height='180px'/></div>

     Fig 5. Comparing various classifiers with different feature sets over Accuracy/Recall/Precision/F1 score
                                    a. Results of PCA reduced data
                                    b. Results of LDA reduced data
                                    c. Results of LDA on PCA reduced data
         
<div align="center"><img src="plots/results_pca.png" height='200px'/></div>
<div align="center"><img src="plots/results_lda.png" height='200px'/></div>
<div align="center"><img src="plots/results_lda_over_pca.png" height='200px'/></div>


                                Fig 5. CNN architecture 
<div align="center"><img src="plots/CNN_model.jpeg"/></div>


        Fig 6. Comparing between deep learning classifiers with Accuracy/Recall/Precision/F1 score
                                    a. CNN
                                    b. ResNet-101 [Stratery-1 : Retrain only last layer]
                                    c. ResNet-101 [Stratery-2 : Retrain last few layers]
<div align="center"><img src="plots/results_deep_learning_classifiers.png" height='150px'/></div>


                    Fig 6. ResNet-101 [Stratergy-2] performance measured using Class Activation Maps
<div align="center"><img src="plots/class_activation_map.png"/></div>


## References

1. [Image Style Transfer Using Convolutional Neural Networks by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). 
2. [Udacity - Pytorch Nanodegree](https://www.udacity.com/course/deep-learning-pytorch--ud188)


## Project Team Members

1. Akanksha Shrimal
