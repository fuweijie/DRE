##### Differentiated Explanation

###### This is the pytorch implemention for paper “Differentiated Explanation of Deep Neural Networks with Skewed Distributions”. 

We propose a simple but efficent approch for the differentiated explanations of black-box classifiers. To do this, we introduce a trainable relevance estimator that produces relevance scores in a skewed distribution. Specifically, we present the concept of distribution controllers and integrate it with a neural network to directly guide the distribution of relevance scores. By analyzing the effect of the skewness of distributions, we develop the controllers with right-skewed distributions for differentiated saliency maps. Then we introduce the classification loss to optimize the estimator. The benefit of this strategy is to better mimic the behavior of deep neural networks without non-trivial hyperparameter tuning, leading to higher faithfulness of explanation.

##### Running the Demo
###### Our code is implemented with:
      matplotlib==3.1.3 
      more-itertools==8.3.0 
      numpy==1.18.1 
      pillow==7.1.2 
      pytorch==1.1.0 
      scikit-image==0.16.2 
      scikit-learn==0.22.1 
      scipy==1.4.1 
      torchvision==0.3.0 
###### Trained models are available at https://drive.google.com/drive/folders/1t8sSK6elwalIyNfuQrw3wnds1rh3GWiF?usp=sharing
###### python saliency_demo.py
