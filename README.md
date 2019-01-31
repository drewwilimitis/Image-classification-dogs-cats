# Image-classification-dogs-cats
Classifying Images of Dogs and Cats from Kaggle

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
  
 This is my attempt at an older Kaggle competition that felt like the "Hello World" of image processing and computer vision.  
   
 To classify images as a dog or a cat, I used algorithms to augment the training data and detect outliers, extract some transformation invariant image features, and segment the images via adjacency regions and graph cuts. 
 
 I primarily used an image feature called Histogram of Oriented Gradients (HOG Features) to input to a standard linear SVM or random forest model. This method is outdated and outperformed by more recent deep learning methods, which made it even more fun to try and get creative with some of these ancient approaches
