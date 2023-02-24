# U-Net_vs_AttentionU-Net_vs_TransformerU-Net
# 1. Abstracts

The U-Net is a model designed to include the positional features of pixels in an image. It is widely used in biomedical image segmentation. The name "U-Net" is given because the network is shaped like a "U" and it consists of a contracting path and an expanding path.<br>
![image](https://user-images.githubusercontent.com/110075956/221114736-713f05f6-e2b3-44eb-a2be-acb7050c635a.png)

In contracting path, U-Net captures context information. And high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information.<br>



# 2. Results

I used used 'dice loss' as the loss function and 'IoU' as the accuracy.<br>
Training was conducted 30 epochs, and the result is as follows.<br><br>
**U-Net**
![image](https://user-images.githubusercontent.com/110075956/220638159-cfab4e2b-3cd0-40be-b90c-83cd2d857074.png)
**AttnU-Net**
![image](https://user-images.githubusercontent.com/110075956/220638528-70b725bf-4ea5-4df5-94ad-4629ec6abe90.png)
**TransU-Net**
![image](https://user-images.githubusercontent.com/110075956/220638548-f688c5c3-f652-465d-9cb8-052865ebfabc.png)
<br>
I checked the results by inserting a test sample into the trained model.<br><br>
**U-Net**
![image](https://user-images.githubusercontent.com/110075956/220639387-aee1881d-ef14-4d51-8560-d374f6660197.png)
**AttnU-Net**
![image](https://user-images.githubusercontent.com/110075956/220639447-040be4db-9ca0-4565-b96c-7eb8df51faf8.png)
**TransU-Net**
![image](https://user-images.githubusercontent.com/110075956/220639504-19ed736d-24d1-4177-ab4b-5a6671269671.png)

# 3. Conclusions
Although U-Net has better loss and accuracy than AttnU-Net and TransU-Net, AttnU-Net and TransU-Net showed more reasonable predictive results. U-Net looked as if it had not been trained at all. So I concluded I made a mistake in setting the loss function and accuracy. The loss function and accuracy metrics didn't feed into the performance of models well or I put the wrong values in the loss function and accuracy metrics.<br>


# 4. References
RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015. p. 234-241.<br>
OKTAY, Ozan, et al. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999, 2018.<br>
DOSOVITSKIY, Alexey, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.<br>
CHEN, Jieneng, et al. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.<br>
