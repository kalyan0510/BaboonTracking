# BaboonTracking

Using YOLO v4 and deep sort to track baboons in the wild.  
- YOLO produces multi-object detections per each frame
- deepSORT stitches these detections across all frames to create trajectories

Given the number of human annotations is very less (~500), an augmentation module with crop-and-stitch augmentation is created.
### In the crop-and-stitch augmentation:
  an image patch of an object annotation is cropped and put back in a similarly looking neighbour hood patch. To search for such a patch, the border strips around the annotation is vectorized to a constant length and used to compare against all locations to find out best spot to stirch the patch over. 
  
 For each labeled object in the image, 5 best possible annotations (new locations and sizes to paste over) are calculated using non-maximal suppression on the distance map produced by comparing the vector with all possible locations in the image space. 
  
 The augmentation module resticts on certain augmentations with help of scoring functions. The score is gathered for each augmentation and only the top few scoring ones are allowed.
 - a score to favor augmenting bigger objects (as its less meaningful to shuffle small baboons in the given environment)
 - a score to penalize putting the baboons on the very top portions of image (in all the images, the ground horizon is in the middle and so its mostly sky in the upper half of image)
 - a method to filter out pasting over other objects

Once the new pasting locations are finalized, the image is pasted over them and boundaries are smoothened using alpha-blending. 

#### The augmentation result looks like this:
![ezgif-4-5164009ae9](https://user-images.githubusercontent.com/14043633/191661883-6f29cb44-15f2-44bb-87bc-658c6a8088eb.gif)
![ezgif-4-208788f916](https://user-images.githubusercontent.com/14043633/191661891-a336c4c7-ae43-49df-aff2-263e65b7f9ca.gif)

With multiple augmentations linked in chain: The 500 images served to create dataset with ~5k annoatated images. 

### Sample YOLO detection results (mAP@0.50 = 88.38 % over the test data):
![vlcsnap-2021-09-27-16h58m07s271_RGB_ShuffleObject8 png](https://user-images.githubusercontent.com/14043633/191662111-71df4fbf-6479-4c4c-b2c7-4caa000a5cae.png)
![vlcsnap-2021-09-27-16h58m38s197_RGB_ShuffleObject8 png](https://user-images.githubusercontent.com/14043633/191662135-392bb499-4b8e-4472-ab42-ed71852e50b5.png)
![vlcsnap-2021-10-04-13h39m21s459_RGB_ShuffleObject8 png](https://user-images.githubusercontent.com/14043633/191662144-0844e1e6-f7a0-4cbf-b060-8dbb19ce74bd.png)
