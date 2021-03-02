**Goal**  
Predicting humidity using given weatherdata

**Approach**  
Predict humidity by minimizing dimensionality with PCA and using LinearRegression from sklearn library

**Results**  
The approach showed that using just half of the maximum number of principal components available gives decent results

![image](https://user-images.githubusercontent.com/67264647/109617093-a3217600-7b3e-11eb-943f-1466cb53c6bb.png)


However increasing the maximum number of principal components to 12 which still is a 25% reduction, proves to be a much 
better solution.

![image](https://user-images.githubusercontent.com/67264647/109616956-7b321280-7b3e-11eb-8908-b39894618b86.png)
