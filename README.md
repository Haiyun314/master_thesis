approximate the **sin** function with **neural network**

### Network Setting
x = [-10, 10], discretised into 100 elements
number of segements = 3
layers of each segement = 2
units of each layer = 5

### implementations

## without
# 0: without middle state, 
![](./images/middle_state_shape_(100,%200)_epochs_1000_learning_rate_0.001_times_0.png)
![](./images/middle_state_shape_(100,%200)_epochs_1000_learning_rate_0.001_times_1.png)
![](./images/middle_state_shape_(100,%200)_epochs_1000_learning_rate_0.001_times_2.png)

## case 1
minimize the difference of the naber segement's last and first units
![](./images/middle_state_shape_(100,%202)_epochs_1000_learning_rate_0.001_times_0.png)
![](./images/middle_state_shape_(100,%202)_epochs_1000_learning_rate_0.001_times_1.png)
![](./images/middle_state_shape_(100,%202)_epochs_1000_learning_rate_0.001_times_2.png)

## case 2
for each segement, introduce a middle state n*1
![](./images/middle_state_shape_(100,%201)_epochs_1000_learning_rate_0.001_times_0.png)
![](./images/middle_state_shape_(100,%201)_epochs_1000_learning_rate_0.001_times_1.png)
![](./images/middle_state_shape_(100,%201)_epochs_1000_learning_rate_0.001_times_2.png)

## case 3
for each segement, introduce a middle state n*5
![](./images/middle_state_shape_(100,%205)_epochs_1000_learning_rate_0.001_times_0.png)
![](./images/middle_state_shape_(100,%205)_epochs_1000_learning_rate_0.001_times_1.png)
![](./images/middle_state_shape_(100,%205)_epochs_1000_learning_rate_0.001_times_2.png)
