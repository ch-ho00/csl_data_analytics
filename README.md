# CSL Data Analytic Competition 2020

For more details on the analysis of the data please refer to the [presentation](https://github.com/chanhopark00/csl_data_analytics/blob/master/CSL%20Data%20Analytics%20Case%20Competition%20-%20Airbnb.pdf)

## Input Data 

1. review.csv
   <img src="https://github.com/chanhopark00/csl_data_analytics/blob/master/img/1.png" >

2. listing.csv
   <img src="https://github.com/chanhopark00/csl_data_analytics/blob/master/img/3.PNG" >


## Processed Data

For each comment, we are able to extract positive/negative keywords based on the word embeddings of comments.
Here we have extracted words based on different categories. For example for negative keywords, we use words like "small","dirty","inconvenient" and etc. Through such method, we are able to label each comment by positive and negative; for the case of negative comments we further catgeorize with a particular category.

<img src="https://github.com/chanhopark00/csl_data_analytics/blob/master/img/2.png" >

Based on the different categories of positive/negative keywords, we are able to see what the general negative comments are for different neighboorhod. These leads to the following map of Hong Kong. Note that this can also be applied to smaller regions as the diagram below.

<img src="https://github.com/chanhopark00/csl_data_analytics/blob/master/img/4.png" width="400" >

Furthermore, we were also able to find out that there is a consistency in the type of negative comment that people leave.
This means that if one person has a negative comment on a particular category, then it is likely that the person will leave a negative review of the same category.

<img src="https://github.com/chanhopark00/csl_data_analytics/blob/master/img/5.png" width="400" >
