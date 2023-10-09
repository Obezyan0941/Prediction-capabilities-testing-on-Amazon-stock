# Prediction-capabilities-testing-on-Amazon-stock

In this project I tried to compare the predicting possibilities of the two most popular sequence predicting neural networks: Transformer and Seq2Seq models.
Originally it was just an attempt to learn the LSTM model with the help of Greg Hogg's lesson: https://www.youtube.com/watch?v=q_HS4s1L8UI&ab_channel=GregHogg
Amazon stock's dataset is also from Greg Hogg's lesson. However the project has been becoming broader than just a lesson for me when I started discovering the features of Seq2Seq and Transformer.

Null hypothesis was that an accuracy of Transformer should be much better than Seq2Seq's as it is based on attention mechanism which, as I supposed, could memorise some patterns in stocks's dynamics.
However it has turned out that Transformer showed lower accuracy than Seq2Seq. Transformer - 29,3% and Seq2Seq - 38,15%

The excel file compares how both algortihms has predicted the trends: uptrend, downdtrend and consolidation. The surprising result is that Transformer has predicted well only uptrends, while Seq2Seq had good accuracy only in downtrends. 

In the end, I assumed thay can be combined to potentially show a really good prediction accuracy, since they work better at entirely different sequences of time-series. 
This topic needs more research, however this is my result for now. Feel free to give comments and suggestions!


