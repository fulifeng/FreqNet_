# FreqNet
This is the implementation of FreqNet in our paper "**Time Horizon-aware Modeling of Financial Texts for Stock Price Prediction**" at [ICAIF'21 – 2nd ACM International Conference on AI in Finance](https://ai-finance.org/conference-program/).

## Requirements
- Python == 2.7
- Keras == 1.2.0
- Theano == 0.9

## Run (Tweet dataset)

```
python isfm_train.py -t ../dataset/tw -d ../dataset/stocknet.npy -td 200 -hd 16 -f 5 -s 5 -tn 5
```
- Hyperparameters:
```
  Namespace(att_dim=50, data_file='../dataset/stocknet.npy', freq_dim=5, hidden_dim=16, learning_rate=0.01, niter=2000, nsnapshot=40, step=5, text_dim=200, text_file='../dataset/tw', text_num=5)
```

- Resuts:
```
2000  training error  0.020432354578405837
 val error  0.03872323613163087
 test error  0.027221137708653517
MSE: (65.40997023809524, 106.27279, 13.379666183728926)
Training duration (s) :  824.857722998
best iteration  1240
smallest val error  0.03276136459362498
associated tes error  0.024872010802031493
associated tes mse  (68.8524712138954, 110.79892, 15.442392344165928)
```

<!---news:
  Namespace(att_dim=50, data_file='../dataset/news.npy', freq_dim=5, hidden_dim=64, learning_rate=0.01, niter=1500, nsnapshot=40, step=5, text_dim=300, text_file='../dataset/news', text_num=3)
-->

This implementation is based on [SFM](https://github.com/z331565360/State-Frequency-Memory-stock-prediction), we sincerely appreciate the authors of SFM.
