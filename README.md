
## 短文本语义一致性分析
对两个短文本，判断对应的语义是否一致

## 训练语料
LCQMC 数据集，样例数据如下：
```shell
喜欢打篮球的男生喜欢什么样的女生	爱打篮球的男生喜欢什么样的女生	1
我手机丢了，我想换个手机	我想买个新手机，求推荐	1
大家觉得她好看吗	大家觉得跑男好看吗？	0
求秋色之空漫画全集	求秋色之空全集漫画	1
```

### 训练
```

gswyhq@gswyhq-PC:~/github_projects/semantic-analysis$ python3 run.py --num_epochs 2 --batch_size 10 --verbose --mode train --model lstm

gswyhq@gswyhq-PC:~/github_projects/semantic-analysis$ python3 run.py --num_epochs 2 --batch_size 10 --verbose --mode train --model esim


```



## 参考资料

[lstm](https://www.jianshu.com/p/a649b568e8fa)

[esim](https://blog.csdn.net/wcy23580/article/details/84990923)

[lstm](https://github.com/zqhZY/semanaly)

[esim](https://blog.csdn.net/wcy23580/article/details/84990923)

[esim](https://github.com/yuhsinliu1993/Quora_QuestionPairs_DL/blob/master/ESIM.py)

[tree-lstm](https://blog.csdn.net/sinat_30665603/article/details/79520012)

[bilstm](https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py)

[drcn](https://github.com/ghif/drcn/blob/master/drcn.py)
