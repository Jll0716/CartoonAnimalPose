# CartoonAnimalPose
1. 如果不想创立自己的数据集，`dataset/datasetbuild` 里的内容可以忽略。

2. 运行代码前，需要修改本地的 `cocoeval.py` 文件，设置如下：
   ```python
   self.kpt_oks_sigmas = np.array([.25, .25, .72, .62, .62, .35, .35, .35, .35, .89, .89, .89, .89, .62, .62, .62, .62, 1.07, 1.07, 1.07, .89]) / 10.0
3. 你需要自己创建 `results`、`save_weights`、`checkpoints` 文件夹。
