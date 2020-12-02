# 安装

目前 *Transformers*已经在Python 3.6+，PyTorch 1.1.0+和TensorFlow 2.0+上面完成了测试。

通常情况下我们会建议你在一个新的虚拟环境中来安装*Transformers*。如果你对于Python虚拟环境并不熟悉，那么你可以点击[此处](https://mp.weixin.qq.com/s/KOFvW5UpAzqJKchCkfv7JA)通过`Conda`来创建一个符合上述条件的Python环境，并且将环境激活进行使用。

完成上述过程后，你就可以开始通过`pip install`来安装*Transformers*了。当然，如果你不喜欢这种安装方式，也可以通过源码来进行安装。

## 1 通过`pip install`进行安装

首先，你需要自行的安装TensorFlow2.0或者PyTorch。你可以分别点击[TensorFlow 安装](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) 和[PyTorch安装](https://pytorch.org/get-started/locally/#start-locally)来阅读相关平台下的安装命令完成安装。当你你完成TensorFlow2.0或者是PyTorch的安装后，通过如下命令就能够完成对于*Transformers*的安装：

```shell
pip install transformers
```

同时，你也可以在仅支持CPU的主机上以如下命令来同时完成*Transformers*和PyTorch的安装：

```shell
pip install transformers[torch]
```

或者，*Transformers*和TensorFlow 2.0的安装：

```shell
pip install transformers[tf-cpu]
```

在这之后，你可以通过运行如下命令来快速的检查*Transformers*是否被安装成功：

```shell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

第一次运行这段代码时会下载一个对应的预训练模型，然后就会输出如下所示的信息：

```shell
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

（注意，如果是TensorFlow的话，可能还会输出其它额外的信息。）

## 2 通过源码进行安装

如果是通过源码进行安装的话，首先需要克隆我们对应的代码仓库，然后完成安装：

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

接着，同样来验证是否安装成功：

```shell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

## 3 模型缓存路径

*Transformers*提供了大量的预训练模型，当你在第一次使用某些模型时，其对应的预训练模型就会被缓存到本地。此时，除非你在`from_pretrained()`方法中通过参数`cache_dir=...`来指定相应的缓存路径，否则默认情况下都会被下载到 `TRANSFORMERS_CACHE`环境变量所对应的路径中。`TRANSFORMERS_CACHE`的默认值是PyTorch的缓存路径加上`/transformers`。如果你并没有安装PyTorch，那么`TRANSFORMERS_CACHE`的默认值将以如下优先级选择：

- `TORCH_HOME`
- `XDG_CACHE_HOME`+`/torch`
- `~/.cache/torch/`

因此，如果你没有进行任何的指定，缓存模型都将被下载到目录`~/.cache/torch/transformers`中。

### 3.1 模型下载注意事项

如果你希望通过我们的托管平台来使用一个非常大的预训练模型（例如通过CI来进行超大规模的产品部署），最好的方法就是将它缓存到你自己的终端上。同时，如果在这过程中遇到了任何问题，请直接联系我们。

## 4 在移动终端上使用*Transformers*

如果需要在移动设备上使用*Transformers*，那么你可以去[swift-coreml-transformers](https://github.com/huggingface/swift-coreml-transformers) 查看相关内容。

在那里我们提供了一系列的工具来将PyTorch和TensorFlow 2.0训练好的预训练模型（目前支持GPT-2，DistilGPT-2，BERT和DistilBERT）转换成可以运行在iOS设备上的CoreML模型。

当然，在未来我们还会支持将PyTorch或TensorFlow中的预训练模型和微调模型无缝迁移到CoreML或者是其它原型产品中。



## [<主页>](README.md)  