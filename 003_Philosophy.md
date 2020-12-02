# 设计理念

**专为以下人员所设计的*Transformers*：**

- 寻找学习、使用和扩展大规模transformers模型的NLP研究者和教育者；
- 希望对模型进行微调或者是直接产品化从业者；
- 想要直接使用预训练模型来完成相应NLP任务的工程师；

***Transformers*的设计主要遵循了如下两大设计理念：**

- 尽可能简单快速的进行使用：
  - 我们极大程度上的减少了面向用户需要学习的若干抽象化概念。事实上在*Transformers*中，我们只需要通过三个标注的类就能够完成对于一个模型的使用，其分别是：[configuration](https://huggingface.co/transformers/master/main_classes/configuration.html), [models](https://huggingface.co/transformers/master/main_classes/model.html) 和 [tokenizer](https://huggingface.co/transformers/master/main_classes/tokenizer.html)。
  - 所有的类都可以通过一个简单和统一的方法来进行实例化。在*Transformers*中，我们可以通过`from_pretrained()`实例化方法来对相应的类进行实例化，并且该实例化方法同时还会考虑到对应数据的下载（如果需要的话），包含相应的类实例，相关数据（超参数配置文件、词表和预训练模型）。这些文件既可以从 [Hugging Face Hub](https://huggingface.co/models)下载（默认），也可以指定自己本地的文件。
  - 除了第一点中提到的三个基本类，*Transformers*还提供了两个额外的API：`pipeline()`和`Trainer()/TFTrainer()`。通过`pipeline()`，我们可以快速的搭建一个相关的网络模型，同时还会完成相应文件的自动下载；通过`Trainer()/TFTrainer()`，我们可以快速的在一个给定的任务场景中完成相应模型训练或者是微调。
  - 最后，*Transformers***并不是**针对于神经网络的一个模型化的工具（building blocks）。这意味着如果你想直接在*Transformers*的基础上进行改进或者拓展是非常困难的。你需要做的应该是使用标准的Python/PyTorch/TensorFlow/Keras框架并继承其中相应的基类来完成你需要的工作。
- 提供最前沿（state-of-the-art）网络模型，并接可能的使其性能接近于论文中的模型：
  - 对于每一种网络模型，*Transformers*至少提供了一个示例来复现原始论文呈现的结果；
  - *Transformers*中所实现的代码做到了尽可能的接近于原生框架风格的代码，这也意味着部分PyTorch代码可能不是那么的*pytorchic*，这是因为*Transformers*考虑到了模型在PyTorch和TensorFlow之间的转换。对于用TensorFlow实现的模型来说，同样如此。

**一些未来的目标：**

- 尽可能一致的来实现模型的内部结构：
  - 仅通过一个API就能够获得网络所有的隐含状态和注意力权重；
  - 标准化Tokenizer和基类模型的API，使其能够轻松在不同模型之间转换；
- 为一些具有潜力能力想法加入一个主观选项来进行模型微调和探索：
  - 实现一种简单的方法来在词表和词嵌入中加入新的token；
  - 实现一种简单的方法来对transformer的多头进行遮蔽和修剪；
- 在PyTorch和TensorFlow2.0中能够更大程度上的达到自由转换的目的，例如通过PyTorch来对模型进行训练，但是可以通过TensorFlow来进行预测。

## 主要概念

*Transformers*中的每一个网络模型都是在三类基本的类上实现的：

- **Model classes**：例如 [`BertModel`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertModel)模型，它包含了超过30+的PyTorch模块（`torch.nn.Module`）或者Keras模块（`tf.keras.Model`)。
- **Configuration classes**：例如 [`BertConfig`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertConfig)，它存储了BERT模型所需要的所有参数，因此你不需要自己再进行一次实例化。尤其是当你在使用某个预训练模型（不需要做任何修改）时，当你定义完这个模型后，其对应的`Config`类就被实例化了。
- **Tokenizer classes**：例如 [`BertTokenizer`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertTokenizer)，它存储了BERT模型所需要的词表，并且提供了方法来编码和解码输入输出的字符串。

所有上面列出的这三大类都可以通过各自对应的`from_pretrained()`方法和`save_pretrained()`方法来进行实例化和保存模型：

- `from_pretrained()`可以通过载入[已存在的](https://huggingface.co/transformers/master/pretrained_models.html)预训练模型来实例化对应的模型类（例如`BertModel`）、配置类（例如`BertConfig`）和分词器类（例如`BertTokenizer`）。

- `save_pretrained()`可以用来保存上述的三个实例化后的类模型，并且还可以再次通过`from_pretrained()`进行复用。

  

## [<主页>](README.md)  