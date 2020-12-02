# 快速指引

让我们快速的来看一看*Transformers*库的特性。通过*Transformers*库，你可以下载一些预训练模型来完成自然语言理解中的常见任务，例如：文本语义分析、自然语言生成、语义推断和文本翻译。

首先，我们会看到如何轻松的通过管道（pipeline API）来快速的使用这些预训练模型进行推断预测。然后，我们将会深入的探究*Transformers*是如何来利用这些模型来处理你的数据的。

**注意：**示例中列出的所有代码均可以通过左上角的按钮在Pytorch和TensorFlow之间切换。如果没有按钮，这表示该代码可以在两个平台上通用。(译者注：在译文中两种代码贴在了同一处，并进行了注释)

## <span id = '011'>1 通过管道来开始我们的第一个任务</span>

使用预训练模型最简单的方法就是通过`pipeline()`来进行实现。

通过*Transformers*你可以完成如下所示的任务建模：

- 语义分析：一段文本为正面倾向还是负面倾向？
- 文本生成（英语）：给出部分提示，模型将自动为其生成后续文本。
- 命名体识别：对于输入的文本，模型会标记对应的文本序列所表示的实体（人物、地点等）。
- 问题回答：将文章和对应问题输入到模型中，模型输出问题所对应的回答。
- 空白推断：输入一段某些词语被遮蔽（如用`[MASK]`进行替代）的文本，模型推断处遮蔽部分的内容
- 摘要生成：对输入的长文本输出其对应的摘要内容。
- 序列翻译：将一段文本序列翻译成另外一种语言。
- 特征抽取：输入一段文本，返回一个能表示该文本的特征向量。

下面，然我们一起来看看如何通过*Transformers*来完成语义分析这一任务（其它任务介绍可查看[任务总结-----------]()）。

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```

当你第一次键入上述代码并运行时，其对应的预训练模型和分词器（tokenizer）所需要的词表都会被下载并缓存到本地，对于这部分我们稍后再进行介绍。但是首先需要明白的就是分词器的作用是将输入的文本序列进行预处理（译者注：中文的话是切分成以字为单位，英文的话就是单词），然后再将其输入到模型中用于预测。在*Transformers*中，可以通过管道来将所有的处理步骤结合到一起，同时还能使得我们能够直观得看到预测后的结果。

```python
classifier('We are very happy to show you the 🤗 Transformers library.')

[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
```

令人振奋的是，还可以直接通过`list`来输入多个样本。这些样本在经过预处理后，将会作为一个batch的数据输入到模型中，然后返回一个包含所有结果的字典。

```python
results = classifier(["We are very happy to show you the 🤗 Transformers library.","We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

你可以看到上面的第二个句子已经被分类成为Negative的标签（注意，模型只会将其分类成正面或者是负面），但是它的得分却非常的偏中性。

默认情况下，在上述过程中`pipeline()`下载的都是一个叫做`distilbert-base-uncased-finetuned-sst-2-english`的预训练模型。我们可以在这个[页面](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)中查看到更多关于该与训练模型的相关信息。这个预训练模型是通过 [DistilBERT architecture](https://huggingface.co/transformers/model_doc/distilbert.html) 网络所训练得到的，并且已经在数据集SST-2上进行了微调以便于更好的进行情感分类任务。

现在假设我们想要使用另外模型（例如一个已经在法文语料上训练好的模型），那么我们只需要可在页面中[model hub](https://huggingface.co/models) 以关键词"French"和“text-classifiction”进行搜索，它就会返回相应的模型建议。例如在这个示例中就会得到`nlptown/bert-base-multilingual-uncased-sentiment`建议。下面然我们看看如何来使用这一模型吧。

你可以通过`pipeline()`直接将这一模型的名字作为参数传入：

```python
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```

现在，上面定义好的这个分类器就能够完成对于英语、法语、荷兰语、德语、意大利语和西班牙语文本的情感分类工作。当然，还可以将参数`model=`替换为自己保存在本地的已经预训练好的模型（见后文）来完成上述工作。同时，你还可以传入一个实际的模型对象和其对应分词器，从头开始训练

对于后面这种方式，我们需要传入两个类对象到`pipeline()`中。第一个是类[`AutoTokenizer`](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoTokenizer)，我们将用它来下载我们指定模型所对应的分词器，并实例化我们指定的模型。第二个是类 [`AutoModelForSequenceClassification`](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification)（或者是[`TFAutoModelForSequenceClassification`](https://huggingface.co/transformers/model_doc/auto.html#transformers.TFAutoModelForSequenceClassification)，如果你正在使用TensorFlow），我们将通过这个类来完成其对应模型的下载。

**注意：**如果我们需要使用*Transformers*来完成其它相关任务，那上述配置将会发生变化。在[任务总结](https://huggingface.co/transformers/task_summary.html)页面，我们将看到哪种任务需要那种配置。

```python
# 针对Pytorch的代码
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#针对TensorFlow的代码
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
```

接下来，我们需要通过[`from_pretrained()`](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification.from_pretrained)方法来下载前面我们搜索到的模型，以及其对应的分词器。同时，你还可以将`model_name`替换为其它任何你可以在model hub所能搜索到的模型，对此你可以放心的进行尝试。

```python
#针对Pytorch的代码
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

#针对TensorFlow的代码
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

接着，你就可以像上面的示例代码一样，对你的目标文本进行情感分类。

如果你在model hub里不能够找到与你的数据类似的预训练模型（译者注：这里应该指的是没有在特定任务中经过微调后的模型，而不是指原始语料下训练得到的通用模型），那么你就需要在自己的数据集上进行微调训练。对此，我们专门提供了一些[示例脚本](https://huggingface.co/transformers/examples.html)来完成这些任务。一旦你在自己的数据上完成微调之后，千万不要忘了将其分享到model hub社区。详细分享上次步骤可参加[此处](https://huggingface.co/transformers/model_sharing.html)。

## <span id = '012'>2 预训练模型的内幕</span>

现在，让我们来看看在使用`pipeline()`的过程中，其背后到底发生了什么事情。正如我们在上面所说到的，模型和分词器都是通过对应的`from_pretrained()`方法所建立的：

```python
# 针对Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#针对TensorFlow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

```

### 2.1 使用分词器

在前面我们提到，分词器的作用就是用来对输入的原始文本进行处理，其具体步骤为：

首先，分词器会将输入的文本分割成一个一个的词（或者是标点），通常我们把这一过程叫做*tokens*。同时，我们也提供了很多不同的文本分割规则来处理你自己的数据，你可以在页面[tokenizer summary](https://huggingface.co/transformers/master/tokenizer_summary.html)中找到更多这方面的介绍。因此这也是为什么我们在载入分词器（tokenizer）的时候需要指定一个模型名（`model_name`）来实例化分词器了，因为我们必须要确保接下来我们在自己的任务中所使用的分词器要和载入的预训练模型使用的分词器是同一个。

其次，分词器接着会将分割后的词（译者注：对于中文来说就是字）转换成词表中的索引，这样做的目的就是将文本转换成向量，然后再喂给模型。为了实现这一目的，我们还需要一份额外的词表，好在当我们通过方法`from_pretrained()`实例化时，这个词表就在后台下载了。

如果需要对一段给定的文本进行tokenize，那么只需要运行如下代码即可：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(inputs)

#
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

如上所示，运行后的结果将会返回一个字典，其包含对应词在词表中的索引（ [ids of the tokens](https://huggingface.co/transformers/master/glossary.html#input-ids)）；以及模型需要用到的注意力掩码（ [attention mask](https://huggingface.co/transformers/master/glossary.html#attention-mask)）。

除此之外，如果你有多句样本，你还可以将它们作为一个batch通过一个`lsit`来传入。此时，你应该指定需要将这一批的样本以多大的长度进行截取（译者注：**如过没有指定则默认以最大的长度**），对于小于最大长度的样本则会进行填充（译者注：以0进行填充）。

```python
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    return_tensors="pt",
    #max_length=5   自己指定最大长度
)

#如果是在TensorFlow环境中，只需要将 return_tensors设置为 "tf"即可
```

整个填充的过程完全是根据模型的需要，自动选择在哪边进行填充的（在这个示例中为右边），不需要我们自己去设定。在运行完上面的代码后，就能得到如下的结果：

```python
for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")
    
#
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
```

你还可以通过点击这个页面来获取更多关于[tokenizers](https://huggingface.co/transformers/master/preprocessing.html)的信息。

### 2.2 使用模型

一旦你通过tokenizer完成了对数据的预处理工作，那么你就可以直接将其输入到对应的模型中了。正如我们上面示例中所提到的，在完成tokenize这一步后我们就会得到所有模型需要输入的东西。如果你使用的是TensorFlow，那么只需要将` tokenizer()`返回后的结果喂入模型即可，如果是Pytorch则需要用`**`来进行解包。

```python
#针对Pytorch
pt_outputs = pt_model(**pt_batch)

#针对TensorFlow
tf_outputs = tf_model(tf_batch)
```

在*Transformers*中，模型的输出结果都是`tuples`类型的，这里我们输出模型最后一层的结果：

```python
print(pt_outputs)
#
(tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>),)
```

可以看到，模型最后返回仅仅只是返回了`tuples`中的一个元素（最后一层），但我们同样可以返回多层的输出结果，而这也是为什么模型返回的是一个`tuples`的原因。

**注意：**所有的*Transformers*模型（Pytorch和TensorFlow）返回的最后一层指的都是在最后一个激活函数前的值（例如SoftMax），因为通常来说正真意义上的最后一层都是和损失函数结合在一起的。

译者注：如果是在分类任务中，最后一层指的就是logits，即最后一个全连接层的线性输出结果（没有经过激活函数）。

下面，让我们将上面的输出结果输入到SoftMax激活函数中来得到一个预测的概率分布：

```python
#针对Pytorch
import torch.nn.functional as F
pt_predictions = F.softmax(pt_outputs[0], dim=-1)

#针对TensorFlow
import tensorflow as tf
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
```

这样，我们就能够得到预测的结果：

```python
#针对Pytorch
tensor([[2.2043e-04, 9.9978e-01],
        [5.3086e-01, 4.6914e-01]], grad_fn=<SoftmaxBackward>)

#针对TensorFlow
tf.Tensor(
[[2.2042994e-04 9.9977952e-01]
 [5.3086340e-01 4.6913657e-01]], shape=(2, 2), dtype=float32)
```

同时，如果你还有数据样本对应的标签，那么你还可以将它输入到模型中，模型就会返回一个包含损失值和最后一层的输出值：

```python
#针对 Pytorch
import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
print(pt_outputs)
#
(tensor(0.3167, grad_fn=<NllLossBackward>), tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>))
#针对 TensorFlow
import tensorflow as tf
tf_outputs = tf_model(tf_batch, labels = tf.constant([1, 0]))
```

在*Transformers*中，所有模型的实现都是基于[`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)或者[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的，因此你一样可以将它们放到你平时训练的代码中。同时，*Transformers*还提供了 [`Trainer`](https://huggingface.co/transformers/master/main_classes/trainer.html#transformers.Trainer) （[`TFTrainer`](https://huggingface.co/transformers/master/main_classes/trainer.html#transformers.TFTrainer) 如果你使用的是TensorFlow）类来帮助我们训练自己的数据，包括分布式训练和混合精度等。关于更多训练相关的介绍，请点击进入页面 [training tutorial](https://huggingface.co/transformers/master/training.html)。

**注意：**Pytorch模型输出的是特殊的数据类型，因此你可以利用IDE来自动补全对应属性。

译者注：在Pytorch中，模型输出的结果都是Tensor张量，因此我们可以通过`.item()`属性来将其输出结果转化为数值类型。例如`print(pt_outputs[0].item())`。

### 2.3 保存模型

一旦你完成模型的微调后，你可以通过下面的方法来保存对应的tokenizer和模型：

```python
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

译者注：`save_directory`为指定保存的文件夹路径，没有将会自动创建。

在这之后，你可以向前面介绍的那样，通过 [`from_pretrained()`](https://huggingface.co/transformers/master/model_doc/auto.html#transformers.AutoModel.from_pretrained)方法来载入训练好的这个模型进行复用。而此时你所需要传入`from_pretrained()`的就不是模型的名字了，而是你上面保存模型对应的路径。同时，在*Transformers*中一个酷炫的功能就是，不管你上面保存好的预训练模型是用`Pytorch`还是`TensorFlow`训练的，你都能将其载入进来用于你接下来的工作中。

如果你要载入一个由`Pytorch`保存的模型到`TensorFlow`的环境中，那么只需要像下面这样使用即可：

```python
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```

如过你要载入一个由`TensorFlow`保存的模型到`Pytorch`的环境中，那么只需要像下面这样使用即可：

```python
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```

最后，你还可以让模型返回所有隐藏状态和注意力权重，如果你需要的话：

```python
#针对 Pytorch
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = pt_outputs[-2:]

#针对 TensorFlow
tf_outputs = tf_model(tf_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = tf_outputs[-2:]

```

### 2.4 触摸代码

在*Transformers*中， `AutoModel` 和`AutoTokenizer`这两个类仅仅只是用于自适应匹配所载入的模型背后所对应的网络架构。也就是说，不管你的预训练模型是通过*Transformers*中的哪种网络架构训练得到的，你都可以通过这两个类来载入与训练好的模型，而不用显示的指定。因为在这背后，*Transformers*为每一种网络模型（结构）都定义好了一个模型类，因此你可以轻松的访问和调整代码。

例如在我们前面的示例过程中，我们所使用的模型的名字是`distilbert-base-uncased-finetuned-sst-2-english`，这意味着该模型是通过[DistilBERT](https://huggingface.co/transformers/master/model_doc/distilbert.html) 这一网络架构训练得到的。但是，我们同样只需要通过类[`AutoModelForSequenceClassification`](https://huggingface.co/transformers/master/model_doc/auto.html#transformers.AutoModelForSequenceClassification)（或者 [`TFAutoModelForSequenceClassification`](https://huggingface.co/transformers/master/model_doc/auto.html#transformers.TFAutoModelForSequenceClassification) 如果你使用的是TensorFlow）就能够将这个模型给载入进来，并且自动匹配到[`DistilBertForSequenceClassification`](https://huggingface.co/transformers/master/model_doc/distilbert.html#transformers.DistilBertForSequenceClassification)这个类。关于这其中的细节之处，你可以点击前面这个类对应的超链接或者是查看相应的源码来进行了解。

当然，你同样可以自己显示的来进行指定模型所对应的类，例如像下面这样使用：

```python
# 针对 Pytorch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

#针对 TensorFlow
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

### 2.5 定制你自己的模型

如果你想更改模型中的结构，你还可以定义自己类来对网络中的参数进行配置。对于每一个网络结构来说，它都对应着一个相应配置类（例如 DistilBERT,它的配置对应的就是[`DistilBertConfig`](https://huggingface.co/transformers/master/model_doc/distilbert.html#transformers.DistilBertConfig)这个类），这个类允许我们来指定网络结构中的任意一个参数，例如隐藏层维度（hidden dimension）、丢弃率等。如果你还想做一些核心部分的修改，例如隐藏层个数（hidden size），那么你将需要从头开始利用你自己搜集的数据集来训练这个模型，这就意味着你不能够使用现有的预训练模型。完成这些核心部分的修改后，模型就会按照你修改的配置被重新进行实例化然后训练。

下面示例是一个从头开始训练的DistilBERT网络模型，其中Tokenizer所用到的词典并没有发生变化，因此我们还是可以通过`from_pretrained()`方法进行载入。

```python
# 针对Pytorch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)

# 针对TensorFlow
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification(config)
```

对于那些细微的改动自处（例如最后的分类数），你同样还是可以使用这些预训练模型。下面，让我们载入一个预训练模型来完成一个十分类的任务。此时我们需要修改一些配置，也就是除了分类数之外，其它的都保持默认不变。对于这步，你可以简单的通过`from_pretrained()`方法传递进去对应的参数即可：

```python
# 针对 Pytorch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

#针对 TensorFlow
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```



## [<主页>](README.md)  

