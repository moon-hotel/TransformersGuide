# 任务总结

在这个页面里将会展示*Transformers*库中最常使用到的一些模型的示例。这些模型都被允许配置不同的参数，也就是说在实际的使用中可以灵活根据需要改动各项配置。下面将会展示一些*Transformers*中最简单的任务，例如：问题回答、序列分类和命名体识别等。

在使用上述的这些模型中，我们可以使用`AutoModel`这个类来进行建模。这个类能够通过给定的模型名称（或预训练模型的路径）自动匹配上正确的网络架构。更多关于该类的介绍信息可以点击[`AutoModel`](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModel) 。同时，你还可以根据你实际的项目需求来改动*Transformers*中的代码。

需要注意的是，为了能够使得模型在你的任务上有着较好的表现，你必须根据任务（问答或分类）对应的模型名称（或预训练模型的路径）来载入模型。因为你所载入的预训练模型通常都是先大的语料上训练后，然后在特定语料下微调后的模型，所以你必须确定你加载的模型是符合你的任务场景的。因此这也就意味着：

- 并不是所有的模型都经过了微调。如果需要用你自己的数据集来对模型进行微调，你可以通过运行[examples](https://github.com/huggingface/transformers/tree/master/examples)目录中的`run_$TASK.py `脚本来进行。
- 虽然微调得到的模型用的是某个特定场景中的数据集训练的，但是该数据集和你任务中所使用到数据集可能并没有任何交集。因此，正如上面提到的，你仍旧可以在自己的数据集进行微调。

为了能够方便的使模型完成推测（预测）任务，*Transformers*库还实现了一些非常简单的调用接口：

- Pipelines：非常简单易用但抽象的接口，仅仅只需要两行代码就能够完成推理任务。
- Direct model use：略微冗长但更加灵活与强大的使用方法，直接通过书写对应的代码来完成推理任务。

上面提到的这两种方式，在下面的示例中都会进行介绍。

注意：

所有在这里展示的案例其使用的预训练模型都是在某个特定数据中微调后的结果。如果选择载入的是一个没有经过微调后的模型，那么得到的模型仅仅只会包含最基本的transformer层所需要的参数，因此你需要根据你自己的任务场景来额外的定义一些网络层然后随机初始化这部分参数。（译者注：不管你载入的是不是一个经过微调后的模型，只要它最后的输出形式跟你的不一样，你就需要自己在额外的定义一些网络层，然后随机初始化这些参数，最后通过你自己的数据集进行微调。例如你载入的模型原本是进行5分类的，而你自己的任务却是10分类的，此时你就需要重新定义一个网络层。）

## <span id ='1001'>1 序列分类任务</span>

序列分类（Sequence Classification）任务指的就是将输入的序列分类到给定的类别数，其中最常见的便是GLUE中的数据集（译者注：GLUE的全称是General Language Understanding Evaluation，它是一些列分类分类数据集的集合，如SST、CoLA等）。如果你希望在GLUE数据集上微调一个分类任务模型，那么你可以通过使用脚本 [run_glue.py](https://github.com/huggingface/transformers/tree/master/examples/text-classification/run_glue.py) 和 [run_pl_glue.py](https://github.com/huggingface/transformers/tree/master/examples/text-classification/run_pl_glue.py) 或 [run_tf_glue.py](https://github.com/huggingface/transformers/tree/master/examples/text-classification/run_tf_glue.py) 来完成。

### 1.1 `pipelines`实现

下面是一个使用`pipelines`接口来做情感分析的例子：判定一个输入序列的情感倾向是正面的还是负面的。下面使用到的这个模型是在SST2数据集上经过微调后的结果：

```python
from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# label: NEGATIVE, with score: 0.9991

result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# label: POSITIVE, with score: 0.9999
```

可以看到，通过`pipeline`我们只需要传入对应模型（已经过微调）的名称就能够得到这个模型，进一步通过传入数据即可完成模型的推断任务。

### 1.2 搭建实现

接下来是一个通过*Transformers*来自己搭建一个分类模型的示例：判定两句话非否属于同一段落。要实现这么一个目的，我们只需要完成如下五个步骤即可：

1. 通过checkpoint名称来实例化一个tokenizer和model类。例如在下面的示例中通过“bert-base-cased-finetuned-mrpc”载入进来的就是一个BERT模型，以及对应的权重。
2. 根据模型指定的分割负（例如'SEP'）将两句话合并成一个序列，并且同时完成注意力掩码（attention masks）生成。（译者注：在下面的示例中这一步是通过`tokenizer`来完成的）。
3. 将构造完成的序列输入到模型中进行二分类（0 表示不是同一段落，1表示是同一段落）。
4. 通过softmax计算预测结果在每个类别上的概率值。
5. 输出结果。

```python
#针对于 Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#第一步
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=True)

#第二步
classes = ["not paraphrase", "is paraphrase"]
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

# 第三步
paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

#第四步
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

#第五步
# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
# not paraphrase: 10%
# is paraphrase: 90%

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
# not paraphrase: 94%
# is paraphrase: 6%
```



```python
# 针对于 TensorFlow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

#第一步
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=True)

#第二步
classes = ["not paraphrase", "is paraphrase"]
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="tf")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="tf")

#第三步
paraphrase_classification_logits = model(paraphrase)[0]
not_paraphrase_classification_logits = model(not_paraphrase)[0]

#第四步
paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
not_paraphrase_results = tf.nn.softmax(not_paraphrase_classification_logits, axis=1).numpy()[0]

#第五步
# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
# not paraphrase: 10%
# is paraphrase: 90%

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
# not paraphrase: 94%
# is paraphrase: 6%
```

可以发现，即使是使用*Transformers*提供的接口来自己搭建一个模型也是非常的简单。对于上面那个情感分类的例子，我们还可以通过如下代码来完成（译者注）：

```python
# 针对于Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",return_dict=True)

paraphrase = tokenizer(["I hate you","I love you"], return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()

print(paraphrase_results)
#[[0.9991129040718079, 0.0008870692690834403], [0.00013436285371426493, 0.9998656511306763]]
```

## <span id ='1002'>2 问题答案抽取</span>

问题答案抽取（Extractive Question Answering）的目的是从文本中抽出与给定问题相关的答案。其中一个常见的问题回答数据集就是SQuAD数据集。因此如果你想要在数据集SQuAD上做微调的话，可以通过 [run_squad.py](https://github.com/huggingface/transformers/tree/master/examples/question-answering/run_squad.py) 和[run_tf_squad.py](https://github.com/huggingface/transformers/tree/master/examples/question-answering/run_tf_squad.py)这两个脚本来实现。

### 2.1 `pipelines`实现

下面，我们通过`pipelines`来实现一个问题答案抽取的示例，同时使用到的这个模型已经在SQuAD数据集上进行了微调。

```python
from transformers import pipeline
nlp = pipeline("question-answering")
context = "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script."

result = nlp(question="What is extractive question answering?", context=context)
result = nlp(question="What is a good example of a question answering dataset?", context=context)
```

在执行完上述代码后，`result`将会变成一个包含问题答案、答案在给定文本中的的起始索引和结束索引的这么一个字典：

```python
Answer: 'the task of extracting an answer from a text given a question.', score: 0.6226, start: 33, end: 95
                
Answer: 'SQuAD dataset,', score: 0.5053, start: 146, end: 160
```

### 2.2 搭建实现

下面，我们再来通过*Transformers*提供的接口来自己搭建完成上述过程，其大致需要完成如下七个步骤：

1. 通过checkpoint名称来实例化一个tokenizer和model类。例如在下面的示例中通过“bert-base-cased-finetuned-mrpc”载入进来的就是一个BERT模型，以及对应的权重。
2. 给定一段文本以及提问。
3. 循环的将文本和提问拼接在一起（两者间以特定的分隔符标记）构造成序列，进而转换成token id。 
4. 将上一步构造得到的序列传入到模型中，然后模型将会返回计算出的答案起始位置和结束位置的logits。（译者注：其实这也相当于是一个分类任务，每个可能的位置就相当于是一个分类，所以需要返回logits）
5. 通过softmax来计算每个可能位置的概率。
6. 根据得到的答案在原文中的起始和结束位置，然后再从原始文本中获取得到答案。
7. 输出结果。

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

#第一步
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", return_dict=True)

#第二步
text = "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch."
questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

#第三步
for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #第四步
    result = model(**inputs)
    answer_start_scores, answer_end_scores = result.start_logits, result.end_logits
#    answer_start_scores, answer_end_scores = model(**inputs)
    #第五步
    answer_start = torch.argmax(answer_start_scores) 
    answer_end = torch.argmax(answer_end_scores) + 1 
    #第六步
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    #第七步
    print(f"Question: {question}")
    print(f"Answer: {answer}")

```

译者注：这里第四步下面的两行代码与原始指南中的代码不同，原始的为第四步下面的第三行代码，但那应该是错误的写法，笔者进行了修正。

通过以上步骤，我们就完成了通过*Transformers*来搭建一个问题抽取的推出模型。

# 语言模型

语言模型（Language Modeling）是指使模型拟合某个语料库的任务，而这个语料库可以是某个特定场景下的语料。几乎所有流行的基于transformers结构的模型都使用了多语种建模方式进行训练，例如基于掩体语言建模的BERT，基于因果语言建模的GPT-2。

同时，语言模型在完成预训练后也非常有用，例如将预训练后的模型迁移到其它特定场景中的任务中，而这也只需要先将模型在一个非常大的语料进行预训练，然后让这个训练好的模型在一个特定场景中的数据集（如：[LysandreJik/arxiv-nlp](https://huggingface.co/lysandre/arxiv-nlp)）上进行微调即可。

## <span id ='1003'>3 掩体语言模型</span>

掩体语言模型（Masked Language Modeling）指的是用一些特定的掩体token将序列中的部分token进行掩盖，然后让模型去预测被遮蔽部分内容的一个任务。这就使得要允许模型能够从掩盖部分的左边和掩盖部分的右边来考虑被掩盖的部分（即考虑上下文）。所以，这也就使得模型具备了一种坚实的基础，能够胜任那些需要理解上下文环境的下游任务（如SQuAD）。

### 3.1 `pipelines`实现

下面是一个使用`pipelines`接口来完成的一个掩体模型语言的预测任务：

```python
from transformers import pipeline
nlp = pipeline("fill-mask")
from pprint import pprint
pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
```

其中`nlp.tokenizer.mask_token `部分表示将原本该处的文字给遮挡住。

待模型运行完成后，我们将会看到如下所示的结果：

```python
[{'score': 0.17927460372447968,
  'sequence': '<s>HuggingFace is creating a tool that the community uses to '
              'solve NLP tasks.</s>',
  'token': 3944,
  'token_str': 'Ġtool'},
 {'score': 0.1134939044713974,
  'sequence': '<s>HuggingFace is creating a framework that the community uses '
              'to solve NLP tasks.</s>',
  'token': 7208,
  'token_str': 'Ġframework'},
 {'score': 0.05243545398116112,
  'sequence': '<s>HuggingFace is creating a library that the community uses to '
              'solve NLP tasks.</s>',
  'token': 5560,
  'token_str': 'Ġlibrary'},
 {'score': 0.03493543714284897,
  'sequence': '<s>HuggingFace is creating a database that the community uses '
              'to solve NLP tasks.</s>',
  'token': 8503,
  'token_str': 'Ġdatabase'},
 {'score': 0.02860247902572155,
  'sequence': '<s>HuggingFace is creating a prototype that the community uses '
              'to solve NLP tasks.</s>',
  'token': 17715,
  'token_str': 'Ġprototype'}]
```

从上述输出的结果可以看出，被掩盖部分的单词最有可能的就是`tool`这个词。

### 3.2 搭建实现

接下来，我们就来通过*Transformers*提供的API来搭建一个掩体模型。通常，要实现上述的过程需要完成以下6个步骤：

1. 通过checkpoint名称来实例化一个tokenizer和model类。例如在下面的示例中通过“distilbert-base-cased”载入进来的就是一个DistilBERT模型，以及对应的权重。
2. 初始化一个序列，然后用`tokenizer.mask_token`来替换掉需要遮蔽的单词。
3. 通过` tokenizer.encode`对上面初始化的序列进行编码，得到一个包含每个单词在此表中位置的索引列表；同时需要得到被遮蔽单词在这个列表中的位置。
4. 预测得到遮蔽位置处的logits向量，该向量的维度同词表维度的大小相等。也就是是说某个维度的值越大，其在词表中对应位置的词约有可能出现在被遮挡的位置。
5. 可以通过PyTorch中的`topk`或者是Tensorflow中的`top_k`来取前`k`个最又可能的单词。
6. 将2中序列里面的被掩盖的单词替换成预测得到的单词，并输出。

```python
# 针对于 PyTorch
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
	# 第一步：
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased", 	return_dict=True)
	# 第二步：
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help" \
           f" {tokenizer.mask_token} our carbon footprint."
    # 第三步：
input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
	# 第四步：
token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]
	# 第五步：
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#[4851, 2773, 9711, 18134, 4607]
top_5_score = torch.topk(mask_token_logits, 5, dim=1).values[0].tolist()
#[13.97828197479248, 11.244152069091797, 10.893712043762207, 10.345271110534668, 10.284646987915039]
 	# 第六步：
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
```

```python
# 针对于 Tensorflow
from transformers import TFAutoModelWithLMHead, AutoTokenizer
import tensorflow as tf
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = TFAutoModelWithLMHead.from_pretrained("distilbert-base-cased", return_dict=True)
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
input = tokenizer.encode(sequence, return_tensors="tf")
mask_token_index = tf.where(input == tokenizer.mask_token_id)[0, 1]
token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
```

在完成上述步骤后，就能够得到如下所示的输出结果，并且其可能性从上到小一次降低：

```python
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
```

## <span id ='1004'>4 因果语言模型</span>

所谓因果语言模型（Causal Language Modeling）指的跟根据输入的部分文本序列来预测接下来的文本内容的一个任务。因此，在这种场景下，模型仅仅只会关注于被挡住部分左边的内容（右边全都被遮蔽），然后生成后续的内容。并且通常情况下，最后预测得到的结果都是通过取模型最后一个隐藏状态的向量作为logits，然后通过分类器分类得到。

下面是一个通过输入一段文本序列，然后对后续内容进行生成预测的示例模型。它首先通过checkpoint名称来实例化一个tokenizer和model类，然后再借助`top_k_top_p_filtering()`方法来完成文本的生成：

```python
# 针对于PyTorch
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2", return_dict=True)
sequence = f"Hugging Face is based in DUMBO, New York City, and"
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# get logits of last hidden state
next_token_logits = model(input_ids).logits[:, -1, :]
# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated = torch.cat([input_ids, next_token], dim=-1)
resulting_string = tokenizer.decode(generated.tolist()[0])
```



```python
# 针对于Tensorflow
from transformers import TFAutoModelWithLMHead, AutoTokenizer, tf_top_k_top_p_filtering
import tensorflow as tf
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelWithLMHead.from_pretrained("gpt2", return_dict=True)
sequence = f"Hugging Face is based in DUMBO, New York City, and "
input_ids = tokenizer.encode(sequence, return_tensors="tf")
# get logits of last hidden state
next_token_logits = model(input_ids)[0][:, -1, :]
# filter
filtered_next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
# sample
next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)
generated = tf.concat([input_ids, next_token], axis=1)
resulting_string = tokenizer.decode(generated.numpy().tolist()[0])
```

在运行完上述代码后，我们就能够得到根据输入序列所生成的下一个单词：

```python
print(resulting_string)
Hugging Face is based in DUMBO, New York City, and has
```

在这个示例中，最后得到的预测输出就是'has'（译者在实际运行时输出的是is）。不过到目前为止这看起来几乎没什么用，因为仅仅只是生成了一个单词。在接下来的内容中，我们将会介绍如何通过`generate()`来生成指定长度的预测序列。

## <span id ='1005'>5 文本生成模型</span>

所谓文本生成模型（Text Generation）就是指将一段文本输入到模型中，让后模型根据文本所处的语境生成后续的内容。在接下来的这个示例中，我们将使用GPT-2模型以`pipelines`的方式来生成文本。同时，对于所有的模型来说，在使用`pipelines`生成文本时默认都只会生成前K个单词（译者注：也就是下面的`max_length`参数，其具体信息可以参见相关的配置文件（例如[gpt-2 config](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json) ）。

### 5.1 `pipelines`实现

```python
from transformers import pipeline
text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

#
[{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
```

在上面的示例中，模型将根据输入的文本随机产生50个单词。同时，对于`PreTrainedModel.generate()`中的默认参数，我们也可以直接在调用`pipelines`的过程中进行修改，例如上面的`max_length`参数。

### 5.2 搭建实现

接下来，我们将使用`XLNet`模型来进行文本的生成：

```python
# 针对于 PyTorch
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
```



```python
# 针对于Tensorflow
from transformers import TFAutoModelWithLMHead, AutoTokenizer
model = TFAutoModelWithLMHead.from_pretrained("xlnet-base-cased", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="tf")
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
```

在运行完上述代码后，即可输出如下的生成结果：

```python
print(generated)
Today the weather is really nice and I am planning on anning on taking a nice...... of a great time!<eop>...............
```

对于文本生成模型来说，在*Transformers*中你可以通过PyTorch来使用 *GPT-2*, *OpenAi-GPT*, *CTRL*, *XLNet*, *Transfo-XL* and *Reformer*这些模型，当然其中的绝大多数也支持了Tensorflow。正如上面所示例的XLNet和Transfo-XL模型，通常他们都需要给定一些前情提示的PADDING_TEXT才能生成更合理的文本。通常，对于开放式的文本生成模型来说，GPT-2都是一个不错的选择，因为它是基于CLM在百万级的语料上训练得到的。

关于更多如何进行文本生成的策略，可以参见我们的这篇[博文](https://huggingface.co/blog/how-to-generate)。

## <span id ='1006'>6 命名体识别</span>

命名体识别（Named Entity Recognition）指的就是对序列片段进行分类的任务。例如识别一个序列片段代表的是人名、组织名还是地名。一个常见的用于命名体识别的数据集就是CoNLL-2003。如果你想微调一个NER任务，那么你可通过PyTorch来运行[`run_ner.py`](https://github.com/huggingface/transformers/tree/master/examples/token-classification/run_ner.py)，或者是通过PyTorch-lightning来运行 [run_pl_ner.py](https://github.com/huggingface/transformers/tree/master/examples/token-classification/run_pl_ner.py) ，或者是使用Tensorflow来运行[run_tf_ner.py](https://github.com/huggingface/transformers/tree/master/examples/token-classification/run_tf_ner.py) 。

下面是一个使用`pipelines`来进行命名体识别的任务，准确的来说是一个包含有9中类别的NER识别任务：

- O，其它
- B-MIS，一个杂项实体的开始，紧接着另一个杂项实体
- I-MIS，杂项实体
- B-PER，一个人名的开始紧跟着另外一个人名
- I-PER，人名
- B-ORG，一个组织名开始紧邻着另外一个组织名
- I-ORG，组织名
- B-LOC，一个地名的开始紧跟着另外一个地名
- I-LOC，地名

### 6.1 `pipelines`实现

```python
from transformers import pipeline

nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."
```

在上面示例的代码中，其加载的预训练模型是由来自[dbmdz](https://github.com/dbmdz)的 [@stefan-it](https://github.com/stefan-it)在数据集CoNLL-2003上微调后得到的。在执行完上述代码后，将会输出一个包含有所有被归类到相应实体类别的单词的列表。如下所示：

```python
[{'word': 'Hu', 'score': 0.999578595161438, 'entity': 'I-ORG', 'index': 1}, {'word': '##gging', 'score': 0.9909763932228088, 'entity': 'I-ORG', 'index': 2}, {'word': 'Face', 'score': 0.9982224702835083, 'entity': 'I-ORG', 'index': 3}, {'word': 'Inc', 'score': 0.9994880557060242, 'entity': 'I-ORG', 'index': 4}, {'word': 'New', 'score': 0.9994345307350159, 'entity': 'I-LOC', 'index': 11}, {'word': 'York', 'score': 0.9993196129798889, 'entity': 'I-LOC', 'index': 12}, {'word': 'City', 'score': 0.9993793964385986, 'entity': 'I-LOC', 'index': 13}, {'word': 'D', 'score': 0.9862582683563232, 'entity': 'I-LOC', 'index': 19}, {'word': '##UM', 'score': 0.9514269828796387, 'entity': 'I-LOC', 'index': 20}, {'word': '##BO', 'score': 0.9336590766906738, 'entity': 'I-LOC', 'index': 21}, {'word': 'Manhattan', 'score': 0.9761653542518616, 'entity': 'I-LOC', 'index': 28}, {'word': 'Bridge', 'score': 0.9914628863334656, 'entity': 'I-LOC', 'index': 29}]
```

从结果可以看到，“Hugging Face”被成功的标记成了组织，“New York City”、“DUMBO”和“Manhattan Bridge”都被标记成了地名。

### 6.2 搭建实现

下来，我们再来通过*Transformers*提供的接口来搭建一个命名体识别模型。通常其过程主要包括如下几个步骤：

1. 通过checkpoint名称来实例化一个tokenizer和model类。例如在下面的示例中我们将通过“dbmdz/bert-large-cased-finetuned-conll03-english”来载入模型，以及对应的权重；
2. 定义实体的标签列表，模型将会基于这些标签进行训练；
3. 尝试以你知道的实体来定义一个序列，例如”Hugging Face“是一个组织，”New York City“是一个地名；
4. 将序列中的单词划分成token，以便于在最后预测的时候将预测得到的结果映射为真实的标签值；
5. 得到序列中每个单词的token；
6. 通过将输入传入到模型中来返回得到输出，然后再通过一个softmax来进行分类处理；
7. 输出预测结果

```python
# 针对于PyTroch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
#   第一步
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#   第二步
label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]
#   第三步
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."
# Bit of a hack to get the tokens with the special tokens

#   第四步
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
#   第五步
inputs = tokenizer.encode(sequence, return_tensors="pt")
#   第六步
outputs = model(inputs).logits
#   第七步
predictions = torch.argmax(outputs, dim=2)
```

```python
# 针对Tesorflow
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf
model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."
# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="tf")
outputs = model(inputs)[0]
predictions = tf.argmax(outputs, axis=2)
```

在运行完上述代码后，我们将看到如下所示的输出结果：

```python
print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])

[('[CLS]', 'O'), ('Hu', 'I-ORG'), ('##gging', 'I-ORG'), ('Face', 'I-ORG'), ('Inc', 'I-ORG'), ('.', 'O'), ('is', 'O'), ('a', 'O'), ('company', 'O'), ('based', 'O'), ('in', 'O'), ('New', 'I-LOC'), ('York', 'I-LOC'), ('City', 'I-LOC'), ('.', 'O'), ('Its', 'O'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'), ('D', 'I-LOC'), ('##UM', 'I-LOC'), ('##BO', 'I-LOC'), (',', 'O'), ('therefore', 'O'), ('very', 'O'), ('##c', 'O'), ('##lose', 'O'), ('to', 'O'), ('the', 'O'), ('Manhattan', 'I-LOC'), ('Bridge', 'I-LOC'), ('.', 'O'), ('[SEP]', 'O')]
```

## <span id ='1007'>7 摘要生成</span>

摘要生成（summarization）指的是输入一个稳当或者一篇文章，然后返回一段简短的总结。

一个常见的用于摘要生成的数据集便是CNN或者是Daily Mail新闻数据集，它们是由很多的长文本和对应的摘要所构成。同时，*Transformers*中提供了多种方法来实现这一任务，当然也包括如何微调一个摘要生成的预训练模型。

### 7.1 `pipelines`实现

下面是通过`pipelines`来实现一个摘要生成模型在Daily Mail数据集上进行微调的示例。

```python
from transformers import pipeline
summarizer = pipeline("summarization")
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
```

因为用于摘要生成的`pipelines`是基于 `PreTrainedModel.generate()`方法的，故这里我们可以指定最后预测输出的最大长度和最小长度等。例如：

```python
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

[{'summary_text': ' Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002 . At one time, she was married to eight men at once, prosecutors say .'}]
```

### 7.2 搭建实现

下面是一个通过*Transformers*提供的API搭建一个摘要生成的示例，其过程主要包含如下四个步骤：

1. 根据checkpoint名称实例化一个tokenizer和模型。通常来说，摘要生成模型使用的都是Encoder-Decoder的网络结构，例如`Bart`或者是`T5`模型。
2. 定义需要生成摘要的原始文本素材。
3. 通过传入指定的前缀"summarize: "来调用T5模型。
4. 使用`PreTrainedModel.generate()`来生成摘要。

在这个示例中，我们将会使用到谷歌所发布的T5模型。尽管T5模型是在一个混合的多任务数据集（包括CNN和Daily Mail）上预训练的，但却取得了不错的结果。

```python
# 针对于 PyTorch
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
```

```python
# 针对于Tensorflow
from transformers import TFAutoModelWithLMHead, AutoTokenizer
model = TFAutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="tf", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
```

在运行完上述代码后，我们便能得到如下所示的结果：

```python
generated = tokenizer.decode(outputs[0])
print(generated)

prosecutors say the marriages were part of an immigration scam. if convicted, barrientos faces two criminal counts of "offering a false instrument for filing in the first degree" she has been married 10 times, nine of them between 1999 and 2002.
```

## <span id ='1008'>8 翻译</span>

翻译（translation）指的是将一种语言的文本翻译成另外一种语言的过程。一个常见的用于翻译的数据集便是WMT English to German，它将英语句子作为输入然后让模型翻译出对应的德语句子。如果你想要在翻译任务上微调某个模型，可以参见这个[文档](https://github.com/huggingface/transformers/blob/master/examples/seq2seq/README.md)。

### 8.1 `pipelines`实现

下面是一个使用`pipelines`来进行文本翻译的示例，并且其中所使用到的是一个在多任务混合训练集上训练得到的T5模型，最后也取得了不错的结果。

```python
from transformers import pipeline
translator = pipeline("translation_en_to_de")
print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))

[{'translation_text': 'Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.'}]
```

因为上述`pipeline()`实质上所依赖的是 `PreTrainedModel.generate()`方法，因此我们也可以直接通过`pipeline()`传入参数来修改 `PreTrainedModel.generate()`中所对应的参数，例如上面的`max_length=40`。

### 8.2 搭建实现

下面是通过API来搭建一个翻译模型的实例，其通常需要以下四个步骤：

1. 根据checkpoint名称实例化一个tokenizer和模型。通常来说，翻译模型使用的都是Encoder-Decoder的网络结构，例如`Bart`或者是`T5`模型。
2. 定义一段需要翻译的文本。
3. 通过传入指定的前缀"translate English to German: "来调用T5模型。
4. 使用`PreTrainedModel.generate()`来生成翻译。

```python
# 针对于PyTorch
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
```

```python
# 针对于Tensorflow
from transformers import TFAutoModelWithLMHead, AutoTokenizer
model = TFAutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="tf")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
```

在运行完上述代码后，我们就能得到与`pipelines()`同样的翻译结果：

```python
generated = tokenizer.decode(outputs[0])
print(generated)

# Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.
```



## [<主页>](README.md)  