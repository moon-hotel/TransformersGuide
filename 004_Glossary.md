# 术语表

## <span id ='041'>1 术语总览</span>

- autoencoding models： 参见 MLM
- autoregressive models:：参见 CLM
- CLM： 因果语言建模（Causal Language Modeling），它是与训练任务的一种，即模型按序读入文本，然后对下一个词进行预测。
- MLM：掩体语言模型（Masked Language Modeling）,与CLM一样也是一种预训练模型。通常情况下，输入模型的都是一句随机某些位置被掩盖了的句子，而模型的任务就是对掩盖部分的内容进行预测。
- multimodel：多模态，即将多种形式的数据（如文本和图像）结合起来输入到模型中。
- NLG：自然语言生成（Natural Language Generation）,即用来生成序列的语言模型，例如对话或者翻译。
- NLP： 自然语言处理（Ntura Language Processing）。
- NLU：自然语言理解（Natural Language Understanding）。
- Pretrained model：预训练模型，即在一些大的语料中（如维基百科）以某个网络结构（如BERT）预先训得到的权重参数。
- RNN：循环神经网络。
- Seq2Seq：序列到序列（Sequence-to-Sequence）模型。
- token：一句话的一部分，通常情况下是一个词（word），但也可以是一个字符或一个字。

## <span id ='042'>2 模型输入</span>

尽管每种模型都不同，但彼此之间具有相似之处。因此，很多模型都可以使用同样形式的输入，具体细节见后续示例。

### 2.1 输入IDs

通常情况下输入ids仅仅只会作为模型一开始的输入。这些ids是输入的文本根据词表中每个token的位置所得到的索引序列，用来对原始文本序列进行表示，然后作为模型的输入。

虽然词表中的每个词都不同，但是其潜在的机制是保持不变的。下面，我们以BERT中的tokernizer（ [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) ）进行说明：

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24GB of VRAM"
```

首先，tokenizer会根据词表中的词将原始的文本序列进行分割：

```python
tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

从上面输出的结果可以看到，分割后序列中的每一个元素即可以是一个词，也可以只是其中的一部分。例如”VRAM“并不存在于词表当中，所以其被划分成了"V"，"RA"和"M"三个部分。同时，为了说明这些token不是可分词而是来自同一个词的不同部分，我们在每个token前面都加上了两个"#"。

这些分割得到的token接下来就会被转换成索引（ids），然后再被未入到模型中。为了提升处理效率，在*Transformers*中你可以直接通过`tokenizer()`来直接将原始的文本序列转化为ids索引。

```python
inputs = tokenizer(sequence)
```

接着，`tokenizer()`就会返回一个字典，里面包含了对应模型所需要的所有输入而不仅仅只是模型输入的索引序列。

```python
inputs = tokenizer(sequence)
encoded_sequence = inputs["input_ids"]
print(inputs)

#
{'input_ids': [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

其中`input_ids`对应的部分就是输入的序列ids。

需要注意的是，`tokenizer()`会自动根据对应模型的需要而加上模型“特殊字符”，如果模型需要的话。

译者注：例如在BERT中，需要在最前面和一句话的结尾处分别加上'符号[CLS]'和'[SEP]'，也就是对应上面`input_ids`中的101和102索引。

如果你需要将索引序列还原成原始的文本，使用`tokenizer`中的`decode()`方法即可：

```python
decoded_sequence = tokenizer.decode(encoded_sequence)
```

接着我们就能看到如下的结果：

```python
print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

之所以会得到这样的结果是因为我们一开始将输入的文本转化为ids时，使用的是`BertTokenizer`中`tokenizer()`方法，所以最终得到的结果就是BERT模型所需要的输入形式。

### 2.2 注意力掩码

注意力掩码为一个可选择的参数，它主要是用来告诉模型哪些位置的token是应该被考虑的，而哪些位置的token是应该被忽略的。

译者注：例如在最后计算模型的损失时，padding部分的token就应该被忽略掉。

例如，考虑如下两个文本序列：

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."
encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

此时我们将token后的的结果输出来就会发现两个序列的长度并不一致：

```python
len(encoded_sequence_a), len(encoded_sequence_b)
# (8,19)
```

因此，我们并不能直接将这两个序列同时一起输入到模型中。所以，我们要做的要么是将第一个序列填充成和第二个学列一样长；要么是将第二个序列截断成和第一个序列一样长。

在第一种情况中，token后的得到的ids将会填充上对应的填充标志（一般会是`0`）。我们可以给`tokenizer()`传入一个包含多个序列的列表来实现这一目的：

```python
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
padded_sequences["input_ids"]
#
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

从上面的运行结果可以看到，`0`被填充到了第一个序列的右边，来使得其保持与第二个序列相同的长度。

因此，注意力掩码其实就是一个仅包含`0`和`1`的张量，用以标识哪些位置的token是填充得到的。这一模型在后续的工作中就会忽略掉这些位置上的token。对于[`BertTokenizer`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertTokenizer)来说，`1`表示该位置上的token应该被模型所考虑，而`0`表示该位置上的token是填充得到的：

```python
padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### 2.3 Token Type IDs

存在一些被用于做文本分类或者是问题回答的网络模型，因此这就需要同时将两句话输入到模型中（译者注：这里的文本分类并不是指的对一句话进行类别划分）。在这种情况下，为了对两句话进行区分，通常我们会在第一句话的开始和结尾分别插入一个特殊的标记符`[CLS]`和`[SEP]`，前者用来表示分类标记，后者用来表示分割标记。例如在BERT模型中，将会以下面的方式来进行处理：

```python
# [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

此时，我们就可以使用`tokenizer()`来自动的进行处理，注意此时是将两个序列分别作为两个参数传入，而不是向上面那样作为一个列表进行传入：

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"
encoded_dict = tokenizer(sequence_a, sequence_b)#注意此处分别是两个参数
decoded = tokenizer.decode(encoded_dict["input_ids"])
print(decoded)
```

在执行完上述代码后就会返回如下结果：

```
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

经过这样的处理后，就足以让大多数模型知道从哪儿开始是一句话的结束，哪儿是一句话的开始。然而，对于BERT模型来说，还需要一种称为Token Type IDs的东西（或者叫段标记segment IDs）。它同样也是通过一个只含`0`和`1`的掩码张量来对两句话进行区分。

```python
print(encoded_dict['token_type_ids'])
#
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

其中，第一个被用于回答问题的上下文句子用`0`来进行标识，而第二个句子（问题）则用`1`来进行标识。

还有一些模型，例如 [`XLNetModel`](https://huggingface.co/transformers/master/model_doc/xlnet.html#transformers.XLNetModel) 还会使用到`2`对其它情况进行标识。

### 2.4 位置 IDs

在RNNs中，由于输入模型的token都是按序进行的，所以并不需要位置IDs这个东西。而在包含有transformers的结构中，由于transformers本身并不能够分辨每个token的位置，因此我们需要一个额外的`position_ids`来告知transformers每个token所处的位置。

当然，这也只是一个可选择的参数。如果模型没有被传入这个参数，那么模型将自动创建一个绝对的位置嵌入信息来进行标识。所谓的绝对位置指的就是以原始输入的token的顺序以`[0, config.max_position_embeddings - 1]`的范围进行标识。同时，还有一些其它模型用的是正弦位置编码和相对位置编码。

### 2.5 标签

标签是一个可选择的参数，它的目的主要是传入模型后用于计算模型的损失值。同时，对于不同的模型来说，其希望接收到的标签的形式并不相同：

- 对于序列分类模型来说（如[`BertForSequenceClassification`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertForSequenceClassification)），它希望接收的标签的形状为`(batch_size,)`，即我们一般模型需要的标签形式。
- 对于字符分类模型来说（如[`BertForTokenClassification`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertForTokenClassification)），它希望接收的标签的形状为 `(batch_size, seq_length)` ,即对于每句话中的每个token，都对应有一个相应的正确标签值。
- 对于掩体语言模型来说（如 [`BertForMaskedLM`](https://huggingface.co/transformers/master/model_doc/bert.html#transformers.BertForMaskedLM)），它希望接收到的标签的形状为`(batch_size, seq_length)`，即对于每句话中被掩盖的token都有一个正确的标签值，同时忽略掉剩下的其它token。
- 对于序列到序列模型来说（如[`BartForConditionalGeneration`](https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration)），它希望接收到的标签的形状为`(batch_size, tgt_seq_length)`，即对于生成的每一句话中（例如翻译），每个token都对应有一个正确的标签值。

对于更为具体的每个模型所需要的标签信息，可以参见对应部分的说明文档。同时，对于一些基本的模型（如[`BertModel`](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel)）并不需要标签，因为它们都是基于transformer的，仅仅只是用来将输入编码成特征输出即可。

译者注：这里指的应该是将这些模型用于预测时不需要标签。

### 2.6 解码输入IDs

这类输入特定指的是encoder-decoder模型，此时的输入将会作为模型解码时的正确标签。例如在翻译模型中，在训练模型时需要将正确的（标签）翻译文本也输入到模型的解码器中，同解码器的预测值计算模型损失。

所幸的是，在大多数的encoder-decoder模型中（如BART和T5），只需要输入正确的文本序列，模型自生就能够将其转换为decoder所需要的token ID形式。因此在这类模型中，更推荐的就是使用这种方法。所有，当使用到一些序列到序列的模型时，一定要仔细查看相应的说明文档，以便于确定模型需要怎样的输入形式。

### 2.7 前向传播部分

在transformers的每个残差注意力模块中，自注意力层的后面紧接着通常都会是两个全连接网络。在全连接层中将的token 嵌入维度通常来说都会设置成大于模型隐藏层的维度。

对于输入形状为`[batch_size,sequence_length]`来说，计算机都需要大量的内存来存储一个形状为`batch_size,sequence_length,config.intermediate_site]`的嵌入表示。[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)的作者表示，由于模型再计算这一输入嵌入表示时是独立于`sequence_length`这个维度的，因此从数学的角度来说其等价于独立的计算`[batch_size,config.hidden_size]_0`,...,`[batch_size,config.hidden_size]_n`，然后再将这些组织成`[batch_size,sequence_length,config.hidden_size]`（此时的`n=sequence_length`）的结果。但是这样做的好处在于，后者以牺牲时间为代价而降低的对于内存的开销。

译者注：这段话理解的不是特别清楚，可能与原文存在一些出入。

对于使用 [`apply_chunking_to_forward()`](https://huggingface.co/transformers/internal/modeling_utils.html#transformers.apply_chunking_to_forward)的模型来说， `chunk_size`定义了输出嵌入表示的维度。

For models employing the function [`apply_chunking_to_forward()`](https://huggingface.co/transformers/internal/modeling_utils.html#transformers.apply_chunking_to_forward), the `chunk_size` defines the number of output embeddings that are computed in parallel and thus defines the trade-off between memory and time complexity. If `chunk_size` is set to 0, no feed forward chunking is done.



## [<主页>](README.md)  