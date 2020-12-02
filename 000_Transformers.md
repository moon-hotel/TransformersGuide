<p align="center">
    <br>
    <img src="./Images/transformers_logo_name.png" width="400"/>
    <br>
<p>
<h3 align="center">
<p>State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
</h3>

# *Transformers*

*Transformers*（之前叫做*pytorch-transformers*和*pytorch-pretrained-ber*）提供了一些自然语言理解（NLU）和自然语言生成（NLG）中常见的网络模型，包括BERT、GPT-2、RoBERTa、XLM、DistilBert、XLNet等等。同时，*Transformers*也提供了在100多种语言中所训练得到的超过32个预训练模型，并且这些模型能够在TensorFlow2.0和Pytorch间做到无缝切换。

点击此处[*Transformers*](https://github.com/huggingface/transformers)可以查看我们的代码仓库！

# 特性

- 高性能：使用*Transformers*能够轻松地在NLU和NLG任务上取得很好的效果；
- 低门槛：使用*Transformers*能够轻松地实现各种NLU和NLG模型；

对于所有人来说，*Transformers*都是你的最有选择：

- 深度学习研究者
- AI项目落地者
- AI/ML/NLP教育工作者

减少计算代价，降低碳排放：

- 研究者们能够轻松的分享自己训练好的模型，而不所有人不停的对同一个模型进行训练；
- 相关从业者能够减少计算时间于部署成本；
- 8种网络架构，超过30+预训练模型，多大上百种语言

在适当的时候选择适当的模型：

- 三行代码就可以训练处一个state-of-the-art网络模型；
- TensorFlow2.0与Pytorch预训练模型间的无缝切换；
- 随意在TensorFlow2.0和Pytorch切换网络架构；
- 无缝的选择合适网络架构进行训练、评估和部署；

# 目录

整个*Transformers*说明文档可以大致分为如下五个部分：

- **简单上手：** 这部分包含了对于*Transformers*的一个快速指南，包括如何安装如何简单使用以及我们的一些理念；
- **总体介绍：** 这部分包含了对于*Transformers*的总体介绍和使用信息；
- **进阶指导：** 这部分包含了对于*Transformers*更为进阶的使用，包括指定自己的配置参数等；
- **研究使用：** 这部分主要包含了一些对应Transformers模型的通用研究介绍；

目前*Transformers*包含了基于Pytorch和Tensorflow两种框架的实现，以及对应的预训练模型、使用说明和两者间的转换说明等。对于*Transformers*已经实现的模型如下所示：

1. [ALBERT](https://huggingface.co/transformers/master/model_doc/albert.html) (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
2. [BART](https://huggingface.co/transformers/master/model_doc/bart.html) (from Facebook) released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
3. [BERT](https://huggingface.co/transformers/master/model_doc/bert.html) (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
4. [BERT For Sequence Generation](https://huggingface.co/transformers/master/model_doc/bertgeneration.html) (from Google) released with the paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
5. [Blenderbot](https://huggingface.co/transformers/master/model_doc/blenderbot.html) (from Facebook) released with the paper [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
6. [CamemBERT](https://huggingface.co/transformers/master/model_doc/camembert.html) (from Inria/Facebook/Sorbonne) released with the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
7. [CTRL](https://huggingface.co/transformers/master/model_doc/ctrl.html) (from Salesforce) released with the paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
8. [DeBERTa](https://huggingface.co/transformers/master/model_doc/deberta.html) (from Microsoft Research) released with the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
9. [DialoGPT](https://huggingface.co/transformers/master/model_doc/dialogpt.html) (from Microsoft Research) released with the paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
10. [DistilBERT](https://huggingface.co/transformers/master/model_doc/distilbert.html) (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
11. [DPR](https://huggingface.co/transformers/master/model_doc/dpr.html) (from Facebook) released with the paper [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
12. [ELECTRA](https://huggingface.co/transformers/master/model_doc/electra.html) (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
13. [FlauBERT](https://huggingface.co/transformers/master/model_doc/flaubert.html) (from CNRS) released with the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
14. [Funnel Transformer](https://huggingface.co/transformers/master/model_doc/funnel.html) (from CMU/Google Brain) released with the paper [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
15. [GPT](https://huggingface.co/transformers/master/model_doc/gpt.html) (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
16. [GPT-2](https://huggingface.co/transformers/master/model_doc/gpt2.html) (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
17. [LayoutLM](https://huggingface.co/transformers/master/model_doc/layoutlm.html) (from Microsoft Research Asia) released with the paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
18. [Longformer](https://huggingface.co/transformers/master/model_doc/longformer.html) (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
19. [LXMERT](https://huggingface.co/transformers/master/model_doc/lxmert.html) (from UNC Chapel Hill) released with the paper [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal.
20. [MarianMT](https://huggingface.co/transformers/master/model_doc/marian.html) Machine translation models trained using [OPUS](http://opus.nlpl.eu/) data by Jörg Tiedemann. The [Marian Framework](https://marian-nmt.github.io/) is being developed by the Microsoft Translator Team.
21. [MBart](https://huggingface.co/transformers/master/model_doc/mbart.html) (from Facebook) released with the paper [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
22. [Pegasus](https://huggingface.co/transformers/master/model_doc/pegasus.html) (from Google) released with the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)> by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu.
23. [ProphetNet](https://huggingface.co/transformers/master/model_doc/prophetnet.html) (from Microsoft Research) released with the paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
24. [Reformer](https://huggingface.co/transformers/master/model_doc/reformer.html) (from Google Research) released with the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
25. [RoBERTa](https://huggingface.co/transformers/master/model_doc/roberta.html) (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. ultilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
26. [SqueezeBert](https://huggingface.co/transformers/master/model_doc/squeezebert.html) released with the paper [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer.
27. [T5](https://huggingface.co/transformers/master/model_doc/t5.html) (from Google AI) released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
28. [Transformer-XL](https://huggingface.co/transformers/master/model_doc/transformerxl.html) (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
29. [XLM](https://huggingface.co/transformers/master/model_doc/xlm.html) (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
30. [XLM-ProphetNet](https://huggingface.co/transformers/master/model_doc/xlmprophetnet.html) (from Microsoft Research) released with the paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
31. [XLM-RoBERTa](https://huggingface.co/transformers/master/model_doc/xlmroberta.html) (from Facebook AI), released together with the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
32. [XLNet](https://huggingface.co/transformers/master/model_doc/xlnet.html) (from Google/CMU) released with the paper [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
33. [Other community models](https://huggingface.co/models), contributed by the [community](https://huggingface.co/users).



## [<主页>](README.md)  