本项目旨在开发一个敏感话题识别系统，采用两种方法对话题进行分类：敏感词判别和基于预训练语言模型的判别。
分类类别包括：正常、政治、违法、色情、暴恐、广告。
构建了包含77,370条敏感文本和22,823个敏感词的高质量数据集。
       

## 🚀 Quick Start
```
#安装软件包
pip install -r requirements.txt

# 划分数据集为训练集、验证集和测试集
python Spilt_CSV.py

# 训练bert、bert_CNN，把model替换为对应名称
python Train1.py --model bert

# 训练TextCNN、TextRCNN、DPCNN,把model替换为对应名称
python Train2.py --model TextCNN
# 训练FastText

python Train2.py --model FastText --embedding random
# 进行推理
python Predict.py 
```
