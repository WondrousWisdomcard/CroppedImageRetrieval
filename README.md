# 残缺图像检索 SDK

[toc] 

## 方法一：模板匹配


### 一、算法说明

* **灰度模板匹配算法**：将裁剪图视作模板，遍历图像库中的每一张图片，对模板和图片应用模板匹配算法，算法输出模板在图片中的匹配的最高相似度，作为模板（裁剪图）来自该图片的可信度，最后选出可信度最高的 5 张图图片作为检索结果。
* **优化策略**：
  * **压缩图像**，在图像比较大的时候可以使用压缩来提高速度。
  * **多角度模板匹配**：设定多个模板旋转角度和镜像模板，一旦一轮检测结束时检测到高于 thredhold 的可信度就不在进行模板的下一次旋转。
  * **多尺寸模板匹配**：设置多个缩放尺度，一旦一轮检测结束时检测到高于 thredhold 的可信度就不在进入下一个尺度。

### 二、函数说明

**`CroppedImageRetrieval.template_retrieval(template_img, retrieval_dir, compress_rate=4, threshold=0.8, show=True, rotate=True, zoom=True)`**

* 参数：
  * `template_img` 裁剪图片的文件路径
  * `retrieval_dir` 检索图片库的文件夹路径
  * `compress_rate` 图片长宽的压缩倍率，默认为 4
  * `threshold` 模板变换的可信度阈值，默认为 0.8
  * `show` 显示匹配图片，默认为 True
  * `rotate` 是否允许模板旋转，默认为 True
  * `zoom` 是否允许模板缩放，默认为 True

* 输出：
  * `best_match_info` 返回可信度最高的五张原图的列表，每个元素是（文件名，可信度）二元组，按可信度从大到小排序

* **提示：如果裁剪图没有没有较原图进行缩放，可以将 zoom 设置为 False；如果裁剪图没有较原图进行旋转，可以将 rotate 设为 False 以缩短检索时间。**

## 方法二：分类神经网络

### 一、思路：分类神经网络

#### 1.1 算法步骤：

* **数据处理阶段：**

  * 将图像按照一定的概率分布随机裁剪为 128 张图像，作为训练样本。

    |       中心点选择        |        长宽选择         |          旋转角度           |
    | :---------------------: | :---------------------: | :-------------------------: |
    | ![](Image/高斯曲线.png) | ![](Image/泊松曲线.png) | ![](Image/混合高斯曲线.png) |

    其中，中心点选择和长宽选择分别遵循高斯分布和泊松分布，横坐标为图片的长或宽，旋转角度遵循混合高斯分布，横坐标从为旋转的角度（$[-180,180]$）

  * 视每一张原图为一个分类，将随机裁剪的 128 张图片（我们将每个类的前4/5的裁剪图当作训练集，剩下1/5作为验证集），将图像统一压缩成到 224x224，再输入输出神经网络（ResNet18）中进行训练。

    ![preview](Image/v2-3cae70aa38fd66f7368d096a1e5d980f_r.jpg)

* **图像检索阶段：**

  * 将待检索图片直接输入网络，跟据输出的分类结果从图片库中找到对应的原图。

#### 1.2 算法评价

* 优点：
  * 速度快，准确率高；
  * 鲁棒性更强，在处理经过缩放和旋转的裁剪图上效果更优于模板匹配；
  * 模型可以不断训练，提高正确率

* 缺点：
  * 可扩展性差，当新图片加入时需要从重新训练神经网络；
  * 对于如 发票图片库 这样的图像高度相似，只有小许细节（如文字）不同的图片库，模型效果较差，需要进行调整（增大输入图片的长宽，修改随机裁剪的概率分布）



### 二、运行说明

#### 第一步：对图片库的每一张图片生成随机裁剪图

**`CroppedImageRetrieval.images_cropping(data_path, crop_path, info_path, num=128)`**

* `data_path` **图片库所在的文件夹路径**，里面图片的数目将是分类神经网络输出的类别数

* `crop_path` **保存随机裁剪结果的文件夹路径**，该文件夹需要事先创建，随机裁剪结果将跟据各自对应的原图保存在子文件夹中

* `info_path` **保存 类别—原图映射表的文件夹路径**，该文件夹需要事先创建

* `num` **随机裁剪的数目**，默认为 128

  例如，原图保存在 `/home/xxx/Data/Tag/` 中，创建文件夹 `/home/xxx/Crop/Tag/` 以保存随机裁剪结果，创建文件夹 `/home/xxx/Info/Tag/` 以保存类别—图片映射表`info.csv`，该文件在测试时需要使用。
  
  ```python
  import CroppedImageRetrieval
  data_path = "/home/xxx/Data/Tag/"
  crop_path = "/home/xxx/Crop/Tag/"
  info_path = "/home/xxx/Info/Tag/"
  CroppedImageRetrieval.images_cropping(data_path, crop_path, info_path)
  ```

#### 第二步：训练模型

> 建议在 GPU 环境下训练网络，运行时间随网络的规模增大而延长。

**`CroppedImageRetrieval.model_train(data_path, model_path, class_num, epoch=5)`**

* `data_path` **随机裁剪结果的文件夹路径**，随机裁剪图由上一步生成，作为神经网络模型的输入
* `model_path` **保存训练模型的文件路径**，所在的文件夹需要事先创建，文件后缀 `pth` ，保存的模型用来实现图像检索，也可以继续训练更新保存的模型
* `class_num` **图片库图片数目**，即分类神经网络输出的类别数
* `epoch` **训练的迭代次数**，默认为 5，可以增加 Epoch 来提高模型准确率

例如， `/home/xxx/Crop/Tag/` 保存随机裁剪结果，图片数目为 172，使用 ResNet18 训练 3 个 Epoch，将模型保存至 `/home/xxx/Model/Tag/resnet.pth`

```python
import CroppedImageRetrieval
crop_path = "/home/xxx/Crop/Tag/"
model_path= "/home/xxx/Model/Tag/resnet.pth"
CroppedImageRetrieval.model_train(crop_path, model_path, 172, 3)
```

#### 第三步：残缺图像检索

**`CroppedImageRetrieval.model_test(model_path, info_path, image_path, class_num)`**

* `model_path` **保存训练模型的文件路径**，所在的文件夹需要事先创建，文件后缀 `pth` ，由上一步骤生成，保存的模型用来实现图像检索，也可以继续训练更新保存的模型
* `info_path` **类别—原图映射表的文件路径**，该文件由生成裁剪图时创建。
* `image_path` **裁剪图的文件路径**，程序将在图片库中找到该裁剪图对应的原图。
* `class_num` **图片库图片数目**，即分类神经网络输出的类别数
* 函数返回返回最有可能是原图的五张结果，以列表的形式组织

例如， `/home/xxx/Model/Tag/` 保存训练的模型，图片数目为 172，映射表的文件路径是 ` /home/xxx/Info/Tag/info.csv`，我们将寻找 `./Data/TagCut/000010.jpg` 对应的原图。

```python
import CroppedImageRetrieval
model_path = "/home/xxx/Model/Tag/resnet.pth"
info_path = "/home/xxx/Info/Tag/info.csv"
image_path = "./Data/TagCut/000010.jpg"
CroppedImageRetrieval.model_test(model_path, info_path, image_path, 172)
```

## 测试结果

### 一、模板匹配算法的准确率

|   数据   | 图片数目 | 准确率 | 准确率（Top5） | 平均检索时间（秒） |
| :------: | :------: | :----: | :------------: | :----------------: |
| **标签** |   172    |  100%  |      100%      |        8.83        |
| **发票** |   114    | 96.5%  |      100%      |        32.4        |
| **无线** |   178    |  100%  |      100%      |        11.3        |
| **隐患** |   190    | 96.3%  |     97.4%      |       13.92        |

### 二、分类网络算法的准确率

> 使用的网络架构：**ResNet18**
>
> * 该模型可以继续训练，甚至可以重新生成新的一批随即裁剪进行训练。
> * 同时，在更大的图片库上进行检索时，可以考虑使用更大的深度学习模型进行训练，如 ResNet-50，ResNet-101 或其他流行的神经网络架构，同时，随机裁剪的图片数目也可以从 128 增加到 512，甚至更大。

**发票图片库的问题：**经过测试，我们发现原先的策略不适用于发票图片库，因为它们图片较大，差异小（仅有文字差异），轮廓基本一致；224x224 的图片压缩丢失重要信息；同时，我们跟据发票裁剪数据的分布，适当调整了裁剪概率分布。

**解决策略：**（还在探究）


**训练结果**：

> 对于以下四个数据集，由于数据集中存在不少几乎相同的图片，我们借助模板匹配算法检测结果，提前手动删除了重复的图片。

|   数据   | 图片（类别）数 | 训练图片数 | 训练集准确率 | 验证图片数 | 验证集准确率 |
| :------: | :------------: | :--------: | :----------: | :--------: | :----------: |
| **标签** |      172       |   17200    |  **99.57%**  |    4816    |  **98.86%**  |
| **发票** |      114       |   11400    |  **42.74%**  |    3192    |  **41.23%**  |
| **无线** |      178       |   17800    |  **97.47%**  |    4984    |  **96.19%**  |
| **隐患** |      190       |   19000    |  **99.02%**  |    5320    |  **98.65%**  |

**测试结果**：

|   数据   | 测试图片数 | 准确率（Top1） | 准确率（Top5） | CPU 平均检索时间 | GPU 平均检索时间 |
| :------: | :--------: | :------------: | :------------: | :--------------: | :--------------: |
| **标签** |    172     |   **95.35%**   |   **98.84%**   |     0.15 秒      |     0.08 秒      |
| **发票** |    114     |   **14.04%**   |   **41.23%**   |     0.25 秒      |     0.15 秒      |
| **无线** |    178     |   **96.07%**   |   **98.88%**   |     0.16 秒      |     0.12 秒      |
| **隐患** |    190     |   **96.32%**   |   **99.47%**   |     0.18 秒      |     0.09 秒      |

### BugFix 日志

* 2022.7.1 修改了模板匹配算法对模板进行缩放时的错误，并增加了两个缩放尺度：[0.5,2] -> [0.5, 0.75, 1.5, 2]。
