# 数字货币短期价格预测

数字货币交易所普遍对于Maker和Taker角色有不同的费率设置，以 [Huobi Futures](https://www.hbdm.com) 为例，一个典型费率结构如下：

| Maker | Taker |
|:-----:|:------:|
|-0.01% |0.037%|

这会为市场微观结构带来显著的影响，例如在某段时间内火币币本位永续合约 (Coin-margined swap) 2s 内收益 (单位 0.1%) 真值的分布如下：

![](https://jayhub.blob.core.chinacloudapi.cn/imgbed/documents/task-2104/Y_histplot.png)

以下提供了一个数据集，为上述合约在`Mar. 11 ~ Mar. 31`区间内的某些特征（`x`, 仅基于切片的深度行情提取，使用合约自身和相关合约，如 USDT-margined swap, Coin-margined futures）、真值 (`y`) 以及对应的时间戳 (`t`)。数据的格式以及读取方式如下：

```python
import h5py
import numpy as np
from datetime import datetime

with h5py.File(path_of_file, 'r') as hf:
    x = np.array(hf['x'])                # Shape = [n_samples, 1, n_features]
    y = np.array(hf['y'])                # Shape = [n_samples, 1]
    t = np.array(hf['timestamp'])        # Shape = [n_samples,]

# t 为 epoch 表示 (从 0001-01-01 开始的 ns 数, int64)，如有需要可以做如下转换到 datetime ：

_EPOCH = 621355968000000000

def to_datetime(t: np.int64) -> datetime:
    return datetime.fromtimestamp((t - EPOCH) / 1e7, tz=timezone.utc)

```

**问题**

请设计合适的神经网络（包括网络结构、损失函数和评价指标等）以及必要的数据处理流程（如有）对 `y` 进行预测，希望：

1. 合理地反映上述数据特征（或者其他方面地特征）
2. 不关注特征工程及预测效果
3. 使用 low-level api 更佳

数据存放在 Azure Blob Storage，可以用 [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) 或各种语言的 sdk 查看和下载。

连接字符串：

    DefaultEndpointsProtocol=https;AccountName=jaytmp;AccountKey=aAAu+WRh8RNFnCzmKRniPtsjFO394C9GtdGTrv/ZbrPugP3FWTWe7SEIlQZEld8rxEQ2CsmAE2FaomFCxlRZ+w==;EndpointSuffix=core.windows.net
