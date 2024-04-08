# A simple CUDA K-NN Implementation

## Intro
+ Based on the [simple-knn](https://github.com/YixunLiang/simple-knn) repo
+ Support to query TOP-K (default 8) neighbours
+ Return their indices and square distance.

## Usage
+ git clone https://github.com/NK-CS-ZZL/custom-knn.git
+ cd custom-knn
+ pip install .

```
from custom_knn._C import topKdistCUDA2
dist2d, idx = topKdistCUDA2(torch.from_numpy(np.asarray(points)).float().cuda())
```