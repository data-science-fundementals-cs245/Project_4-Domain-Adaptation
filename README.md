# Project_4---Domain-Adaptation
Domain Adaptation for Image Classification

| DA | A->R | C->R | P->R |
|-|-|-|-|
| TCA | 69.02 | 61.42 | 67.51 |
| CORAL | 74.87 | 66.49 | 72.69 |


### Hongbin Chen's Results
|            | A->R              | C->R               | P->R               |
| ---------- | ----------------- | ------------------ | ------------------ |
| Without DA | 75.48 {c=5.14666} | 66.46{c=2.416179}  | 72.96{c=5.14666}   |
| KMM        | 75.53 {c=5.14666} | 66.55{c=2.416179}  | 74.36{c=0.5325205} |
| SA_512     | 75.25{2.416179}   | 65.08{c=10.962808} | 72.64{c=5.14666}   |
| SA_1024    | 75.60{c=2.416179} | 66.00{c=5.14666}   | 72.91{c=5.14666}   |
| SA_2048    | 75.48 {c=5.14666} | 66.46{c=2.416179}  | 72.96{c=5.14666}   |

### Dongyue‘s Results

Clone from Newly's Repository: <https://github.com/ustcnewly/domain_adaptation>

1. **DASVM**: Domain adaptation problems: A DASVM classification technique and a circular validation strategy
2. **DIP**: Unsupervised domain adaptation by domain invariant projection
3. **GFK**: Geodesic flow kernel for unsupervised domain adaptation **(deprecated)**
4. **KMM**: Correcting sample selection bias by unlabeled data **(deprecated)**
5. **SA**: Unsupervised visual domain adaptation using subspace alignment **(deprecated)**
6. **SGF**: Domain adaptation for object recognition: An unsupervised approach
7. **STM**: Selective transfer machine for personalized facial action unit detection
8. **TCA**: Domain adaptation via transfer component analysis **(deprecated)**
9. **RDALR**: Robust visual domain adaptation with low-rank reconstruction

|  DA   | A->R  | C->R  | P->R  |
| :---: | :---: | :---: | :---: |
| DASVM |       |       |       |
|  DIP  | 74.10 | 62.07 | 69.81 |
|  SGF  | 72.56 | 61.89 | 69.14 |
|  STM  |       |       |       |
| RDALR |       |       |       |

- DASVM 不能用，因为牛力的demo里面是针对二分类的代码

- RDALR 不能用，因为牛力的demo里面写得太不清楚，不知如何调用

- STM 不知是跑得太慢还是不能直接使用，正在处理中
