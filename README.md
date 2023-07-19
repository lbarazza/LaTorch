# LaTorch
LaTorch is a simple-to-use library to convert LaTeX-like tensor expressions into PyTorch models. It was developed with numerical relativity in mind and, as such, also handles automatic index highering and lowering.

## How to Use
Let us start with a simple example: the product between two matrices $3 \times 3$ matrices

$$ A^i_{ㅤj}ㅤB^j_{ㅤi} $$

In $\latex$ this would be written as

```latex
A^{i}_{j} B^{j}_{i} $$
```

We can use LaTorch to translate this expression into a PyTorch model with the following code

```python
import latorch as lt

latex = u'#^{i}_{k} = A^{i}_{j} B^{j}_{k}'
symbols = [
    'A__',
    'B__',
    '@g__'
    ]

model = lt.LaModel(latex, symbols)

result = model.forward({
    'A': torch.randn(3,3),
    'B': torch.randn(3,3),
    'g': torch.eye(3, 3)
    })
```
