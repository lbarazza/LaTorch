# LaTorch
LaTorch is a simple-to-use library to convert LaTeX-like tensor expressions into PyTorch models. It was developed with numerical relativity in mind and, as such, also handles automatic index highering and lowering.

## How to Use
Let us start with a simple example: the product between two $3 \times 3$ matrices

$$ A^i_{ㅤj}ㅤB^j_{ㅤi} $$

which, in LaTeX, would be written as

```latex
A^{i}_{j} B^{j}_{i}
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
```
here, we define two new variables that we will use as the input to create the PyTorch model. The string `latex`{:.python} stores the latex expression, while the string `symbols`{:.python} defines which latex symbols we want to use as arguments for our model. The '_' after the variable name indicates that the variable is a tensor with a down index for every '_' used. We could also use '^' to represent an up index. Hence, we cou be 'A^_', 'A

```python
model = lt.LaModel(latex, symbols)
```
in this line we create the PyTorch model for the above expression.

```python
result = model.forward({
    'A': torch.randn(3,3),
    'B': torch.randn(3,3),
    'g': torch.eye(3, 3)
    })
```
By calling the `forward`{:.python} method we can evaluate the model. In particular the `forward`{:.python} method accepts a dictionary as an argument, which specifies the latex symbol and the value we want it to have to perform the calculation.












