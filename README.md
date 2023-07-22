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
here, we define two new variables that we will use as the input to create the PyTorch model. The string `latex`{:.python} stores the latex expression, while the string `symbols` defines which latex symbols we want to use as arguments for our model. More specifically this variable also indicates the form the arguments will be given. For example, in this case, it specifies that both the `A` and `B` tensors are going to be passed as arguments with both indices down. This is indicated by the two '\_' after the name of the variable. In case we want to pass the tensor with an up index we can also use '\^' to represent an up index. Finally, the '@' before `g` indicates that `g` is the metric and can be used for index highering and lowering.

```python
model = lt.LaModel(latex, symbols)
```
in this line, we create the PyTorch model for the above expression.

```python
result = model.forward({
    'A': torch.randn(3,3),
    'B': torch.randn(3,3),
    'g': torch.eye(3, 3)
    })
```
By calling the `.forward(...)` method we can evaluate the model. In particular, this method accepts a dictionary as an argument, where the keys are the symbol names chosen in the `symbols` variable above (without possible indices) and the values are those that these variables should assume for the computation.











