import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    torch.__version__
    return (torch,)


@app.cell
def _(torch):
    torch.cuda.is_available()
    return


@app.cell
def _(torch):
    tensor1d = torch.tensor([1, 2, 3])
    print(tensor1d.dtype)
    return (tensor1d,)


@app.cell
def _(tensor1d):
    print(tensor1d)
    return


@app.cell
def _(torch):
    floatvec = torch.tensor([1.0, 2.0, 3])
    print(floatvec.dtype), print(floatvec)
    return


@app.cell
def _(tensor1d, torch):
    floatvec2 = tensor1d.to(torch.float)
    floatvec2.dtype
    return


@app.cell
def _(torch):
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(tensor2d)
    return (tensor2d,)


@app.cell
def _(tensor1d, tensor2d):
    print(tensor1d.shape), print(tensor2d.shape)
    return


@app.cell
def _(tensor1d):
    print(tensor1d.reshape(1,3))
    return


@app.cell
def _(tensor2d):
    print(tensor2d.reshape(3,2))
    return


@app.cell
def _(tensor2d):
    print(tensor2d.view(3,2))
    return


@app.cell
def _(tensor2d):
    print(tensor2d.T)
    return


@app.cell
def _(tensor2d):
    print(tensor2d.view(3,2).T)
    return


@app.cell
def _(tensor2d):
    print(tensor2d.matmul(tensor2d.T))
    return


@app.cell
def _(tensor2d):
    print(tensor2d @ tensor2d.T)
    return


@app.cell
def _(torch):
    tensor3d = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    print(tensor3d)
    return (tensor3d,)


@app.cell
def _(tensor3d):
    tensor3d.shape
    return


@app.cell
def _(tensor3d):
    tensor3d.view(1,8), tensor3d.view(2,4), tensor3d.view(4,2), tensor3d.view(4,2,1), tensor3d.view(1,2,4), tensor3d.view(1,4,2), tensor3d.view(8)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
