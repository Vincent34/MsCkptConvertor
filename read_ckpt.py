import sys
import mindspore as ms
from mindspore.train.serialization import load_checkpoint

param_dict = load_checkpoint(sys.argv[1])
for x in param_dict:
    print("----------------")
    print(x)
    weight = param_dict[x].asnumpy()
    print(weight.shape)
