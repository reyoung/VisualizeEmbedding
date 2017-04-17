import sys
import paddle.v2 as paddle
import numpy

params = paddle.parameters.Parameters.from_tar(open(sys.argv[1]))
numpy.savez(sys.stdout, **params)
