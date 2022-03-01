import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

'''
test_cases = {
    '1d': ([1, 2],
           [3, 4]),
    '2d': ([[1, 2], [3, 4]],
           [[5, 6], [7, 8]]),
    '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
           [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
}  # type: Dict[Text, Sequence[Any]]
'''

test_cases = {
    '2d': ([[1, 2], [3, 4], [10, 11]],
           [[5, 6], [7, 8]]),
} 
for test_case, values_ in test_cases.items():
    print(values_)
    values = [np.asarray(v, dtype=np.float32) for v in values_]
    print(values)
    print(values[0].shape)
    print(values[1].shape)
    for i in range(len(values[0].shape)):
        in_args = ['value' + str(k) for k in range(len(values))]
        print(in_args)
        node = onnx.helper.make_node(
            'Concat',
            inputs=[s for s in in_args],
            outputs=['output'],
            axis=i
        )
        output = np.concatenate(values, i)
        print(i)
        print(output)
        #expect(node, inputs=[v for v in values], outputs=[output],
        #       name='test_concat_' + test_case + '_axis_' + str(i))

'''
    for i in range(-len(values[0].shape), 0):
        in_args = ['value' + str(k) for k in range(len(values))]
        node = onnx.helper.make_node(
            'Concat',
            inputs=[s for s in in_args],
            outputs=['output'],
            axis=i
        )
        output = np.concatenate(values, i)
        print(output)
        #expect(node, inputs=[v for v in values], outputs=[output],
               #name='test_concat_' + test_case + '_axis_negative_' + str(abs(i)))
'''

