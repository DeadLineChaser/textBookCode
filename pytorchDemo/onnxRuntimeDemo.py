import onnxruntime as ort
import numpy as np
import tkinter as tk

import torch
from PIL import Image, ImageTk



Session = ort.InferenceSession('myNN.onnx', providers=['CPUExecutionProvider'])

image = Image.open("1d3323c2aedb855387cf50847cf579ac.jpeg").resize((32, 32))
image_array = np.array(image)
image_array = image_array.transpose((2, 0, 1)).astype(np.float32)
image_array = np.expand_dims(image_array, axis=0)

input_name = Session.get_inputs()[0].name
output_name = Session.get_outputs()[0].name
input_dict = {input_name: image_array}
outputs = Session.run([output_name], input_dict)


# 创建左边的图片展示区域
root = tk.Tk()
root.title("Inference Result Display")
image = image.resize((320, 320))
photo = ImageTk.PhotoImage(image)
classes_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
tensor_outputs = torch.from_numpy(outputs[0])
softmax_result = torch.softmax(tensor_outputs, dim=1)
values, indices = torch.topk(softmax_result, k=3, dim=1)
image_label = tk.Label(root, image=photo)
image_label.image = photo
image_label.pack(side=tk.LEFT)
# 创建右边的文本展示区域
text_label = tk.Label(root, text=f"{classes_name[int(indices[0][0])]}:{values[0][0]}\n"
                                 f"{classes_name[int(indices[0][1])]}:{values[0][1]}\n"
                                 f"{classes_name[int(indices[0][2])]}:{values[0][2]}", wraplength=200, justify=tk.LEFT)
text_label.pack(side=tk.RIGHT)

root.mainloop()
