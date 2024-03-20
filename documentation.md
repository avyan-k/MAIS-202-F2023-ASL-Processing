
# Update - Tracy - March 19th 2024

## main.js

We used to have a problem with the way the webcam was saving the image (it was compressing it instead of cropping) I fixed that

And I also changed the way it is displayed (not as a square anymore but as the default webcam size)

## app.py

Inside the ```receive_data()``` method, I get the image data that is an (480,640,4) array from javascript

I save the image in png format to the "images/saved_images" path with the helper method ```save_image()```

Then I call getLetter to get the letter

I have 3 helper functions:

1. ```save_image()``` saves the png image to the directory path "images/saved_images"

2. ```croped_shaped_image(folder_path)```

Takes the folder path ("images/saved_images") and finds the latest saved png image (which is of size 480x640) and then crop it and resize it to 512x512.

3. ```getLetter()```

Calls ```croped_shaped_image()``` to get the most recent image in the right size

Converts image to RGB then to media pipe image type then to hand landmard then to an array of float 32

Then I have a loader I put batch size at 4 but i tried it at batch size of 1 and it still didn't work

### Error Code

So im not sure why it says its the wrong size. You can see in the getLetter method I print out the sizes of the image and hand landmark and they all check out. so yeah ...


```File "/Users/tatax/Documents/GitHub/MAIS-202-F2023-ASL-Processing/src/CNN_model.py", line 41, in forward
    x = F.leaky_relu(layer(x))
  File "/Users/tatax/Library/Python/3.9/lib/python/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/tatax/Library/Python/3.9/lib/python/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/tatax/Library/Python/3.9/lib/python/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: linear(): input and weight.T shapes cannot be multiplied (4x2 and 42x64)
```

# Update - Tracy - March 19th 2024

Error fixed, just needed to unsqueeze it.

Updated README.md and Recorded Demo

Project Completed
