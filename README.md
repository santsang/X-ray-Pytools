# X-ray-Pytools - Scripts

I think most programmes here are intuitive.  For a scientist, repeatability is a number one concern.  It took me sometimes to figure our that we need to set both global and operational seeds to get repeatable results from Keras.  This issue is fixed by set_all_seed() in the _auxNN.py_ file.

I may also explain the algorithm of the _edit_image.py_.  For now, I may note that I feel that impaint may improve the quality of an image at visual level.  You may apply it for editing images one-by-one.
