import tensorflow as tf

from tensorflow.keras import layers
import albumentations as A

transform_train = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
    A.Resize(227,227, always_apply=True)
])

# transform_train  = tf.keras.Sequential([
#                         layers.Resizing(227, 227),
#                         layers.RandomFlip("horizontal_and_vertical"),
#                         layers.RandomRotation(0.2)
#                     ])                      

transform_val = A.Compose([
#     A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
#     A.FancyPCA(alpha=0.1, always_apply=False, p=0.5)
    A.Resize(227,227, always_apply=True)

])

# transform_val  = tf.keras.Sequential([
#                         layers.Resizing(227, 227),
#                         layers.RandomFlip("horizontal_and_vertical"),
#                         layers.RandomRotation(0.2)
#                 ])

transform_test = A.Compose([
#     A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
#     A.FancyPCA(alpha=0.1, always_apply=False, p=0.5)
    A.Resize(227,227, always_apply=True)

])

# transform_test  = tf.keras.Sequential([
#                         layers.Resizing(227, 227),
#                         layers.RandomFlip("horizontal_and_vertical"),
#                         layers.RandomRotation(0.2)
#                 ])
