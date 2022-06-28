import tensorflow as tf

from tensorflow.keras import layers

transform_train  = tf.keras.Sequential([
                        layers.Resizing(227, 227),
                        layers.RandomFlip("horizontal_and_vertical"),
                        layers.RandomRotation(0.2)
                    ])
#                         T.Compose([
#                         T.Resize((227,227)),
#                         T.RandomApply([
#                             T.RandomChoice([
#                                 T.ColorJitter(hue=.05, saturation=.05),
#                                 T.RandomRotation(20),
#                                 T.RandomAffine(0, shear=0.2)
#                         ])], p = 0.5),
#                         T.RandomHorizontalFlip(p= 0.4),
#                         T.ToTensor(),
#                         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                     ])

transform_val  = tf.keras.Sequential([
                        layers.Resizing(227, 227),
                        layers.RandomFlip("horizontal_and_vertical"),
                        layers.RandomRotation(0.2)
                ])
                # T.Compose([
                #     T.Resize((227,227)),
                #     T.ToTensor(),
                #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # ])

transform_test  = tf.keras.Sequential([
                        layers.Resizing(227, 227),
                        layers.RandomFlip("horizontal_and_vertical"),
                        layers.RandomRotation(0.2)
                ])
                # T.Compose([
                #     T.Resize((227,227)),
                #     T.ToTensor(),
                #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # ])