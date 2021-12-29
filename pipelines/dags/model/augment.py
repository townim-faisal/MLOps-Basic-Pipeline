import torchvision.transforms as T
from PIL import Image 

transform_train  = T.Compose([
                        T.Resize((227,227)),
                        T.RandomApply([
                            T.RandomChoice([
                                T.ColorJitter(hue=.05, saturation=.05),
                                T.RandomRotation(20),
                                T.RandomAffine(0, shear=0.2)
                        ])], p = 0.5),
                        T.RandomHorizontalFlip(p= 0.4),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

transform_val  = T.Compose([
                    T.Resize((227,227)),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

transform_test  = T.Compose([
                    T.Resize((227,227)),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])