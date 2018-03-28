from realExample import real_example
from nnparser import Sequential

# run and print the real example
print("----real output----")
real_example()

code1 = '''    
    #num_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))'''

print("")
print("*" * 10 + " my output " + "*" * 10)

seq = Sequential()
seq.parse(code1)
seq.summary()

