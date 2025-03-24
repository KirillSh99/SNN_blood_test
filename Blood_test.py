import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), pool_size=(2, 2)):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, activation='relu')
        self.batch_norm = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(pool_size)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return self.pool(x)


class CNNModel(models.Model):
    def __init__(self, input_shape, num_classes, num_layers):
        super(CNNModel, self).__init__()
        self.conv_blocks = models.Sequential(
            [CNNBlock(32) for _ in range(num_layers)]
        )

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.conv_blocks(input)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


train_dir = 'OUR-PROJECT-DATASET/train'
val_dir = 'OUR-PROJECT-DATASET/val'
test_dir = 'OUR-PROJECT-DATASET/test'

img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)
epochs = 30
batch_size = 16
num_classes = 10

np_train_samples = 2700
np_validation_samples = 100
np_test_samples = 100

model = CNNModel(input_shape, num_classes, num_layers=3)
model.build(input_shape=(None,) + input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
test_generator = datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

model.fit(train_generator, steps_per_epoch=np_train_samples // batch_size, epochs=epochs, validation_data=val_generator, validation_steps=np_validation_samples // batch_size)

test_loss, test_acc = model.evaluate(test_generator, steps=np_test_samples // batch_size)
print(f"Точность модели: {round(test_acc, 2)}")

print("Сохраняем модель")
model_json = model.to_json()
with open("model_project.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_project.weights.h5")
print("Сохранение завершено")