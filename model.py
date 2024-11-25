# Basic CNN 모델 정의
input_shape = x_train.shape[1:]
inputs = Input(shape=input_shape)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[
        ModelCheckpoint("basic_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=10, verbose=1, min_lr=1e-5)
    ]
)

# 모델 평가
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Basic CNN Validation Loss: {val_loss:.4f}")
print(f"Basic CNN Validation Accuracy: {val_acc:.4f}")

history_basic = model.fit(train_generator, epochs=10, validation_data=val_generator)
# Residual Block 정의
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# ResNet 모델 정의
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Residual Blocks 추가
x = residual_block(x, 32)
x = residual_block(x, 64, stride=2)
x = residual_block(x, 128, stride=2)

# Global Average Pooling 및 Fully Connected Layer
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# MobileNet 모델 정의
input_shape = (32, 32, 3)

# Pre-trained MobileNet 사용
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# MobileNet의 출력에 새로운 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Base 모델의 가중치를 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[
        ModelCheckpoint("mobilenet_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=10, verbose=1, min_lr=1e-5)
    ]
)

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"MobileNet Test Loss: {test_loss:.4f}")
print(f"MobileNet Test Accuracy: {test_acc:.4f}")

history_mobilenet = model.fit(train_generator, epochs=10, validation_data=val_generator)
