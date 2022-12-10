from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import config
import random
import os
import time
from aiogram import Bot, Dispatcher, executor, types

bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(content_types=['photo'])
async def photo(message: types.Message):
    index_photo = random.randint(0, 1000)
    await message.photo[-1].download(r'C:\Users\ученик\PycharmProjects\pythonProjectaboba\{0}.png'.format(index_photo))

    np.set_printoptions(suppress=True)

    model = load_model('keras_Model.h5', compile=False)
    class_names = open('labels.txt', 'r').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(r'C:\Users\ученик\PycharmProjects\pythonProjectaboba\{0}.png'.format(index_photo)).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    if class_name[:1] == '0':
        await message.answer('Class: в шапке')
    else:
        await message.answer('Class: без шапки')
    time.sleep(10)
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{0}.png'.format(index_photo))
    os.remove(path)


executor.start_polling(dp)
