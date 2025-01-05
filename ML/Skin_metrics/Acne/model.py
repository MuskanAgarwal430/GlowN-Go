import tensorflow as tf

print('Loading model ...')
model = tf.keras.models.load_model(r'C:\Users\hp\Desktop\Skyn-main\ML\Skin_metrics\Acne\saved_model\my_model')

class_names = ['Low', 'Moderate', 'Severe']


def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img / 255.0
    return img


def predict_class(filename):
    print('Loading image ...')
    img = load_and_prep_image(filename)

    print('Predicting class of image ...')

    pred = model.predict(tf.expand_dims(img, axis=0))
    print(pred)

    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    print('Predicted class:', pred_class)
    return pred_class


predict_class('test_image.jpeg')
