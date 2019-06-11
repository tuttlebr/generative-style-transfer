from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--content_path", help="String filename for your original jpeg.", required=True)
parser.add_argument("-s", "--style_path", help="String filename for your style jpeg.", required=True)
parser.add_argument("--epochs", type=int, default=1, required=False, help="Epochs to learn style.")
parser.add_argument("--steps_per_epoch", type=int, default=100, required=False, help="Steps-per-epoch to learn style.")
parser.add_argument("--style_weight", type=float, default=1e-2, required=False, help="Bias of the style")
parser.add_argument("--content_weight", type=float, default=1e4, required=False, help="Bias of the original")
parser.add_argument("--total_variation_weight", type=float, default=1e8, required=False, help="Global difference")
parser.add_argument("--max_dim", type=int, default=512, required=False, help="Longest dimension")

args = parser.parse_args()
    


    
def st_model(content_path=args.content_path,
             style_path=args.style_path,
             epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             style_weight=args.style_weight,
             content_weight=args.content_weight,
             total_variation_weight=args.total_variation_weight):


    def load_img(source, max_dim=args.max_dim):
        destination = os.path.join(os.getcwd(), os.path.basename(source))
        path_to_img = tf.keras.utils.get_file(destination, source)

        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img



    def saver(tensorarray):
        save_as = tensorarray.numpy()
        file_name = str(int(tf.timestamp().numpy())) + ".png"
        mpl.image.imsave(file_name, save_as)
        print('\nSaved style transfer image to: {}'.format(file_name))



    def vgg_layers(layer_names):
        """Creates a vgg model that returns a list of intermediate output values."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model



    def gram_matrix(input_tensor):
        """the style of an image can be described by the means and correlations across the different feature maps. 
        Calculate a Gram matrix that includes this information by taking the outer product of the feature vector 
        with itself at each location, and averaging that outer product over all locations."""

        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)



    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg =  vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs*255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                              outputs[self.num_style_layers:])

            style_outputs = [gram_matrix(style_output)
                             for style_output in style_outputs]

            content_dict = {content_name:value 
                            for content_name, value 
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name:value
                          for style_name, value
                          in zip(self.style_layers, style_outputs)}

            return {'content':content_dict, 'style':style_dict}



    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss



    def high_pass_x_y(image):
        x_var = image[:,:,1:,:] - image[:,:,:-1,:]
        y_var = image[:,1:,:,:] - image[:,:-1,:,:]

        return x_var, y_var



    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)



    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight*total_variation_loss(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))


    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 


    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']



    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    style_results = results['style']
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


    image = tf.Variable(content_image)

    step = 0
    for n in range(epochs):
        print("Running epoch {} of {}".format(n+1, epochs), end="\n")
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(f"{m/(steps_per_epoch+1)*100:.1f} %", end="\r")


    saver(image[0])
    
    
    
    
if __name__ == '__main__':
    st_model()
