import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from tensorboard.plugins import projector


def _make_sprite(images, spritesize=50):
    """
    Input a 4D tensor, output a sprite image. 
    assumes your pictures are all the
    same size and square
    """
    num_sprites = images.shape[0]
    imsize = images.shape[1]
    gridsize = np.int(np.ceil(np.sqrt(num_sprites)))
    output = np.zeros((imsize*gridsize, imsize*gridsize, 3), dtype=np.uint8)

    for i in range(num_sprites):
        col = i // gridsize
        row = i % gridsize
        output[imsize*col:imsize*(col+1), imsize*row:imsize*(row+1),:] = (255*images[i,:,:,:]).astype(np.uint8)
    img = Image.fromarray(output)
    img = img.resize((spritesize*gridsize, spritesize*gridsize))
    return img


def _save_metadata(labels, logdir):
    with open(logdir + "metadata.tsv",'w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(labels):
            f.write("%d\t%d\n" % (index,label))
            
            
def prepare_embedding_metadata(ds, logdir, spritesize=50):
    """
    Input a test dataset and log directory; save a sprite
    image and tsv containing labels.
    """
    # load the images
    ims, labs = ds.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        ims, labs = sess.run([ims, labs])
        
    # build the sprite image
    sprite = _make_sprite(ims, spritesize)
    sprite.save(logdir + "sprites.png")
    
    # write out labels
    _save_metadata(labs, logdir)
    
    
def add_conv_histograms():
    tensors = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
             if "conv" in x.name]
    for t in tensors:
        tf.summary.histogram(t.name.replace(":", "_"), t)
        
        
        
def generate_embedding_op(vec, N, spritesize=50):
    embed_dummy = tf.get_variable("dense_embeddings", shape=[N, vec.get_shape().as_list()[1]],
                              initializer=tf.initializers.random_uniform())
    store_embeddings = tf.assign(embed_dummy, vec)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embed_dummy.name
    embedding.metadata_path = "metadata.tsv"
    embedding.sprite.image_path = "sprites.png" 
    embedding.sprite.single_image_dim.extend([spritesize, spritesize])
    
    return store_embeddings, config