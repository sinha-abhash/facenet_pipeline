from __future__ import unicode_literals

from scipy import misc
import tensorflow as tf
import numpy as np
import math
import sys
import os
import argparse
import facenet
import detect_face
import json, codecs
from prompt_toolkit import prompt
import logging
import cv2


logger = logging.getLogger('face recognition')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
logger.addHandler(ch)


def main(args):
    # create database folder which will contain a json file with known user embeddings
    userDBFolder = 'userDBFolder'
    if not os.path.exists(userDBFolder):
        os.makedirs(userDBFolder)

    # load and align image
    faces, bboxes = load_and_align_data(args.image_file, args.image_size, args.margin)

    # get embedding from facenet for the input image
    embeddings = getEmbedding(args, faces)

    # create user database file if not already exists
    userDBFile = 'userDBfile.json'
    dbPath = os.path.join(userDBFolder, userDBFile)

    # names of all the persons in the image
    names = []

    # setting logging to info
    logger.setLevel(logging.INFO)

    confidence_scores = []  # contains confidence scores for each faces in the image
    name_asked = False  # keep track if name of the person is asked or not

    distances, found_dist = [], []
    if not os.path.isfile(dbPath):
        name = prompt('First time registration!! Please enter the name to register:')
        json_dump = json.dumps({name: embeddings[0].tolist()})
        with open(dbPath, 'w') as f:
            f.write(json_dump)
        logger.info('Added')
        names = [name]
        name_asked = True
    else:
        json_load = codecs.open(dbPath, 'r', encoding='utf-8').read()
        json_load = json.loads(json_load)
        found = False


        for embed in embeddings:
            name_found = False
            dist_embed = []   # contains distances between the current unknown face and all the other known faces
            for k, v in json_load.items():
                # calculate the cosine distance between the unknown embedding and known embedding
                dist = cosine(embed, np.array(v))
                dist_embed.append(dist)
                if dist < args.tolerance:
                    names.append(k)
                    found_dist.append(dist)
                    name_found = True
            if not name_found:
                # append '?' for those persons whose name is not found in the DB
                names.append('?')

                # append sum of all the distances calculated above if distance less than threshold is not found
                found_dist.append(sum(dist_embed))
            else:
                found = True
            distances.append(dist_embed)


        if not found:
            name = prompt('The user doesn\'t exist in the DB. Please enter the name to register:')
            json_load[name] = embeddings[0].tolist()    # name of first person in the image is always asked.
            with open(dbPath, 'w') as f:
                f.write(json.dumps(json_load))
            logger.info('Added')
            names = [name]
            name_asked = True

    if name_asked:      # add confidence score for the person whose name has been asked
        confidence_scores.append(1.0)
    for id, dist_embed in enumerate(distances):
        if not name_asked:
            if len(dist_embed) == 1:
                confidence_scores.append(1.0)
            else:
                min_dist = found_dist[id]
                confidence_score = 1 - (min_dist / sum(dist_embed))
                print(min_dist, dist_embed, confidence_score)
                confidence_scores.append(confidence_score)

    # add rectangle around the face in the image
    im = cv2.imread(args.image_file[0])
    for bb, name, score in zip(bboxes, names, confidence_scores):
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

        # add name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(im, name, (bb[0] + 5, bb[3] + 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(im, 'confidence score: %s' %(str(score)), (bb[0] + 5, bb[3] + 28), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('image', im)
    cv2.waitKey(0)


def cosine(emb1, emb2):
    sim = np.inner(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if sim < 1.0:
        dist = np.arccos(sim) / math.pi
    else:
        dist = 0.0
    return dist


def getEmbedding(args, images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


def load_and_align_data(image_path, image_size, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_list = []
    img = misc.imread(os.path.expanduser(image_path[0]), mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face, ", image_path)
        return
    box_for_each_face = []
    for box in bounding_boxes:
        det = np.squeeze(box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        box_for_each_face.append(bb)
    images = np.stack(img_list)
    box_for_each_face = np.stack(box_for_each_face)
    return images, box_for_each_face


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_file', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--tolerance', type=float,
                        help='Keep value low for more strict face recognition.', default=0.35)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
