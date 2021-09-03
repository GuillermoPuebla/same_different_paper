import io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

def translate_question(q):
    if len(q) != 11:
        return 'Not a proper question'
    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'gray']
    idx= np.argwhere(q[:6])[0][0]
    color = colors[idx]
    if q[6]:
        if q[8]:
            return "What's the shape of the nearest object to the object in " + color + "?" 
        elif q[9]:
            return "What's the shape of the farthest object away from the object in " + color + "?"
        elif q[10]:
            return 'How many objects have the same shape as the object in ' + color + '?'
    else:
        if q[8]:
            return 'Is the object in ' + color + ' a circle or a square?'
        elif q[9]:
            return 'Is the object in ' + color + ' on the bottom of the image?'
        elif q[10]:
            return 'Is the object in ' + color + ' on the left of the image?'
        
def translate_answer(a):
    if len(a) != 10:
        return 'Not a proper answer'
    if a[0]:
        return 'yes'
    if a[1]:
        return 'no'
    if a[2]:
        return 'square'
    if a[3]:
        return 'circle'
    return np.argwhere(a[4:])[0][0] + 1

class SortOfCLEVRGenerator(object):
    def __init__(self, img_size=128):
        self.img_size = img_size
        self.bg_color = (231, 231, 231)
        self.colors = (
            (252, 51, 56),      # red
            (103, 107, 251),    # blue
            (80, 179, 81),      # green
            (253, 116, 34),     # orange
            (255, 253, 92),     # yellow
            (129, 129, 129)     # grey
        ) 
        self.n_grid = 4
        self.num_color = len(self.colors)
        self.num_shape = 6
         # Avoid a color shared by more than one objects
        self.num_shape = min(self.num_shape, self.num_color)
        self.block_size = int(img_size*0.9/self.n_grid)
        self.shape_size = int((img_size*0.9/self.n_grid)*0.7/2)
        self.question_vector_size = 11
        self.answer_vector_size = 10

    def generate_sample(self):
        # Generate I: [img_size, img_size, 3]
        img = Image.new('RGB', (self.img_size, self.img_size), color=self.bg_color)
        drawer = ImageDraw.Draw(img)
        idx_coor = np.arange(self.n_grid*self.n_grid)
        np.random.shuffle(idx_coor)
        # Don't shuffle the colors! They are the criteria to identify the objects.
        idx_color_shape = np.arange(self.num_color)
        coin = np.random.rand(self.num_shape)
        X = []
        Y = []
        for i in range(self.num_shape):
            x = idx_coor[i] % self.n_grid
            y = (self.n_grid - np.floor(idx_coor[i] / self.n_grid) - 1).astype(np.uint8)
            # sqaure terms are added to remove ambiguity of distance
            position = ((x+0.5)*self.block_size-self.shape_size+x**2, (y+0.5)*self.block_size-self.shape_size+y**2,
                        (x+0.5)*self.block_size+self.shape_size+x**2, (y+0.5)*self.block_size+self.shape_size+y**2)
            X.append((x+0.5)*self.block_size+x**2)
            Y.append((y+0.5)*self.block_size+y**2)
            if coin[i] < 0.5:
                drawer.ellipse(position, fill=self.colors[idx_color_shape[i]], outline=(0, 0, 0))
            else:
                drawer.rectangle(position, fill=self.colors[idx_color_shape[i]], outline=(0, 0, 0))

        # Generate its representation
        color = idx_color_shape[:self.num_shape]
        shape = coin < 0.5
        shape_str = ['c' if x else 's' for x in shape] # c=circle, s=square

        representation = []
        for i in range(len(X)):
            center = (X[i], Y[i])
            shape = shape_str[i]
            representation.append([center, shape]) # color order is constant so you don't need it.
        
        return np.array(img), representation

    def biased_distance(self, center1, center2):
        """center1 and center2 are lists [x, y]"""
        return (center1[0] - center2[0]) ** 2 + 1.1*(center1[1] - center2[1]) ** 2
    
    def generate_questions(self, rep, number_questions=10):
        """
        Given a queried color, all the possible questions are as follows.
        Non-relational questions:
            Is it a circle or a rectangle?
            Is it on the bottom of the image?
            Is it on the left of the image?
        Relational questions:
            The shape of the nearest object?
            The shape of the farthest object?
            How many objects have the same shape?
        Questions are encoded into a one-hot vector of size 11:
        [red, blue, green, orange, yellow, gray, relational, non-relational, question 1, question 2, question 3]
        """
        questions = []
        for q in range(number_questions):
            for r in range(2):
                question = [0] * self.question_vector_size
                color = np.random.randint(6)
                question[color] = 1
                question[6 + r] = 1
                question_type = np.random.randint(3)
                question[8 + question_type] = 1
                questions.append(question)
        return questions

    def get_all_rel_questions(self, rep):
        questions = []
        # Iterate over color indexes.
        for i in range(6):
            # Iterate over question number.
            for j in range(3):
                # Initialize question vec.
                question = [0] * self.question_vector_size
                # Fill color
                question[i] = 1
                # Fill relational question type.
                question[6] = 1
                # Fill question number.
                question[8 + j] = 1
                # Append to list.
                questions.append(question)
        return questions
    
    def get_all_nonrel_questions(self, rep):
        questions = []
        # Iterate over color indexes.
        for i in range(6):
            # Iterate over question number.
            for j in range(3):
                # Initialize question vec.
                question = [0] * self.question_vector_size
                # Fill color
                question[i] = 1
                # Fill non-relational question type.
                question[7] = 1
                # Fill question number.
                question[8 + j] = 1
                # Append to list.
                questions.append(question)
        return questions
    
    def generate_answers(self, rep, questions):
        """
        The possible answer is a fixed length one-hot vector whose elements represent:
        [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]
        """
        answers = []
        for question in questions:
            answer = [0] * self.answer_vector_size
            color = question[:6].index(1)
            if question[6]:
                if question[8]: #The shape of the nearest object?
                    # dist = [((rep[color][0]-obj[0])**2).sum() for obj in rep]
                    dist = [self.biased_distance(rep[color][0], obj[0]) for obj in rep]
                    dist[dist.index(0)] = float('inf')
                    closest = dist.index(min(dist))
                    if rep[closest][1] == 's':
                        answer[2] = 1
                    else:
                        answer[3] = 1
                elif question[9]: #The shape of the farthest object?
                    dist = [self.biased_distance(rep[color][0], obj[0]) for obj in rep]
                    furthest = dist.index(max(dist))
                    if rep[furthest][1] == 's':
                        answer[2] = 1
                    else:
                        answer[3] = 1

                else: #How many objects have the same shape?
                    count = -1
                    shape = rep[color][1]
                    for obj in rep:
                        if obj[1] == shape:
                            count += 1
                    answer[count + 4] = 1
            else:
                if question[8]: #Is it a circle or a rectangle?
                    if rep[color][1] == 's':
                        answer[2] = 1
                    else:
                        answer[3] = 1
                elif question[9]: #Is it on the bottom of the image?
                    if rep[color][0][1] > self.img_size/2:
                        answer[0] = 1
                    else:
                        answer[1] = 1
                else: #Is it on the left of the image?
                    if rep[color][0][0] > self.img_size/2:
                        answer[1] = 1
                    else:
                        answer[0] = 1
            answers.append(answer)
        return answers

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

if __name__ == '__main__':
    # Parameters.
    TRAIN_SIZE = 9800
    TEST_SIZE = 200
    train_file_name = "data/sort_of_clevr_128_train.tfrecords"
    test_file_name_relational = "data/sort_of_clevr_128_relational_test.tfrecords"
    test_file_name_non_relational = "data/sort_of_clevr_128_non_relational_test.tfrecords"
    gen = SortOfCLEVRGenerator(img_size=128)

    # Make train dataset.
    with tf.io.TFRecordWriter(train_file_name) as writer:
        for i in range(TRAIN_SIZE):
            # Generate data.
            img, rep = gen.generate_sample()
            questions = gen.generate_questions(rep)
            answers = gen.generate_answers(rep, questions)
            questions = [np.array(q) for q in questions]
            answers = [np.array(a) for a in answers]

            for j in range(len(questions)):
                # Save image to bytes.
                pil_image=Image.fromarray(img)
                buf = io.BytesIO()
                pil_image.save(buf, format='PNG')
                byte_im = buf.getvalue()

                question = serialize_array(questions[j])
                answer = serialize_array(answers[j])
                feature = {
                    'image_raw': _bytes_feature(byte_im),
                    'question': _bytes_feature(question),
                    'answer': _bytes_feature(answer)}
                # Parse example.
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Write example.
                writer.write(example.SerializeToString())
    print('Train dataset save at \data')

    # Make test datasets.
    with tf.io.TFRecordWriter(test_file_name_relational) as writer1:
        with tf.io.TFRecordWriter(test_file_name_non_relational) as writer2:
            for i in range(TEST_SIZE):
                # Generate data.
                img, rep = gen.generate_sample()
                questions_rel = gen.get_all_rel_questions(rep)
                questions_nonrel = gen.get_all_nonrel_questions(rep)
                answers_rel = gen.generate_answers(rep, questions_rel)
                answers_nonrel = gen.generate_answers(rep, questions_nonrel)

                questions_rel = [np.array(q) for q in questions_rel]
                questions_nonrel = [np.array(q) for q in questions_nonrel]
                answers_rel = [np.array(a) for a in answers_rel]
                answers_nonrel = [np.array(a) for a in answers_nonrel]

                for j in range(len(questions_rel)):
                    # Save image to bytes.
                    pil_image=Image.fromarray(img)
                    buf = io.BytesIO()
                    pil_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()

                    # Write to relational test set.
                    question = serialize_array(questions_rel[j])
                    answer = serialize_array(answers_rel[j])
                    feature = {
                        'image_raw': _bytes_feature(byte_im),
                        'question': _bytes_feature(question),
                        'answer': _bytes_feature(answer)}
                    # Parse example.
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    # Write example.
                    writer1.write(example.SerializeToString())

                    # Write to non-relational test set.
                    question = serialize_array(questions_nonrel[j])
                    answer = serialize_array(answers_nonrel[j])
                    feature = {
                        'image_raw': _bytes_feature(byte_im),
                        'question': _bytes_feature(question),
                        'answer': _bytes_feature(answer)}
                    # Parse example.
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    # Write example.
                    writer2.write(example.SerializeToString())
    print('Done!')
    
    print('Test datasets saved at \data')
