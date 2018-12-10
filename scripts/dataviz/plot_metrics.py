'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import matplotlib.pyplot as plt
import json
import tensorflow as tf
import os.path
import time
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("mode", "eval", "")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
    
    raw_filenames = tf.gfile.Glob("./ckpts/*/metrics.*.json")
    model_names = list(set([os.path.basename(os.path.dirname(x)) for x in raw_filenames]))
    target_filenames = []
    for model_name in model_names:
        best_filename, best_time = None, -999999.999999
        for filename in raw_filenames:
            if model_name in filename:
                x_time = float(os.path.basename(filename)[8:-5])
                if x_time > best_time:
                    best_filename = filename
                    best_time = x_time
        target_filenames.append(best_filename)
    for i, x in enumerate(target_filenames):
        print(x)
        with open(x, "r") as f:
            metric_names, values = list(zip(*list(sorted(json.load(f).items(), key=lambda i: i[0]))))
            plt.bar(
                i + np.arange(len(metric_names)) * (len(model_names) + 1),
                values,
                tick_label=metric_names if i == len(model_names)//2 else None)
    plt.title("Comparison of performance on {0} dataset".format(FLAGS.mode))
    plt.xlabel("Evaluation metric name")
    plt.ylabel("Value achieved by model")
    plt.legend(model_names)
    plt.savefig("{0}.metrics.{1}.png".format(FLAGS.mode, str(time.time())))


if __name__ == "__main__":
    
    tf.app.run()