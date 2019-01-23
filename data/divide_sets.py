import os
import sys
import glob
import utils.util as util

if len(sys.argv) is not 2:
    print("Usage: python divide_sets.py DATASET_PATH")
    sys.exit(-1)

dataset_path = sys.argv[1]
save_path = '../dataset'
util.mkdir(save_path)

tr = 0.8
te = 0.15
ev = 0.05

s = float(tr + te + ev)

p_train = float(tr / s)
p_test = float(te / s)
p_eval = float(ev / s)
assert(p_train + p_test + p_eval == 1)

train_set = []
test_set = []
eval_set = []

classes = glob.glob(os.path.join(dataset_path, '*'))
for cl in classes:
    img_list = glob.glob(os.path.join(cl, '*'))
    num_imgs = len(img_list)
    num_train = int(num_imgs * p_train)
    num_test = int(num_imgs * p_test)
    train_set += img_list[:num_train]
    test_set += img_list[num_train:num_train+num_test]
    eval_set += img_list[num_train+num_test:]

sets = {'train': train_set, 'test': test_set, 'eval': eval_set}
for k in sets.keys():
    dst = os.path.join(save_path, k + '.txt')
    f = open(dst, 'w')
    for img in sets[k]:
        f.write(img + '\n')
    f.close()
