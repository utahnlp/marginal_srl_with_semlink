import codecs
import os
import sys

input_data_path=sys.argv[1]
output_file_path=sys.argv[2]
output_props_file=sys.argv[3]
output_propid_file=sys.argv[4]
output_fileid_file=sys.argv[5]
output_origfileid_file=sys.argv[6]

tag_dict={}

fout = codecs.open(output_file_path, 'w')
fout_props = codecs.open(output_props_file, 'w')
fout_propid = codecs.open(output_propid_file, 'w')
fout_fileid = codecs.open(output_fileid_file, 'w')
fout_origfileid = codecs.open(output_origfileid_file, 'w')
#flist_out = open('filelist.out', 'w')

total_props = 0
total_props2 = 0
total_sents = 0
total_sents2 = 0

domain = ''
dpath = []

doc_counts = 0
v_counts = 0
ner_counts = 0

words = []
lemmas = []
props = []
tags = []
spans = []
all_props = []
frames = []
vnclasses = []

label_dict = {}

def print_new_sentence():
  global total_props
  global total_props2
  global total_sents
  global words
  global lemmas
  global props
  global tags
  global span
  global all_props
  global frames
  global vnclasses
  global file_id
  global sent_id
  global fout
  global fout_props
  global fout_propid
  global fout_fileid
  global fout_origfileid
  global fd_out

  ''' ALso output sentences without any predicates '''
  total_props += len(props)
  total_sents += 1


  propid_labels = ['O' for _ in words]
  prop_lemma = ['-' for _ in words]
  for t in range(len(props)):
    assert(len(tags[t]) == len(words))
    #assert(tags[t][props[t]] in {"B-V", "B-I"})
    fout.write(str(props[t]) + " " + " ".join(words) + " ||| " + " ".join(tags[t]) + " ||| " + " ".join(lemmas) + " ||| " + frames[t] + " ||| " + vnclasses[t] + " \n")
    propid_labels[props[t]] = 'V'
    #fd_out.write(domain + '\n')
    fout_fileid.write(file_id + '\t' + str(sent_id) + '\n')
  
  if not all(p == 'O' for p in propid_labels):
    fout_propid.write(" ".join(words) + " ||| " + " ".join(propid_labels) + " ||| " + " ".join(lemmas) + "\n")
  total_props2 += len(all_props)
  words = []
  lemmas = []
  props = []
  tags = []
  spans = []
  all_props = []
  frames = []
  vnclasses = []


root = os.path.dirname(input_data_path)
fin = codecs.open(input_data_path, mode='r', encoding='utf8')
doc_counts += 1
for line in fin:
  line = line.strip()
  if line == '':
    print_new_sentence()
    fout_props.write('\n')
    total_sents2 += 1
  
    words = []
    lemmas = []
    props = []
    tags = []
    spans = []
    all_props = []
    continue

  if line[0] == "#":
    if len(words) > 0:
      print_new_sentence()
      fout_props.write('\n')
      total_sents2 += 1
    continue

  info = line.split(' ')
  try:
    word = info[2]
  except UnicodeEncodeError:
    print(input_data_path)
    print(info[2])
    word = "*UNKNOWN*";

  file_id = "UNKNOWN"
  sent_id = int(info[0])

  words.append(word)
  idx = len(words) - 1
  if idx == 0:
    tags = [[] for _ in info[7:]]
    spans = ["" for _ in info[7:]]

  is_predicate = (info[4] != '-')
  is_verbal_predicate = False

  lemma = info[6] if info[4] != '-' else '-'
  lemmas.append(lemma)
  fout_props.write(lemma + '\t' + '\t'.join(info[7:]) + '\n')

  for t in range(len(tags)):
    arg = info[7 + t]
    label = arg.strip("()*")
    label_dict[arg] = 1

    if "(" in arg:
      tags[t].append("B-" + label)
      spans[t] = label
    elif spans[t] != "":
      tags[t].append("I-" + spans[t])
    else:
      tags[t].append("O")
    if ")" in arg:
      spans[t] = ""
    if "(V" in arg:
      is_verbal_predicate = True
      v_counts += 1

  if is_verbal_predicate:
    props.append(idx)
  if is_predicate:
    all_props.append(idx)
    frames.append(info[4])
    vnclasses.append(info[5])

fin.close()
''' Output last sentence.'''
if len(words) > 0:
  print_new_sentence()
  fout_props.write('\n')
  total_sents2 += 1

fout.close()
fout_props.close()
fout_propid.close()
fout_fileid.close()
fout_origfileid.close()
#fd_out.close()
#flist_out.close()

print('documents', doc_counts)
print('all sentences', total_sents, total_sents2)
print('props', total_props)
print('verbal props:', v_counts)
print('sentences', total_sents)
