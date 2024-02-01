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

prev_words = ''
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
  global file_id
  global orig_file_id
  global sent_id
  global fout
  global fout_props
  global fout_propid
  global fout_fileid
  global fout_origfileid
  global fd_out

  ''' ALso output sentences without any predicates '''
  #if len(props) > 0:
  total_props += len(props)
  total_sents += 1
  #print(len(props), len(tags))
  #print(words)
  #print(props)
  #print(tags)
  #assert(len(props) == len(tags))

  propid_labels = ['O' for _ in words]
  prop_lemma = ['-' for _ in words]
  for t in range(len(props)):
    assert(len(tags[t]) == len(words))
    #assert(tags[t][props[t]] in {"B-V", "B-I"})
    fout.write(str(props[t]) + " " + " ".join(words) + " ||| " + " ".join(tags[t]) + " ||| " + " ".join(lemmas) + " ||| " + frames[t] + " \n")
    propid_labels[props[t]] = 'V'
    #fd_out.write(domain + '\n')
    fout_fileid.write(file_id + '\t' + str(sent_id) + '\n')
    fout_origfileid.write(orig_file_id + '\t' + str(sent_id) + '\n')
  
  fout_propid.write(" ".join(words) + " ||| " + " ".join(propid_labels) + " ||| " + " ".join(lemmas) + "\n")
  total_props2 += len(all_props)
  words = []
  lemmas = []
  props = []
  tags = []
  spans = []
  all_props = []
  frames = []


for root, dirs, files in os.walk(input_data_path):
  for f in files:
    if not '.conll' in f:
      continue
    #print root, dirs, f
    dpath = root.split('/')
    #domain = '_'.join(dpath[dpath.index('annotations')+1:-1])
    fin = codecs.open(root + "/" + f, mode='r', encoding='utf8')
    #flist_out.write(f + '\n')
    doc_counts += 1
    for line in fin:
      line = line.strip()
      if line == '':
        joined_words = " ".join(words)
        joined_lemma = " ".join(lemmas)
        #if joined_words == prev_words:
        #  print "Skipping dup sentence in: ", root, f
        #else:
        prev_words = joined_words
        prev_lemmas = joined_lemma
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
        prev_words = ""
        if len(words) > 0:
          print_new_sentence()
          fout_props.write('\n')
          total_sents2 += 1
        continue

      info = line.split(' ')
      try:
        word = info[3]
      except UnicodeEncodeError:
        print(root, dirs, f)
        print(info[3])
        word = "*UNKNOWN*";

      orig_file_id = info[0]
      file_id = info[0].split('/')[-1]
      sent_id = int(info[1])

      words.append(word)
      idx = len(words) - 1
      if idx == 0:
        tags = [[] for _ in info[11:-1]]
        spans = ["" for _ in info[11:-1]]

      is_predicate = (info[7] != '-')
      is_verbal_predicate = False

      lemma = info[6] if info[7] != '-' else '-'
      lemmas.append(lemma)
      fout_props.write(lemma + '\t' + '\t'.join(info[11:-1]) + '\n')

      for t in range(len(tags)):
        arg = info[11 + t]
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
      
      if '(' in info[10]:
        ner_counts += 1

      if is_verbal_predicate:
        props.append(idx)
      if is_predicate:
        all_props.append(idx)
        frames.append(info[7])

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
print('ner counts:', ner_counts)
print('sentences', total_sents)
