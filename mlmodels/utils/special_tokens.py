# -*- coding: utf-8 -*-
"""
Created on 2020-03-10
@author: duytinvo
"""
import enum
# ----------------------
#    personal symbols
# ----------------------
PAD = "<pad>"
SOT = "<s>"
EOT = "</s>"
UNK = "<unk>"
COL = '<col>'
TAB = '<tab>'
NL = "\\n"
NL2LC = '<nl2lc>'

PAD_id = 0
SOT_id = 1
EOT_id = 2
UNK_id = 3
COL_id = 4
TAB_id = 5

NULL = '<null>'
SEP = '<sep>'
SENSP = "<ssp>"  # sentiment space
SENGE = "<sge>"  # sentiment generation

SEP = "<sep>"
CLS = "<cls>"
MASK = "<mask>"

# ----------------------
#    BERT symbols
# ----------------------
BSEP = u"[SEP]"
BUNK = u"[UNK]"
BPAD = u"[PAD]"
BCLS = u"[CLS]"
BMASK = u"[MASK]"
BPRE = "##"

# ----------------------
#    BPE symbols
# ----------------------
# charBPE
pad_token = PAD
unk_token = UNK
suffix_token = "</w>"

# sentBPE
rep_token = "▁"
