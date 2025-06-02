from typing import Literal


TfType = Literal["linear", "sqrt", "log"]
IdfType = Literal["smooth", "standard"]
DlType = Literal["linear", "log"]
NormStr = Literal["l1", "l2"]

# TODO: use this types in the project, rather than just the basic types
# Token      = NewType("Token", str)
# TokenList  = list[Token]
# TweetID    = NewType("TweetID", str)
# TopicVec   = list[float]
