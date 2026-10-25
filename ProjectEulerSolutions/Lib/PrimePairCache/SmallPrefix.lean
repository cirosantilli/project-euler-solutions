import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows000
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows001
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows002
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows003
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows004
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows005
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows006
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows007
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows008
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows009
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows010
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows011
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows012
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows013
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows014
import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows015

namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallEmpty : Graph := .empty
theorem smallEmpty_agrees : Graph.Agrees PrimePairVertices.smallTree smallEmpty := Graph.empty_agrees _

def smallNode3 : Graph := .node 3 smallRow3 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode3_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode3 :=
  Graph.node_agrees _ 3 smallRow3 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow3_checked

def smallNode7 : Graph := .node 7 smallRow7 smallNode3 smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode7_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode7 :=
  Graph.node_agrees _ 7 smallRow7 smallNode3 smallEmpty
    smallNode3_agrees smallEmpty_agrees (by decide) (by decide) smallRow7_checked

def smallNode13 : Graph := .node 13 smallRow13 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode13_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode13 :=
  Graph.node_agrees _ 13 smallRow13 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow13_checked

def smallNode11 : Graph := .node 11 smallRow11 smallNode7 smallNode13
set_option maxRecDepth 4096 in
theorem smallNode11_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode11 :=
  Graph.node_agrees _ 11 smallRow11 smallNode7 smallNode13
    smallNode7_agrees smallNode13_agrees (by decide) (by decide) smallRow11_checked

def smallNode19 : Graph := .node 19 smallRow19 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode19_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode19 :=
  Graph.node_agrees _ 19 smallRow19 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow19_checked

def smallNode29 : Graph := .node 29 smallRow29 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode29_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode29 :=
  Graph.node_agrees _ 29 smallRow29 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow29_checked

def smallNode23 : Graph := .node 23 smallRow23 smallNode19 smallNode29
set_option maxRecDepth 4096 in
theorem smallNode23_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode23 :=
  Graph.node_agrees _ 23 smallRow23 smallNode19 smallNode29
    smallNode19_agrees smallNode29_agrees (by decide) (by decide) smallRow23_checked

def smallNode17 : Graph := .node 17 smallRow17 smallNode11 smallNode23
set_option maxRecDepth 4096 in
theorem smallNode17_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode17 :=
  Graph.node_agrees _ 17 smallRow17 smallNode11 smallNode23
    smallNode11_agrees smallNode23_agrees (by decide) (by decide) smallRow17_checked

def smallNode37 : Graph := .node 37 smallRow37 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode37_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode37 :=
  Graph.node_agrees _ 37 smallRow37 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow37_checked

def smallNode43 : Graph := .node 43 smallRow43 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode43_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode43 :=
  Graph.node_agrees _ 43 smallRow43 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow43_checked

def smallNode41 : Graph := .node 41 smallRow41 smallNode37 smallNode43
set_option maxRecDepth 4096 in
theorem smallNode41_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode41 :=
  Graph.node_agrees _ 41 smallRow41 smallNode37 smallNode43
    smallNode37_agrees smallNode43_agrees (by decide) (by decide) smallRow41_checked

def smallNode53 : Graph := .node 53 smallRow53 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode53_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode53 :=
  Graph.node_agrees _ 53 smallRow53 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow53_checked

def smallNode61 : Graph := .node 61 smallRow61 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode61_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode61 :=
  Graph.node_agrees _ 61 smallRow61 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow61_checked

def smallNode59 : Graph := .node 59 smallRow59 smallNode53 smallNode61
set_option maxRecDepth 4096 in
theorem smallNode59_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode59 :=
  Graph.node_agrees _ 59 smallRow59 smallNode53 smallNode61
    smallNode53_agrees smallNode61_agrees (by decide) (by decide) smallRow59_checked

def smallNode47 : Graph := .node 47 smallRow47 smallNode41 smallNode59
set_option maxRecDepth 4096 in
theorem smallNode47_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode47 :=
  Graph.node_agrees _ 47 smallRow47 smallNode41 smallNode59
    smallNode41_agrees smallNode59_agrees (by decide) (by decide) smallRow47_checked

def smallNode31 : Graph := .node 31 smallRow31 smallNode17 smallNode47
set_option maxRecDepth 4096 in
theorem smallNode31_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode31 :=
  Graph.node_agrees _ 31 smallRow31 smallNode17 smallNode47
    smallNode17_agrees smallNode47_agrees (by decide) (by decide) smallRow31_checked

def smallNode71 : Graph := .node 71 smallRow71 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode71_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode71 :=
  Graph.node_agrees _ 71 smallRow71 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow71_checked

def smallNode79 : Graph := .node 79 smallRow79 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode79_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode79 :=
  Graph.node_agrees _ 79 smallRow79 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow79_checked

def smallNode73 : Graph := .node 73 smallRow73 smallNode71 smallNode79
set_option maxRecDepth 4096 in
theorem smallNode73_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode73 :=
  Graph.node_agrees _ 73 smallRow73 smallNode71 smallNode79
    smallNode71_agrees smallNode79_agrees (by decide) (by decide) smallRow73_checked

def smallNode89 : Graph := .node 89 smallRow89 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode89_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode89 :=
  Graph.node_agrees _ 89 smallRow89 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow89_checked

def smallNode101 : Graph := .node 101 smallRow101 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode101_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode101 :=
  Graph.node_agrees _ 101 smallRow101 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow101_checked

def smallNode97 : Graph := .node 97 smallRow97 smallNode89 smallNode101
set_option maxRecDepth 4096 in
theorem smallNode97_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode97 :=
  Graph.node_agrees _ 97 smallRow97 smallNode89 smallNode101
    smallNode89_agrees smallNode101_agrees (by decide) (by decide) smallRow97_checked

def smallNode83 : Graph := .node 83 smallRow83 smallNode73 smallNode97
set_option maxRecDepth 4096 in
theorem smallNode83_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode83 :=
  Graph.node_agrees _ 83 smallRow83 smallNode73 smallNode97
    smallNode73_agrees smallNode97_agrees (by decide) (by decide) smallRow83_checked

def smallNode107 : Graph := .node 107 smallRow107 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode107_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode107 :=
  Graph.node_agrees _ 107 smallRow107 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow107_checked

def smallNode113 : Graph := .node 113 smallRow113 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode113_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode113 :=
  Graph.node_agrees _ 113 smallRow113 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow113_checked

def smallNode109 : Graph := .node 109 smallRow109 smallNode107 smallNode113
set_option maxRecDepth 4096 in
theorem smallNode109_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode109 :=
  Graph.node_agrees _ 109 smallRow109 smallNode107 smallNode113
    smallNode107_agrees smallNode113_agrees (by decide) (by decide) smallRow109_checked

def smallNode131 : Graph := .node 131 smallRow131 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode131_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode131 :=
  Graph.node_agrees _ 131 smallRow131 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow131_checked

def smallNode139 : Graph := .node 139 smallRow139 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode139_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode139 :=
  Graph.node_agrees _ 139 smallRow139 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow139_checked

def smallNode137 : Graph := .node 137 smallRow137 smallNode131 smallNode139
set_option maxRecDepth 4096 in
theorem smallNode137_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode137 :=
  Graph.node_agrees _ 137 smallRow137 smallNode131 smallNode139
    smallNode131_agrees smallNode139_agrees (by decide) (by decide) smallRow137_checked

def smallNode127 : Graph := .node 127 smallRow127 smallNode109 smallNode137
set_option maxRecDepth 4096 in
theorem smallNode127_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode127 :=
  Graph.node_agrees _ 127 smallRow127 smallNode109 smallNode137
    smallNode109_agrees smallNode137_agrees (by decide) (by decide) smallRow127_checked

def smallNode103 : Graph := .node 103 smallRow103 smallNode83 smallNode127
set_option maxRecDepth 4096 in
theorem smallNode103_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode103 :=
  Graph.node_agrees _ 103 smallRow103 smallNode83 smallNode127
    smallNode83_agrees smallNode127_agrees (by decide) (by decide) smallRow103_checked

def smallNode67 : Graph := .node 67 smallRow67 smallNode31 smallNode103
set_option maxRecDepth 4096 in
theorem smallNode67_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode67 :=
  Graph.node_agrees _ 67 smallRow67 smallNode31 smallNode103
    smallNode31_agrees smallNode103_agrees (by decide) (by decide) smallRow67_checked

def smallNode151 : Graph := .node 151 smallRow151 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode151_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode151 :=
  Graph.node_agrees _ 151 smallRow151 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow151_checked

def smallNode163 : Graph := .node 163 smallRow163 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode163_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode163 :=
  Graph.node_agrees _ 163 smallRow163 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow163_checked

def smallNode157 : Graph := .node 157 smallRow157 smallNode151 smallNode163
set_option maxRecDepth 4096 in
theorem smallNode157_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode157 :=
  Graph.node_agrees _ 157 smallRow157 smallNode151 smallNode163
    smallNode151_agrees smallNode163_agrees (by decide) (by decide) smallRow157_checked

def smallNode173 : Graph := .node 173 smallRow173 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode173_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode173 :=
  Graph.node_agrees _ 173 smallRow173 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow173_checked

def smallNode181 : Graph := .node 181 smallRow181 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode181_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode181 :=
  Graph.node_agrees _ 181 smallRow181 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow181_checked

def smallNode179 : Graph := .node 179 smallRow179 smallNode173 smallNode181
set_option maxRecDepth 4096 in
theorem smallNode179_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode179 :=
  Graph.node_agrees _ 179 smallRow179 smallNode173 smallNode181
    smallNode173_agrees smallNode181_agrees (by decide) (by decide) smallRow179_checked

def smallNode167 : Graph := .node 167 smallRow167 smallNode157 smallNode179
set_option maxRecDepth 4096 in
theorem smallNode167_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode167 :=
  Graph.node_agrees _ 167 smallRow167 smallNode157 smallNode179
    smallNode157_agrees smallNode179_agrees (by decide) (by decide) smallRow167_checked

def smallNode193 : Graph := .node 193 smallRow193 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode193_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode193 :=
  Graph.node_agrees _ 193 smallRow193 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow193_checked

def smallNode199 : Graph := .node 199 smallRow199 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode199_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode199 :=
  Graph.node_agrees _ 199 smallRow199 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow199_checked

def smallNode197 : Graph := .node 197 smallRow197 smallNode193 smallNode199
set_option maxRecDepth 4096 in
theorem smallNode197_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode197 :=
  Graph.node_agrees _ 197 smallRow197 smallNode193 smallNode199
    smallNode193_agrees smallNode199_agrees (by decide) (by decide) smallRow197_checked

def smallNode223 : Graph := .node 223 smallRow223 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode223_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode223 :=
  Graph.node_agrees _ 223 smallRow223 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow223_checked

def smallNode229 : Graph := .node 229 smallRow229 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode229_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode229 :=
  Graph.node_agrees _ 229 smallRow229 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow229_checked

def smallNode227 : Graph := .node 227 smallRow227 smallNode223 smallNode229
set_option maxRecDepth 4096 in
theorem smallNode227_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode227 :=
  Graph.node_agrees _ 227 smallRow227 smallNode223 smallNode229
    smallNode223_agrees smallNode229_agrees (by decide) (by decide) smallRow227_checked

def smallNode211 : Graph := .node 211 smallRow211 smallNode197 smallNode227
set_option maxRecDepth 4096 in
theorem smallNode211_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode211 :=
  Graph.node_agrees _ 211 smallRow211 smallNode197 smallNode227
    smallNode197_agrees smallNode227_agrees (by decide) (by decide) smallRow211_checked

def smallNode191 : Graph := .node 191 smallRow191 smallNode167 smallNode211
set_option maxRecDepth 4096 in
theorem smallNode191_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode191 :=
  Graph.node_agrees _ 191 smallRow191 smallNode167 smallNode211
    smallNode167_agrees smallNode211_agrees (by decide) (by decide) smallRow191_checked

def smallNode239 : Graph := .node 239 smallRow239 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode239_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode239 :=
  Graph.node_agrees _ 239 smallRow239 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow239_checked

def smallNode251 : Graph := .node 251 smallRow251 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode251_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode251 :=
  Graph.node_agrees _ 251 smallRow251 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow251_checked

def smallNode241 : Graph := .node 241 smallRow241 smallNode239 smallNode251
set_option maxRecDepth 4096 in
theorem smallNode241_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode241 :=
  Graph.node_agrees _ 241 smallRow241 smallNode239 smallNode251
    smallNode239_agrees smallNode251_agrees (by decide) (by decide) smallRow241_checked

def smallNode263 : Graph := .node 263 smallRow263 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode263_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode263 :=
  Graph.node_agrees _ 263 smallRow263 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow263_checked

def smallNode271 : Graph := .node 271 smallRow271 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode271_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode271 :=
  Graph.node_agrees _ 271 smallRow271 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow271_checked

def smallNode269 : Graph := .node 269 smallRow269 smallNode263 smallNode271
set_option maxRecDepth 4096 in
theorem smallNode269_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode269 :=
  Graph.node_agrees _ 269 smallRow269 smallNode263 smallNode271
    smallNode263_agrees smallNode271_agrees (by decide) (by decide) smallRow269_checked

def smallNode257 : Graph := .node 257 smallRow257 smallNode241 smallNode269
set_option maxRecDepth 4096 in
theorem smallNode257_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode257 :=
  Graph.node_agrees _ 257 smallRow257 smallNode241 smallNode269
    smallNode241_agrees smallNode269_agrees (by decide) (by decide) smallRow257_checked

def smallNode281 : Graph := .node 281 smallRow281 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode281_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode281 :=
  Graph.node_agrees _ 281 smallRow281 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow281_checked

def smallNode293 : Graph := .node 293 smallRow293 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode293_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode293 :=
  Graph.node_agrees _ 293 smallRow293 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow293_checked

def smallNode283 : Graph := .node 283 smallRow283 smallNode281 smallNode293
set_option maxRecDepth 4096 in
theorem smallNode283_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode283 :=
  Graph.node_agrees _ 283 smallRow283 smallNode281 smallNode293
    smallNode281_agrees smallNode293_agrees (by decide) (by decide) smallRow283_checked

def smallNode311 : Graph := .node 311 smallRow311 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode311_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode311 :=
  Graph.node_agrees _ 311 smallRow311 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow311_checked

def smallNode317 : Graph := .node 317 smallRow317 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode317_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode317 :=
  Graph.node_agrees _ 317 smallRow317 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow317_checked

def smallNode313 : Graph := .node 313 smallRow313 smallNode311 smallNode317
set_option maxRecDepth 4096 in
theorem smallNode313_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode313 :=
  Graph.node_agrees _ 313 smallRow313 smallNode311 smallNode317
    smallNode311_agrees smallNode317_agrees (by decide) (by decide) smallRow313_checked

def smallNode307 : Graph := .node 307 smallRow307 smallNode283 smallNode313
set_option maxRecDepth 4096 in
theorem smallNode307_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode307 :=
  Graph.node_agrees _ 307 smallRow307 smallNode283 smallNode313
    smallNode283_agrees smallNode313_agrees (by decide) (by decide) smallRow307_checked

def smallNode277 : Graph := .node 277 smallRow277 smallNode257 smallNode307
set_option maxRecDepth 4096 in
theorem smallNode277_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode277 :=
  Graph.node_agrees _ 277 smallRow277 smallNode257 smallNode307
    smallNode257_agrees smallNode307_agrees (by decide) (by decide) smallRow277_checked

def smallNode233 : Graph := .node 233 smallRow233 smallNode191 smallNode277
set_option maxRecDepth 4096 in
theorem smallNode233_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode233 :=
  Graph.node_agrees _ 233 smallRow233 smallNode191 smallNode277
    smallNode191_agrees smallNode277_agrees (by decide) (by decide) smallRow233_checked

def smallNode149 : Graph := .node 149 smallRow149 smallNode67 smallNode233
set_option maxRecDepth 4096 in
theorem smallNode149_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode149 :=
  Graph.node_agrees _ 149 smallRow149 smallNode67 smallNode233
    smallNode67_agrees smallNode233_agrees (by decide) (by decide) smallRow149_checked

def smallNode337 : Graph := .node 337 smallRow337 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode337_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode337 :=
  Graph.node_agrees _ 337 smallRow337 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow337_checked

def smallNode349 : Graph := .node 349 smallRow349 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode349_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode349 :=
  Graph.node_agrees _ 349 smallRow349 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow349_checked

def smallNode347 : Graph := .node 347 smallRow347 smallNode337 smallNode349
set_option maxRecDepth 4096 in
theorem smallNode347_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode347 :=
  Graph.node_agrees _ 347 smallRow347 smallNode337 smallNode349
    smallNode337_agrees smallNode349_agrees (by decide) (by decide) smallRow347_checked

def smallNode359 : Graph := .node 359 smallRow359 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode359_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode359 :=
  Graph.node_agrees _ 359 smallRow359 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow359_checked

def smallNode373 : Graph := .node 373 smallRow373 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode373_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode373 :=
  Graph.node_agrees _ 373 smallRow373 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow373_checked

def smallNode367 : Graph := .node 367 smallRow367 smallNode359 smallNode373
set_option maxRecDepth 4096 in
theorem smallNode367_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode367 :=
  Graph.node_agrees _ 367 smallRow367 smallNode359 smallNode373
    smallNode359_agrees smallNode373_agrees (by decide) (by decide) smallRow367_checked

def smallNode353 : Graph := .node 353 smallRow353 smallNode347 smallNode367
set_option maxRecDepth 4096 in
theorem smallNode353_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode353 :=
  Graph.node_agrees _ 353 smallRow353 smallNode347 smallNode367
    smallNode347_agrees smallNode367_agrees (by decide) (by decide) smallRow353_checked

def smallNode383 : Graph := .node 383 smallRow383 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode383_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode383 :=
  Graph.node_agrees _ 383 smallRow383 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow383_checked

def smallNode397 : Graph := .node 397 smallRow397 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode397_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode397 :=
  Graph.node_agrees _ 397 smallRow397 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow397_checked

def smallNode389 : Graph := .node 389 smallRow389 smallNode383 smallNode397
set_option maxRecDepth 4096 in
theorem smallNode389_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode389 :=
  Graph.node_agrees _ 389 smallRow389 smallNode383 smallNode397
    smallNode383_agrees smallNode397_agrees (by decide) (by decide) smallRow389_checked

def smallNode409 : Graph := .node 409 smallRow409 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode409_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode409 :=
  Graph.node_agrees _ 409 smallRow409 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow409_checked

def smallNode421 : Graph := .node 421 smallRow421 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode421_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode421 :=
  Graph.node_agrees _ 421 smallRow421 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow421_checked

def smallNode419 : Graph := .node 419 smallRow419 smallNode409 smallNode421
set_option maxRecDepth 4096 in
theorem smallNode419_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode419 :=
  Graph.node_agrees _ 419 smallRow419 smallNode409 smallNode421
    smallNode409_agrees smallNode421_agrees (by decide) (by decide) smallRow419_checked

def smallNode401 : Graph := .node 401 smallRow401 smallNode389 smallNode419
set_option maxRecDepth 4096 in
theorem smallNode401_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode401 :=
  Graph.node_agrees _ 401 smallRow401 smallNode389 smallNode419
    smallNode389_agrees smallNode419_agrees (by decide) (by decide) smallRow401_checked

def smallNode379 : Graph := .node 379 smallRow379 smallNode353 smallNode401
set_option maxRecDepth 4096 in
theorem smallNode379_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode379 :=
  Graph.node_agrees _ 379 smallRow379 smallNode353 smallNode401
    smallNode353_agrees smallNode401_agrees (by decide) (by decide) smallRow379_checked

def smallNode433 : Graph := .node 433 smallRow433 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode433_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode433 :=
  Graph.node_agrees _ 433 smallRow433 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow433_checked

def smallNode443 : Graph := .node 443 smallRow443 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode443_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode443 :=
  Graph.node_agrees _ 443 smallRow443 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow443_checked

def smallNode439 : Graph := .node 439 smallRow439 smallNode433 smallNode443
set_option maxRecDepth 4096 in
theorem smallNode439_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode439 :=
  Graph.node_agrees _ 439 smallRow439 smallNode433 smallNode443
    smallNode433_agrees smallNode443_agrees (by decide) (by decide) smallRow439_checked

def smallNode457 : Graph := .node 457 smallRow457 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode457_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode457 :=
  Graph.node_agrees _ 457 smallRow457 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow457_checked

def smallNode463 : Graph := .node 463 smallRow463 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode463_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode463 :=
  Graph.node_agrees _ 463 smallRow463 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow463_checked

def smallNode461 : Graph := .node 461 smallRow461 smallNode457 smallNode463
set_option maxRecDepth 4096 in
theorem smallNode461_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode461 :=
  Graph.node_agrees _ 461 smallRow461 smallNode457 smallNode463
    smallNode457_agrees smallNode463_agrees (by decide) (by decide) smallRow461_checked

def smallNode449 : Graph := .node 449 smallRow449 smallNode439 smallNode461
set_option maxRecDepth 4096 in
theorem smallNode449_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode449 :=
  Graph.node_agrees _ 449 smallRow449 smallNode439 smallNode461
    smallNode439_agrees smallNode461_agrees (by decide) (by decide) smallRow449_checked

def smallNode479 : Graph := .node 479 smallRow479 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode479_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode479 :=
  Graph.node_agrees _ 479 smallRow479 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow479_checked

def smallNode491 : Graph := .node 491 smallRow491 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode491_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode491 :=
  Graph.node_agrees _ 491 smallRow491 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow491_checked

def smallNode487 : Graph := .node 487 smallRow487 smallNode479 smallNode491
set_option maxRecDepth 4096 in
theorem smallNode487_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode487 :=
  Graph.node_agrees _ 487 smallRow487 smallNode479 smallNode491
    smallNode479_agrees smallNode491_agrees (by decide) (by decide) smallRow487_checked

def smallNode503 : Graph := .node 503 smallRow503 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode503_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode503 :=
  Graph.node_agrees _ 503 smallRow503 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow503_checked

def smallNode521 : Graph := .node 521 smallRow521 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode521_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode521 :=
  Graph.node_agrees _ 521 smallRow521 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow521_checked

def smallNode509 : Graph := .node 509 smallRow509 smallNode503 smallNode521
set_option maxRecDepth 4096 in
theorem smallNode509_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode509 :=
  Graph.node_agrees _ 509 smallRow509 smallNode503 smallNode521
    smallNode503_agrees smallNode521_agrees (by decide) (by decide) smallRow509_checked

def smallNode499 : Graph := .node 499 smallRow499 smallNode487 smallNode509
set_option maxRecDepth 4096 in
theorem smallNode499_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode499 :=
  Graph.node_agrees _ 499 smallRow499 smallNode487 smallNode509
    smallNode487_agrees smallNode509_agrees (by decide) (by decide) smallRow499_checked

def smallNode467 : Graph := .node 467 smallRow467 smallNode449 smallNode499
set_option maxRecDepth 4096 in
theorem smallNode467_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode467 :=
  Graph.node_agrees _ 467 smallRow467 smallNode449 smallNode499
    smallNode449_agrees smallNode499_agrees (by decide) (by decide) smallRow467_checked

def smallNode431 : Graph := .node 431 smallRow431 smallNode379 smallNode467
set_option maxRecDepth 4096 in
theorem smallNode431_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode431 :=
  Graph.node_agrees _ 431 smallRow431 smallNode379 smallNode467
    smallNode379_agrees smallNode467_agrees (by decide) (by decide) smallRow431_checked

def smallNode541 : Graph := .node 541 smallRow541 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode541_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode541 :=
  Graph.node_agrees _ 541 smallRow541 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow541_checked

def smallNode557 : Graph := .node 557 smallRow557 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode557_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode557 :=
  Graph.node_agrees _ 557 smallRow557 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow557_checked

def smallNode547 : Graph := .node 547 smallRow547 smallNode541 smallNode557
set_option maxRecDepth 4096 in
theorem smallNode547_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode547 :=
  Graph.node_agrees _ 547 smallRow547 smallNode541 smallNode557
    smallNode541_agrees smallNode557_agrees (by decide) (by decide) smallRow547_checked

def smallNode569 : Graph := .node 569 smallRow569 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode569_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode569 :=
  Graph.node_agrees _ 569 smallRow569 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow569_checked

def smallNode577 : Graph := .node 577 smallRow577 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode577_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode577 :=
  Graph.node_agrees _ 577 smallRow577 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow577_checked

def smallNode571 : Graph := .node 571 smallRow571 smallNode569 smallNode577
set_option maxRecDepth 4096 in
theorem smallNode571_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode571 :=
  Graph.node_agrees _ 571 smallRow571 smallNode569 smallNode577
    smallNode569_agrees smallNode577_agrees (by decide) (by decide) smallRow571_checked

def smallNode563 : Graph := .node 563 smallRow563 smallNode547 smallNode571
set_option maxRecDepth 4096 in
theorem smallNode563_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode563 :=
  Graph.node_agrees _ 563 smallRow563 smallNode547 smallNode571
    smallNode547_agrees smallNode571_agrees (by decide) (by decide) smallRow563_checked

def smallNode593 : Graph := .node 593 smallRow593 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode593_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode593 :=
  Graph.node_agrees _ 593 smallRow593 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow593_checked

def smallNode601 : Graph := .node 601 smallRow601 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode601_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode601 :=
  Graph.node_agrees _ 601 smallRow601 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow601_checked

def smallNode599 : Graph := .node 599 smallRow599 smallNode593 smallNode601
set_option maxRecDepth 4096 in
theorem smallNode599_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode599 :=
  Graph.node_agrees _ 599 smallRow599 smallNode593 smallNode601
    smallNode593_agrees smallNode601_agrees (by decide) (by decide) smallRow599_checked

def smallNode613 : Graph := .node 613 smallRow613 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode613_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode613 :=
  Graph.node_agrees _ 613 smallRow613 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow613_checked

def smallNode619 : Graph := .node 619 smallRow619 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode619_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode619 :=
  Graph.node_agrees _ 619 smallRow619 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow619_checked

def smallNode617 : Graph := .node 617 smallRow617 smallNode613 smallNode619
set_option maxRecDepth 4096 in
theorem smallNode617_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode617 :=
  Graph.node_agrees _ 617 smallRow617 smallNode613 smallNode619
    smallNode613_agrees smallNode619_agrees (by decide) (by decide) smallRow617_checked

def smallNode607 : Graph := .node 607 smallRow607 smallNode599 smallNode617
set_option maxRecDepth 4096 in
theorem smallNode607_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode607 :=
  Graph.node_agrees _ 607 smallRow607 smallNode599 smallNode617
    smallNode599_agrees smallNode617_agrees (by decide) (by decide) smallRow607_checked

def smallNode587 : Graph := .node 587 smallRow587 smallNode563 smallNode607
set_option maxRecDepth 4096 in
theorem smallNode587_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode587 :=
  Graph.node_agrees _ 587 smallRow587 smallNode563 smallNode607
    smallNode563_agrees smallNode607_agrees (by decide) (by decide) smallRow587_checked

def smallNode641 : Graph := .node 641 smallRow641 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode641_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode641 :=
  Graph.node_agrees _ 641 smallRow641 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow641_checked

def smallNode647 : Graph := .node 647 smallRow647 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode647_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode647 :=
  Graph.node_agrees _ 647 smallRow647 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow647_checked

def smallNode643 : Graph := .node 643 smallRow643 smallNode641 smallNode647
set_option maxRecDepth 4096 in
theorem smallNode643_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode643 :=
  Graph.node_agrees _ 643 smallRow643 smallNode641 smallNode647
    smallNode641_agrees smallNode647_agrees (by decide) (by decide) smallRow643_checked

def smallNode659 : Graph := .node 659 smallRow659 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode659_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode659 :=
  Graph.node_agrees _ 659 smallRow659 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow659_checked

def smallNode673 : Graph := .node 673 smallRow673 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode673_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode673 :=
  Graph.node_agrees _ 673 smallRow673 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow673_checked

def smallNode661 : Graph := .node 661 smallRow661 smallNode659 smallNode673
set_option maxRecDepth 4096 in
theorem smallNode661_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode661 :=
  Graph.node_agrees _ 661 smallRow661 smallNode659 smallNode673
    smallNode659_agrees smallNode673_agrees (by decide) (by decide) smallRow661_checked

def smallNode653 : Graph := .node 653 smallRow653 smallNode643 smallNode661
set_option maxRecDepth 4096 in
theorem smallNode653_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode653 :=
  Graph.node_agrees _ 653 smallRow653 smallNode643 smallNode661
    smallNode643_agrees smallNode661_agrees (by decide) (by decide) smallRow653_checked

def smallNode683 : Graph := .node 683 smallRow683 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode683_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode683 :=
  Graph.node_agrees _ 683 smallRow683 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow683_checked

def smallNode701 : Graph := .node 701 smallRow701 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode701_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode701 :=
  Graph.node_agrees _ 701 smallRow701 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow701_checked

def smallNode691 : Graph := .node 691 smallRow691 smallNode683 smallNode701
set_option maxRecDepth 4096 in
theorem smallNode691_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode691 :=
  Graph.node_agrees _ 691 smallRow691 smallNode683 smallNode701
    smallNode683_agrees smallNode701_agrees (by decide) (by decide) smallRow691_checked

def smallNode719 : Graph := .node 719 smallRow719 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode719_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode719 :=
  Graph.node_agrees _ 719 smallRow719 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow719_checked

def smallNode733 : Graph := .node 733 smallRow733 smallEmpty smallEmpty
set_option maxRecDepth 4096 in
theorem smallNode733_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode733 :=
  Graph.node_agrees _ 733 smallRow733 smallEmpty smallEmpty
    smallEmpty_agrees smallEmpty_agrees (by decide) (by decide) smallRow733_checked

def smallNode727 : Graph := .node 727 smallRow727 smallNode719 smallNode733
set_option maxRecDepth 4096 in
theorem smallNode727_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode727 :=
  Graph.node_agrees _ 727 smallRow727 smallNode719 smallNode733
    smallNode719_agrees smallNode733_agrees (by decide) (by decide) smallRow727_checked

def smallNode709 : Graph := .node 709 smallRow709 smallNode691 smallNode727
set_option maxRecDepth 4096 in
theorem smallNode709_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode709 :=
  Graph.node_agrees _ 709 smallRow709 smallNode691 smallNode727
    smallNode691_agrees smallNode727_agrees (by decide) (by decide) smallRow709_checked

def smallNode677 : Graph := .node 677 smallRow677 smallNode653 smallNode709
set_option maxRecDepth 4096 in
theorem smallNode677_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode677 :=
  Graph.node_agrees _ 677 smallRow677 smallNode653 smallNode709
    smallNode653_agrees smallNode709_agrees (by decide) (by decide) smallRow677_checked

def smallNode631 : Graph := .node 631 smallRow631 smallNode587 smallNode677
set_option maxRecDepth 4096 in
theorem smallNode631_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode631 :=
  Graph.node_agrees _ 631 smallRow631 smallNode587 smallNode677
    smallNode587_agrees smallNode677_agrees (by decide) (by decide) smallRow631_checked

def smallNode523 : Graph := .node 523 smallRow523 smallNode431 smallNode631
set_option maxRecDepth 4096 in
theorem smallNode523_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode523 :=
  Graph.node_agrees _ 523 smallRow523 smallNode431 smallNode631
    smallNode431_agrees smallNode631_agrees (by decide) (by decide) smallRow523_checked

def smallNode331 : Graph := .node 331 smallRow331 smallNode149 smallNode523
set_option maxRecDepth 4096 in
theorem smallNode331_agrees : Graph.Agrees PrimePairVertices.smallTree smallNode331 :=
  Graph.node_agrees _ 331 smallRow331 smallNode149 smallNode523
    smallNode149_agrees smallNode523_agrees (by decide) (by decide) smallRow331_checked

def smallPrefix : Graph := smallNode331
theorem smallPrefix_agrees : Graph.Agrees PrimePairVertices.smallTree smallPrefix := smallNode331_agrees
set_option maxRecDepth 4096 in
theorem smallPrefix_vertices : smallPrefix.vertices = PrimePairVertices.small.take 128 := rfl

theorem smallPrefix_vertices_subset (p : Nat) (member : p ∈ smallPrefix.vertices) : p ∈ PrimePairVertices.small := by
  rw [smallPrefix_vertices] at member
  exact List.mem_of_mem_take member

end ProjectEulerSolutions.PrimePairCache
