import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk039

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_345867007 : CertifiedPrime :=
  certify 345867007 (3) [p_2, p_3, p_17, p_547, p_6199] (by decide)

def p_345960001 : CertifiedPrime :=
  certify 345960001 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_31, p_31] (by decide)

def p_346332101 : CertifiedPrime :=
  certify 346332101 (2) [p_2, p_2, p_5, p_5, p_1861, p_1861] (by decide)

def p_346499611 : CertifiedPrime :=
  certify 346499611 (2) [p_2, p_3, p_5, p_17, p_41, p_73, p_227] (by decide)

def p_346704401 : CertifiedPrime :=
  certify 346704401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_7, p_7, p_19, p_19] (by decide)

def p_346946503 : CertifiedPrime :=
  certify 346946503 (3) [p_2, p_3, p_7, p_67, p_139, p_887] (by decide)

def p_347281861 : CertifiedPrime :=
  certify 347281861 (6) [p_2, p_2, p_3, p_5, p_1553, p_3727] (by decide)

def p_347542807 : CertifiedPrime :=
  certify 347542807 (3) [p_2, p_3, p_13, p_103, p_181, p_239] (by decide)

def p_347598737 : CertifiedPrime :=
  certify 347598737 (3) [p_2, p_2, p_2, p_2, p_59, p_59, p_79, p_79] (by decide)

def p_347617381 : CertifiedPrime :=
  certify 347617381 (2) [p_2, p_2, p_3, p_5, p_11, p_59, p_79, p_113] (by decide)

def p_347953063 : CertifiedPrime :=
  certify 347953063 (3) [p_2, p_3, p_23, p_811, p_3109] (by decide)

def p_347990371 : CertifiedPrime :=
  certify 347990371 (10) [p_2, p_3, p_5, p_7, p_13, p_41, p_3109] (by decide)

def p_348419557 : CertifiedPrime :=
  certify 348419557 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_17, p_17, p_61, p_61] (by decide)

def p_348662257 : CertifiedPrime :=
  certify 348662257 (5) [p_2, p_2, p_2, p_2, p_3, p_71, p_263, p_389] (by decide)

def p_348718277 : CertifiedPrime :=
  certify 348718277 (2) [p_2, p_2, p_9337, p_9337] (by decide)

def p_349091857 : CertifiedPrime :=
  certify 349091857 (15) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_173, p_173] (by decide)

def p_349465637 : CertifiedPrime :=
  certify 349465637 (2) [p_2, p_2, p_13, p_13, p_719, p_719] (by decide)

def p_349690001 : CertifiedPrime :=
  certify 349690001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_11, p_11, p_17, p_17] (by decide)

def p_349858321 : CertifiedPrime :=
  certify 349858321 (22) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_29, p_43, p_167] (by decide)

def p_349914437 : CertifiedPrime :=
  certify 349914437 (2) [p_2, p_2, p_47, p_47, p_199, p_199] (by decide)

def p_350344807 : CertifiedPrime :=
  certify 350344807 (5) [p_2, p_3, p_7, p_7, p_17, p_191, p_367] (by decide)

def p_350438401 : CertifiedPrime :=
  certify 350438401 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_13, p_13] (by decide)

def p_350869093 : CertifiedPrime :=
  certify 350869093 (6) [p_2, p_2, p_3, p_7, p_223, p_18731] (by decide)

def p_351018961 : CertifiedPrime :=
  certify 351018961 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_1171, p_1249] (by decide)

def p_351187601 : CertifiedPrime :=
  certify 351187601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_937, p_937] (by decide)

def p_351693763 : CertifiedPrime :=
  certify 351693763 (3) [p_2, p_3, p_7, p_19, p_47, p_9377] (by decide)

def p_351881323 : CertifiedPrime :=
  certify 351881323 (2) [p_2, p_3, p_13, p_13, p_37, p_83, p_113] (by decide)

def p_351918841 : CertifiedPrime :=
  certify 351918841 (11) [p_2, p_2, p_2, p_3, p_5, p_7, p_13, p_13, p_37, p_67] (by decide)

def p_351937601 : CertifiedPrime :=
  certify 351937601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_67, p_67] (by decide)

def p_351993883 : CertifiedPrime :=
  certify 351993883 (3) [p_2, p_3, p_53, p_59, p_73, p_257] (by decide)

def p_352331671 : CertifiedPrime :=
  certify 352331671 (3) [p_2, p_3, p_5, p_1877, p_6257] (by decide)

def p_352444303 : CertifiedPrime :=
  certify 352444303 (5) [p_2, p_3, p_3, p_7, p_149, p_18773] (by decide)

def p_352594507 : CertifiedPrime :=
  certify 352594507 (2) [p_2, p_3, p_11, p_41, p_229, p_569] (by decide)

def p_352932583 : CertifiedPrime :=
  certify 352932583 (3) [p_2, p_3, p_31, p_101, p_18787] (by decide)

def p_353214437 : CertifiedPrime :=
  certify 353214437 (2) [p_2, p_2, p_9397, p_9397] (by decide)

def p_353834911 : CertifiedPrime :=
  certify 353834911 (21) [p_2, p_3, p_3, p_5, p_11, p_13, p_19, p_1447] (by decide)

def p_353966597 : CertifiedPrime :=
  certify 353966597 (2) [p_2, p_2, p_23, p_23, p_409, p_409] (by decide)

def p_354568901 : CertifiedPrime :=
  certify 354568901 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_269, p_269] (by decide)

def p_354587731 : CertifiedPrime :=
  certify 354587731 (2) [p_2, p_3, p_5, p_7, p_269, p_6277] (by decide)

def p_354851407 : CertifiedPrime :=
  certify 354851407 (5) [p_2, p_3, p_3, p_7, p_13, p_23, p_9419] (by decide)

def p_355096337 : CertifiedPrime :=
  certify 355096337 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_673, p_673] (by decide)

def p_355379053 : CertifiedPrime :=
  certify 355379053 (6) [p_2, p_2, p_3, p_7, p_1571, p_2693] (by decide)

def p_355473317 : CertifiedPrime :=
  certify 355473317 (2) [p_2, p_2, p_11, p_11, p_857, p_857] (by decide)

def p_356284501 : CertifiedPrime :=
  certify 356284501 (6) [p_2, p_2, p_3, p_5, p_5, p_5, p_11, p_11, p_13, p_151] (by decide)

def p_356888773 : CertifiedPrime :=
  certify 356888773 (2) [p_2, p_2, p_3, p_3, p_2099, p_4723] (by decide)

def p_357002131 : CertifiedPrime :=
  certify 357002131 (10) [p_2, p_3, p_5, p_47, p_67, p_3779] (by decide)

def p_357191101 : CertifiedPrime :=
  certify 357191101 (13) [p_2, p_2, p_3, p_3, p_3, p_5, p_5, p_7, p_18899] (by decide)

def p_357228901 : CertifiedPrime :=
  certify 357228901 (19) [p_2, p_2, p_3, p_3, p_3, p_5, p_5, p_7, p_41, p_461] (by decide)

def p_357455743 : CertifiedPrime :=
  certify 357455743 (5) [p_2, p_3, p_7, p_23, p_37, p_73, p_137] (by decide)

def p_357531373 : CertifiedPrime :=
  certify 357531373 (2) [p_2, p_2, p_3, p_3, p_11, p_29, p_163, p_191] (by decide)

def p_357758311 : CertifiedPrime :=
  certify 357758311 (3) [p_2, p_3, p_5, p_7, p_7, p_13, p_97, p_193] (by decide)

def p_358666783 : CertifiedPrime :=
  certify 358666783 (12) [p_2, p_3, p_17, p_59, p_107, p_557] (by decide)

def p_358723601 : CertifiedPrime :=
  certify 358723601 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_947, p_947] (by decide)

def p_359007757 : CertifiedPrime :=
  certify 359007757 (5) [p_2, p_2, p_3, p_1579, p_18947] (by decide)

def p_359045653 : CertifiedPrime :=
  certify 359045653 (13) [p_2, p_2, p_3, p_7, p_1579, p_2707] (by decide)

def p_359709157 : CertifiedPrime :=
  certify 359709157 (2) [p_2, p_2, p_3, p_3, p_29, p_29, p_109, p_109] (by decide)

def p_359860901 : CertifiedPrime :=
  certify 359860901 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_271, p_271] (by decide)

def p_360183463 : CertifiedPrime :=
  certify 360183463 (3) [p_2, p_3, p_3163, p_18979] (by decide)

def p_360240401 : CertifiedPrime :=
  certify 360240401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13, p_73, p_73] (by decide)

def p_360259381 : CertifiedPrime :=
  certify 360259381 (2) [p_2, p_2, p_3, p_3, p_3, p_5, p_13, p_19, p_37, p_73] (by decide)

def p_360392257 : CertifiedPrime :=
  certify 360392257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_113, p_113] (by decide)

def p_360525157 : CertifiedPrime :=
  certify 360525157 (2) [p_2, p_2, p_3, p_47, p_101, p_6329] (by decide)

def p_360772037 : CertifiedPrime :=
  certify 360772037 (2) [p_2, p_2, p_9497, p_9497] (by decide)

def p_360848017 : CertifiedPrime :=
  certify 360848017 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_1583, p_1583] (by decide)

def p_361000001 : CertifiedPrime :=
  certify 361000001 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_19, p_19] (by decide)

def p_361399111 : CertifiedPrime :=
  certify 361399111 (13) [p_2, p_3, p_5, p_1901, p_6337] (by decide)

def p_361513183 : CertifiedPrime :=
  certify 361513183 (3) [p_2, p_3, p_3169, p_19013] (by decide)

def p_361608257 : CertifiedPrime :=
  certify 361608257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2377, p_2377] (by decide)

def p_361969651 : CertifiedPrime :=
  certify 361969651 (10) [p_2, p_3, p_3, p_5, p_5, p_7, p_151, p_761] (by decide)

def p_362140901 : CertifiedPrime :=
  certify 362140901 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_173, p_173] (by decide)

def p_362236057 : CertifiedPrime :=
  certify 362236057 (5) [p_2, p_2, p_2, p_3, p_7, p_13, p_61, p_2719] (by decide)

def p_362312191 : CertifiedPrime :=
  certify 362312191 (6) [p_2, p_3, p_3, p_3, p_3, p_5, p_31, p_47, p_307] (by decide)

def p_362464483 : CertifiedPrime :=
  certify 362464483 (3) [p_2, p_3, p_19, p_79, p_167, p_241] (by decide)

def p_362750117 : CertifiedPrime :=
  certify 362750117 (2) [p_2, p_2, p_89, p_89, p_107, p_107] (by decide)

def p_362769163 : CertifiedPrime :=
  certify 362769163 (2) [p_2, p_3, p_7, p_89, p_107, p_907] (by decide)

def p_362921551 : CertifiedPrime :=
  certify 362921551 (3) [p_2, p_3, p_5, p_5, p_127, p_19051] (by decide)

def p_362997757 : CertifiedPrime :=
  certify 362997757 (2) [p_2, p_2, p_3, p_3, p_11, p_29, p_73, p_433] (by decide)

def p_363054917 : CertifiedPrime :=
  certify 363054917 (2) [p_2, p_2, p_7, p_7, p_1361, p_1361] (by decide)

def p_363283601 : CertifiedPrime :=
  certify 363283601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_953, p_953] (by decide)

def p_363722113 : CertifiedPrime :=
  certify 363722113 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_149, p_163] (by decide)

def p_363912853 : CertifiedPrime :=
  certify 363912853 (2) [p_2, p_2, p_3, p_19, p_251, p_6359] (by decide)

def p_364141807 : CertifiedPrime :=
  certify 364141807 (3) [p_2, p_3, p_7, p_29, p_47, p_6361] (by decide)

def p_364199057 : CertifiedPrime :=
  certify 364199057 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_367, p_367] (by decide)

def p_364523557 : CertifiedPrime :=
  certify 364523557 (6) [p_2, p_2, p_3, p_37, p_43, p_61, p_313] (by decide)

def p_364810001 : CertifiedPrime :=
  certify 364810001 (7) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_191, p_191] (by decide)

def p_365211211 : CertifiedPrime :=
  certify 365211211 (3) [p_2, p_3, p_5, p_7, p_7, p_13, p_29, p_659] (by decide)

def p_365402341 : CertifiedPrime :=
  certify 365402341 (7) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_59, p_3823] (by decide)

def p_365440573 : CertifiedPrime :=
  certify 365440573 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_59, p_2731] (by decide)

def p_365574401 : CertifiedPrime :=
  certify 365574401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_239, p_239] (by decide)

def p_365727377 : CertifiedPrime :=
  certify 365727377 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_683, p_683] (by decide)

def p_365861257 : CertifiedPrime :=
  certify 365861257 (5) [p_2, p_2, p_2, p_3, p_31, p_617, p_797] (by decide)

def p_365956901 : CertifiedPrime :=
  certify 365956901 (2) [p_2, p_2, p_5, p_5, p_1913, p_1913] (by decide)

def p_366186497 : CertifiedPrime :=
  certify 366186497 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_23, p_23] (by decide)

def p_366205633 : CertifiedPrime :=
  certify 366205633 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_13, p_23, p_6379] (by decide)

def p_366339601 : CertifiedPrime :=
  certify 366339601 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_11, p_29, p_29] (by decide)

def p_366473593 : CertifiedPrime :=
  certify 366473593 (10) [p_2, p_2, p_2, p_3, p_3, p_3, p_709, p_2393] (by decide)

def p_366665053 : CertifiedPrime :=
  certify 366665053 (2) [p_2, p_2, p_3, p_13, p_491, p_4787] (by decide)

def p_366722501 : CertifiedPrime :=
  certify 366722501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_383, p_383] (by decide)

def p_366779953 : CertifiedPrime :=
  certify 366779953 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_11, p_19, p_1741] (by decide)

def p_367278061 : CertifiedPrime :=
  certify 367278061 (6) [p_2, p_2, p_3, p_5, p_1597, p_3833] (by decide)

def p_367335557 : CertifiedPrime :=
  certify 367335557 (2) [p_2, p_2, p_7, p_7, p_37, p_37, p_37, p_37] (by decide)

def p_367584757 : CertifiedPrime :=
  certify 367584757 (2) [p_2, p_2, p_3, p_7, p_11, p_83, p_4793] (by decide)

def p_368198533 : CertifiedPrime :=
  certify 368198533 (7) [p_2, p_2, p_3, p_3, p_13, p_31, p_41, p_619] (by decide)

def p_368409637 : CertifiedPrime :=
  certify 368409637 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_457, p_457] (by decide)

def p_368736007 : CertifiedPrime :=
  certify 368736007 (3) [p_2, p_3, p_37, p_173, p_9601] (by decide)

def p_368889643 : CertifiedPrime :=
  certify 368889643 (5) [p_2, p_3, p_3, p_11, p_97, p_19207] (by decide)

def p_368966473 : CertifiedPrime :=
  certify 368966473 (5) [p_2, p_2, p_2, p_3, p_7, p_7, p_7, p_7, p_19, p_337] (by decide)

def p_369081733 : CertifiedPrime :=
  certify 369081733 (6) [p_2, p_2, p_3, p_1601, p_19211] (by decide)

def p_369466063 : CertifiedPrime :=
  certify 369466063 (3) [p_2, p_3, p_7, p_43, p_149, p_1373] (by decide)

def p_369562177 : CertifiedPrime :=
  certify 369562177 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_89, p_89] (by decide)

def p_369792901 : CertifiedPrime :=
  certify 369792901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_641, p_641] (by decide)

def p_369889057 : CertifiedPrime :=
  certify 369889057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_601, p_2137] (by decide)

def p_370023697 : CertifiedPrime :=
  certify 370023697 (23) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_229, p_229] (by decide)

def p_370273807 : CertifiedPrime :=
  certify 370273807 (5) [p_2, p_3, p_3, p_7, p_1069, p_2749] (by decide)

def p_370350781 : CertifiedPrime :=
  certify 370350781 (2) [p_2, p_2, p_3, p_5, p_17, p_283, p_1283] (by decide)

def p_370504753 : CertifiedPrime :=
  certify 370504753 (13) [p_2, p_2, p_2, p_2, p_3, p_401, p_19249] (by decide)

def p_370620253 : CertifiedPrime :=
  certify 370620253 (2) [p_2, p_2, p_3, p_3, p_3, p_23, p_31, p_4813] (by decide)

def p_370697263 : CertifiedPrime :=
  certify 370697263 (3) [p_2, p_3, p_13, p_1481, p_3209] (by decide)

def p_370716517 : CertifiedPrime :=
  certify 370716517 (2) [p_2, p_2, p_3, p_3, p_3209, p_3209] (by decide)

def p_370966861 : CertifiedPrime :=
  certify 370966861 (7) [p_2, p_2, p_3, p_3, p_5, p_11, p_17, p_103, p_107] (by decide)

def p_371159491 : CertifiedPrime :=
  certify 371159491 (10) [p_2, p_3, p_5, p_13, p_13, p_19, p_3853] (by decide)

def p_371390713 : CertifiedPrime :=
  certify 371390713 (13) [p_2, p_2, p_2, p_3, p_7, p_11, p_73, p_2753] (by decide)

def p_371429257 : CertifiedPrime :=
  certify 371429257 (5) [p_2, p_2, p_2, p_3, p_11, p_73, p_19273] (by decide)

def p_371506351 : CertifiedPrime :=
  certify 371506351 (3) [p_2, p_3, p_5, p_5, p_23, p_257, p_419] (by decide)

def p_371564177 : CertifiedPrime :=
  certify 371564177 (3) [p_2, p_2, p_2, p_2, p_61, p_61, p_79, p_79] (by decide)

def p_371853373 : CertifiedPrime :=
  certify 371853373 (5) [p_2, p_2, p_3, p_11, p_1607, p_1753] (by decide)

def p_372104101 : CertifiedPrime :=
  certify 372104101 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_643, p_643] (by decide)

def p_372239143 : CertifiedPrime :=
  certify 372239143 (3) [p_2, p_3, p_11, p_59, p_109, p_877] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
