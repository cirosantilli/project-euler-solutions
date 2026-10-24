import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk040

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_372258437 : CertifiedPrime :=
  certify 372258437 (2) [p_2, p_2, p_11, p_11, p_877, p_877] (by decide)

def p_372644417 : CertifiedPrime :=
  certify 372644417 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_127, p_127] (by decide)

def p_372702331 : CertifiedPrime :=
  certify 372702331 (7) [p_2, p_3, p_3, p_3, p_5, p_7, p_7, p_11, p_13, p_197] (by decide)

def p_373127173 : CertifiedPrime :=
  certify 373127173 (2) [p_2, p_2, p_3, p_11, p_47, p_137, p_439] (by decide)

def p_373243081 : CertifiedPrime :=
  certify 373243081 (17) [p_2, p_2, p_2, p_3, p_5, p_7, p_23, p_19319] (by decide)

def p_373416977 : CertifiedPrime :=
  certify 373416977 (3) [p_2, p_2, p_2, p_2, p_4831, p_4831] (by decide)

def p_373745557 : CertifiedPrime :=
  certify 373745557 (5) [p_2, p_2, p_3, p_3, p_3, p_179, p_19333] (by decide)

def p_373803557 : CertifiedPrime :=
  certify 373803557 (2) [p_2, p_2, p_7, p_7, p_1381, p_1381] (by decide)

def p_373861561 : CertifiedPrime :=
  certify 373861561 (7) [p_2, p_2, p_2, p_3, p_5, p_1289, p_2417] (by decide)

def p_374035601 : CertifiedPrime :=
  certify 374035601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_967, p_967] (by decide)

def p_374635381 : CertifiedPrime :=
  certify 374635381 (10) [p_2, p_2, p_3, p_5, p_7, p_7, p_79, p_1613] (by decide)

def p_374964497 : CertifiedPrime :=
  certify 374964497 (3) [p_2, p_2, p_2, p_2, p_47, p_47, p_103, p_103] (by decide)

def p_375196901 : CertifiedPrime :=
  certify 375196901 (10) [p_2, p_2, p_5, p_5, p_13, p_13, p_149, p_149] (by decide)

def p_375584401 : CertifiedPrime :=
  certify 375584401 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_17, p_17, p_19, p_19] (by decide)

def p_375681307 : CertifiedPrime :=
  certify 375681307 (2) [p_2, p_3, p_7, p_11, p_13, p_71, p_881] (by decide)

def p_375913933 : CertifiedPrime :=
  certify 375913933 (2) [p_2, p_2, p_3, p_23, p_37, p_131, p_281] (by decide)

def p_376360001 : CertifiedPrime :=
  certify 376360001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_97, p_97] (by decide)

def p_376495813 : CertifiedPrime :=
  certify 376495813 (15) [p_2, p_2, p_3, p_3, p_7, p_7, p_11, p_19403] (by decide)

def p_376651057 : CertifiedPrime :=
  certify 376651057 (5) [p_2, p_2, p_2, p_2, p_3, p_1213, p_6469] (by decide)

def p_376981057 : CertifiedPrime :=
  certify 376981057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_809, p_809] (by decide)

def p_377000473 : CertifiedPrime :=
  certify 377000473 (5) [p_2, p_2, p_2, p_3, p_809, p_19417] (by decide)

def p_377136401 : CertifiedPrime :=
  certify 377136401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_971, p_971] (by decide)

def p_377466613 : CertifiedPrime :=
  certify 377466613 (6) [p_2, p_2, p_3, p_1619, p_19429] (by decide)

def p_377544331 : CertifiedPrime :=
  certify 377544331 (2) [p_2, p_3, p_3, p_5, p_17, p_29, p_67, p_127] (by decide)

def p_377680357 : CertifiedPrime :=
  certify 377680357 (2) [p_2, p_2, p_3, p_3, p_41, p_41, p_79, p_79] (by decide)

def p_377816407 : CertifiedPrime :=
  certify 377816407 (3) [p_2, p_3, p_11, p_19, p_31, p_9719] (by decide)

def p_377894161 : CertifiedPrime :=
  certify 377894161 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_7, p_2777] (by decide)

def p_378146917 : CertifiedPrime :=
  certify 378146917 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_463, p_463] (by decide)

def p_378399757 : CertifiedPrime :=
  certify 378399757 (5) [p_2, p_2, p_3, p_7, p_7, p_397, p_1621] (by decide)

def p_378458117 : CertifiedPrime :=
  certify 378458117 (2) [p_2, p_2, p_71, p_71, p_137, p_137] (by decide)

def p_378477571 : CertifiedPrime :=
  certify 378477571 (3) [p_2, p_3, p_5, p_71, p_137, p_1297] (by decide)

def p_378633223 : CertifiedPrime :=
  certify 378633223 (19) [p_2, p_3, p_3, p_11, p_23, p_29, p_47, p_61] (by decide)

def p_378711061 : CertifiedPrime :=
  certify 378711061 (18) [p_2, p_2, p_3, p_5, p_7, p_13, p_139, p_499] (by decide)

def p_378944623 : CertifiedPrime :=
  certify 378944623 (6) [p_2, p_3, p_3, p_3, p_7, p_103, p_9733] (by decide)

def p_379295101 : CertifiedPrime :=
  certify 379295101 (34) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_41, p_541] (by decide)

def p_379450921 : CertifiedPrime :=
  certify 379450921 (14) [p_2, p_2, p_2, p_3, p_5, p_43, p_151, p_487] (by decide)

def p_379470401 : CertifiedPrime :=
  certify 379470401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_487, p_487] (by decide)

def p_379528843 : CertifiedPrime :=
  certify 379528843 (3) [p_2, p_3, p_7, p_11, p_11, p_17, p_23, p_191] (by decide)

def p_379860101 : CertifiedPrime :=
  certify 379860101 (3) [p_2, p_2, p_5, p_5, p_1949, p_1949] (by decide)

def p_380016037 : CertifiedPrime :=
  certify 380016037 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_19, p_19, p_19, p_19] (by decide)

def p_380035531 : CertifiedPrime :=
  certify 380035531 (7) [p_2, p_3, p_3, p_3, p_5, p_7, p_19, p_19, p_557] (by decide)

def p_380269501 : CertifiedPrime :=
  certify 380269501 (6) [p_2, p_2, p_3, p_5, p_5, p_5, p_13, p_19501] (by decide)

def p_380484037 : CertifiedPrime :=
  certify 380484037 (2) [p_2, p_2, p_3, p_3, p_3251, p_3251] (by decide)

def p_380581573 : CertifiedPrime :=
  certify 380581573 (2) [p_2, p_2, p_3, p_7, p_929, p_4877] (by decide)

def p_380640101 : CertifiedPrime :=
  certify 380640101 (3) [p_2, p_2, p_5, p_5, p_1951, p_1951] (by decide)

def p_380737657 : CertifiedPrime :=
  certify 380737657 (5) [p_2, p_2, p_2, p_3, p_3, p_13, p_19, p_79, p_271] (by decide)

def p_381088963 : CertifiedPrime :=
  certify 381088963 (2) [p_2, p_3, p_3, p_3, p_3, p_43, p_227, p_241] (by decide)

def p_381186577 : CertifiedPrime :=
  certify 381186577 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1627, p_1627] (by decide)

def p_381440431 : CertifiedPrime :=
  certify 381440431 (11) [p_2, p_3, p_3, p_5, p_7, p_31, p_19531] (by decide)

def p_381655297 : CertifiedPrime :=
  certify 381655297 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11, p_37, p_37] (by decide)

def p_381987481 : CertifiedPrime :=
  certify 381987481 (11) [p_2, p_2, p_2, p_3, p_5, p_7, p_349, p_1303] (by decide)

def p_382261153 : CertifiedPrime :=
  certify 382261153 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_7, p_13, p_19, p_47] (by decide)

def p_382378471 : CertifiedPrime :=
  certify 382378471 (3) [p_2, p_3, p_5, p_3259, p_3911] (by decide)

def p_382437137 : CertifiedPrime :=
  certify 382437137 (3) [p_2, p_2, p_2, p_2, p_4889, p_4889] (by decide)

def p_383200201 : CertifiedPrime :=
  certify 383200201 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_5, p_29, p_2447] (by decide)

def p_383219777 : CertifiedPrime :=
  certify 383219777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2447, p_2447] (by decide)

def p_383748511 : CertifiedPrime :=
  certify 383748511 (3) [p_2, p_3, p_5, p_19, p_653, p_1031] (by decide)

def p_383866057 : CertifiedPrime :=
  certify 383866057 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_31, p_79, p_311] (by decide)

def p_384160001 : CertifiedPrime :=
  certify 384160001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_7, p_7] (by decide)

def p_384571711 : CertifiedPrime :=
  certify 384571711 (6) [p_2, p_3, p_3, p_5, p_37, p_53, p_2179] (by decide)

def p_384944401 : CertifiedPrime :=
  certify 384944401 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_109, p_109] (by decide)

def p_385042507 : CertifiedPrime :=
  certify 385042507 (2) [p_2, p_3, p_31, p_211, p_9811] (by decide)

def p_385179877 : CertifiedPrime :=
  certify 385179877 (2) [p_2, p_2, p_3, p_3, p_3271, p_3271] (by decide)

def p_385395793 : CertifiedPrime :=
  certify 385395793 (7) [p_2, p_2, p_2, p_2, p_3, p_67, p_293, p_409] (by decide)

def p_385493957 : CertifiedPrime :=
  certify 385493957 (2) [p_2, p_2, p_9817, p_9817] (by decide)

def p_385572497 : CertifiedPrime :=
  certify 385572497 (3) [p_2, p_2, p_2, p_2, p_4909, p_4909] (by decide)

def p_386102851 : CertifiedPrime :=
  certify 386102851 (15) [p_2, p_3, p_5, p_5, p_7, p_7, p_131, p_401] (by decide)

def p_386142151 : CertifiedPrime :=
  certify 386142151 (3) [p_2, p_3, p_5, p_5, p_43, p_131, p_457] (by decide)

def p_386574583 : CertifiedPrime :=
  certify 386574583 (3) [p_2, p_3, p_29, p_113, p_19661] (by decide)

def p_386751557 : CertifiedPrime :=
  certify 386751557 (2) [p_2, p_2, p_9833, p_9833] (by decide)

def p_386928571 : CertifiedPrime :=
  certify 386928571 (2) [p_2, p_3, p_5, p_7, p_79, p_83, p_281] (by decide)

def p_386967913 : CertifiedPrime :=
  certify 386967913 (5) [p_2, p_2, p_2, p_3, p_79, p_83, p_2459] (by decide)

def p_387204007 : CertifiedPrime :=
  certify 387204007 (5) [p_2, p_3, p_7, p_937, p_9839] (by decide)

def p_387794557 : CertifiedPrime :=
  certify 387794557 (5) [p_2, p_2, p_3, p_3, p_47, p_419, p_547] (by decide)

def p_388227913 : CertifiedPrime :=
  certify 388227913 (5) [p_2, p_2, p_2, p_3, p_17, p_19, p_61, p_821] (by decide)

def p_388326437 : CertifiedPrime :=
  certify 388326437 (2) [p_2, p_2, p_59, p_59, p_167, p_167] (by decide)

def p_388622083 : CertifiedPrime :=
  certify 388622083 (2) [p_2, p_3, p_6571, p_9857] (by decide)

def p_388977007 : CertifiedPrime :=
  certify 388977007 (3) [p_2, p_3, p_11, p_11, p_19, p_163, p_173] (by decide)

def p_389095351 : CertifiedPrime :=
  certify 389095351 (7) [p_2, p_3, p_5, p_5, p_7, p_263, p_1409] (by decide)

def p_389509697 : CertifiedPrime :=
  certify 389509697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2467, p_2467] (by decide)

def p_389529433 : CertifiedPrime :=
  certify 389529433 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_17, p_43, p_2467] (by decide)

def p_389667601 : CertifiedPrime :=
  certify 389667601 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_47, p_47] (by decide)

def p_389884771 : CertifiedPrime :=
  certify 389884771 (10) [p_2, p_3, p_3, p_5, p_11, p_359, p_1097] (by decide)

def p_390358807 : CertifiedPrime :=
  certify 390358807 (3) [p_2, p_3, p_23, p_37, p_89, p_859] (by decide)

def p_390477361 : CertifiedPrime :=
  certify 390477361 (41) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_13, p_19, p_941] (by decide)

def p_390516883 : CertifiedPrime :=
  certify 390516883 (2) [p_2, p_3, p_7, p_41, p_241, p_941] (by decide)

def p_390635461 : CertifiedPrime :=
  certify 390635461 (7) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_59, p_61, p_67] (by decide)

def p_391070401 : CertifiedPrime :=
  certify 391070401 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_103, p_113] (by decide)

def p_391248401 : CertifiedPrime :=
  certify 391248401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_23, p_23, p_43, p_43] (by decide)

def p_391545157 : CertifiedPrime :=
  certify 391545157 (5) [p_2, p_2, p_3, p_17, p_47, p_97, p_421] (by decide)

def p_391881617 : CertifiedPrime :=
  certify 391881617 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_7, p_7, p_101, p_101] (by decide)

def p_391941007 : CertifiedPrime :=
  certify 391941007 (3) [p_2, p_3, p_19, p_521, p_6599] (by decide)

def p_392040001 : CertifiedPrime :=
  certify 392040001 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_11, p_11] (by decide)

def p_392772943 : CertifiedPrime :=
  certify 392772943 (21) [p_2, p_3, p_3, p_3, p_367, p_19819] (by decide)

def p_393248731 : CertifiedPrime :=
  certify 393248731 (12) [p_2, p_3, p_5, p_7, p_661, p_2833] (by decide)

def p_393486733 : CertifiedPrime :=
  certify 393486733 (5) [p_2, p_2, p_3, p_3, p_19, p_29, p_83, p_239] (by decide)

def p_393784337 : CertifiedPrime :=
  certify 393784337 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_11, p_11, p_41, p_41] (by decide)

def p_394042351 : CertifiedPrime :=
  certify 394042351 (3) [p_2, p_3, p_5, p_5, p_13, p_397, p_509] (by decide)

def p_394082053 : CertifiedPrime :=
  certify 394082053 (2) [p_2, p_2, p_3, p_7, p_13, p_509, p_709] (by decide)

def p_394816901 : CertifiedPrime :=
  certify 394816901 (3) [p_2, p_2, p_5, p_5, p_1987, p_1987] (by decide)

def p_394975877 : CertifiedPrime :=
  certify 394975877 (3) [p_2, p_2, p_19, p_19, p_523, p_523] (by decide)

def p_394995751 : CertifiedPrime :=
  certify 394995751 (3) [p_2, p_3, p_5, p_5, p_5, p_19, p_53, p_523] (by decide)

def p_395154763 : CertifiedPrime :=
  certify 395154763 (3) [p_2, p_3, p_103, p_193, p_3313] (by decide)

def p_395850817 : CertifiedPrime :=
  certify 395850817 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_829, p_829] (by decide)

def p_396169217 : CertifiedPrime :=
  certify 396169217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_311, p_311] (by decide)

def p_396348373 : CertifiedPrime :=
  certify 396348373 (19) [p_2, p_2, p_3, p_3, p_7, p_43, p_79, p_463] (by decide)

def p_396467833 : CertifiedPrime :=
  certify 396467833 (20) [p_2, p_2, p_2, p_3, p_19, p_131, p_6637] (by decide)

def p_396587311 : CertifiedPrime :=
  certify 396587311 (3) [p_2, p_3, p_5, p_7, p_569, p_3319] (by decide)

def p_396647057 : CertifiedPrime :=
  certify 396647057 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_383, p_383] (by decide)

def p_397045477 : CertifiedPrime :=
  certify 397045477 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_41, p_41] (by decide)

def p_397204901 : CertifiedPrime :=
  certify 397204901 (2) [p_2, p_2, p_5, p_5, p_1993, p_1993] (by decide)

def p_397743193 : CertifiedPrime :=
  certify 397743193 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_11, p_37, p_277] (by decide)

def p_397763137 : CertifiedPrime :=
  certify 397763137 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_277, p_277] (by decide)

def p_397842917 : CertifiedPrime :=
  certify 397842917 (2) [p_2, p_2, p_9973, p_9973] (by decide)

def p_398022451 : CertifiedPrime :=
  certify 398022451 (3) [p_2, p_3, p_5, p_5, p_7, p_19, p_71, p_281] (by decide)

def p_398221981 : CertifiedPrime :=
  certify 398221981 (7) [p_2, p_2, p_3, p_5, p_13, p_307, p_1663] (by decide)

def p_398261893 : CertifiedPrime :=
  certify 398261893 (5) [p_2, p_2, p_3, p_7, p_1663, p_2851] (by decide)

def p_398621191 : CertifiedPrime :=
  certify 398621191 (3) [p_2, p_3, p_5, p_11, p_11, p_11, p_67, p_149] (by decide)

def p_398641157 : CertifiedPrime :=
  certify 398641157 (2) [p_2, p_2, p_67, p_67, p_149, p_149] (by decide)

def p_398701057 : CertifiedPrime :=
  certify 398701057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_13, p_41, p_487] (by decide)

def p_398820871 : CertifiedPrime :=
  certify 398820871 (6) [p_2, p_3, p_3, p_5, p_7, p_317, p_1997] (by decide)

def p_399180421 : CertifiedPrime :=
  certify 399180421 (23) [p_2, p_2, p_3, p_3, p_3, p_5, p_37, p_19979] (by decide)

def p_399340273 : CertifiedPrime :=
  certify 399340273 (10) [p_2, p_2, p_2, p_2, p_3, p_1249, p_6661] (by decide)

def p_399360257 : CertifiedPrime :=
  certify 399360257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1249, p_1249] (by decide)

def p_399460183 : CertifiedPrime :=
  certify 399460183 (3) [p_2, p_3, p_11, p_23, p_79, p_3331] (by decide)

def p_399600101 : CertifiedPrime :=
  certify 399600101 (2) [p_2, p_2, p_5, p_5, p_1999, p_1999] (by decide)

def p_399660073 : CertifiedPrime :=
  certify 399660073 (13) [p_2, p_2, p_2, p_3, p_7, p_7, p_17, p_19991] (by decide)

def p_399760037 : CertifiedPrime :=
  certify 399760037 (2) [p_2, p_2, p_13, p_13, p_769, p_769] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
