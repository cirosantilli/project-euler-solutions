import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk042

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_429898757 : CertifiedPrime :=
  certify 429898757 (2) [p_2, p_2, p_7, p_7, p_1481, p_1481] (by decide)

def p_430458757 : CertifiedPrime :=
  certify 430458757 (6) [p_2, p_2, p_3, p_7, p_13, p_19, p_20747] (by decide)

def p_430562501 : CertifiedPrime :=
  certify 430562501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_83, p_83] (by decide)

def p_430728517 : CertifiedPrime :=
  certify 430728517 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_1153, p_1153] (by decide)

def p_430749271 : CertifiedPrime :=
  certify 430749271 (7) [p_2, p_3, p_3, p_5, p_7, p_593, p_1153] (by decide)

def p_430977601 : CertifiedPrime :=
  certify 430977601 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_173, p_173] (by decide)

def p_430998361 : CertifiedPrime :=
  certify 430998361 (21) [p_2, p_2, p_2, p_3, p_5, p_13, p_173, p_1597] (by decide)

def p_431122933 : CertifiedPrime :=
  certify 431122933 (2) [p_2, p_2, p_3, p_3, p_3, p_29, p_179, p_769] (by decide)

def p_431330593 : CertifiedPrime :=
  certify 431330593 (20) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_23, p_43, p_59] (by decide)

def p_431372131 : CertifiedPrime :=
  certify 431372131 (2) [p_2, p_3, p_5, p_7, p_23, p_31, p_43, p_67] (by decide)

def p_431496757 : CertifiedPrime :=
  certify 431496757 (5) [p_2, p_2, p_3, p_3, p_577, p_20773] (by decide)

def p_431870743 : CertifiedPrime :=
  certify 431870743 (5) [p_2, p_3, p_3, p_2309, p_10391] (by decide)

def p_431995441 : CertifiedPrime :=
  certify 431995441 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_433, p_4157] (by decide)

def p_433222597 : CertifiedPrime :=
  certify 433222597 (2) [p_2, p_2, p_3, p_3, p_3469, p_3469] (by decide)

def p_433638977 : CertifiedPrime :=
  certify 433638977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_137, p_137] (by decide)

def p_433701451 : CertifiedPrime :=
  certify 433701451 (7) [p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_13, p_17, p_89] (by decide)

def p_433826413 : CertifiedPrime :=
  certify 433826413 (2) [p_2, p_2, p_3, p_41, p_53, p_127, p_131] (by decide)

def p_433888901 : CertifiedPrime :=
  certify 433888901 (2) [p_2, p_2, p_5, p_5, p_2083, p_2083] (by decide)

def p_434326441 : CertifiedPrime :=
  certify 434326441 (11) [p_2, p_2, p_2, p_3, p_5, p_521, p_6947] (by decide)

def p_434493181 : CertifiedPrime :=
  certify 434493181 (17) [p_2, p_2, p_3, p_3, p_3, p_5, p_11, p_193, p_379] (by decide)

def p_434576563 : CertifiedPrime :=
  certify 434576563 (2) [p_2, p_3, p_7, p_1489, p_6949] (by decide)

def p_434701651 : CertifiedPrime :=
  certify 434701651 (3) [p_2, p_3, p_5, p_5, p_139, p_20849] (by decide)

def p_435306497 : CertifiedPrime :=
  certify 435306497 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_163, p_163] (by decide)

def p_435452557 : CertifiedPrime :=
  certify 435452557 (5) [p_2, p_2, p_3, p_7, p_11, p_37, p_47, p_271] (by decide)

def p_435995281 : CertifiedPrime :=
  certify 435995281 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_19, p_29, p_157] (by decide)

def p_436141457 : CertifiedPrime :=
  certify 436141457 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_227, p_227] (by decide)

def p_436371211 : CertifiedPrime :=
  certify 436371211 (2) [p_2, p_3, p_3, p_5, p_11, p_211, p_2089] (by decide)

def p_436747303 : CertifiedPrime :=
  certify 436747303 (13) [p_2, p_3, p_3, p_3, p_3, p_3, p_43, p_20899] (by decide)

def p_437709163 : CertifiedPrime :=
  certify 437709163 (3) [p_2, p_3, p_11, p_317, p_20921] (by decide)

def p_438378907 : CertifiedPrime :=
  certify 438378907 (2) [p_2, p_3, p_7, p_19, p_19, p_29, p_997] (by decide)

def p_438504541 : CertifiedPrime :=
  certify 438504541 (7) [p_2, p_2, p_3, p_5, p_43, p_349, p_487] (by decide)

def p_438651137 : CertifiedPrime :=
  certify 438651137 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_11, p_11, p_17, p_17] (by decide)

def p_439070117 : CertifiedPrime :=
  certify 439070117 (2) [p_2, p_2, p_10477, p_10477] (by decide)

def p_439468333 : CertifiedPrime :=
  certify 439468333 (5) [p_2, p_2, p_3, p_1747, p_20963] (by decide)

def p_439594123 : CertifiedPrime :=
  certify 439594123 (2) [p_2, p_3, p_11, p_29, p_241, p_953] (by decide)

def p_439992577 : CertifiedPrime :=
  certify 439992577 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_19, p_19, p_23, p_23] (by decide)

def p_440160401 : CertifiedPrime :=
  certify 440160401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1049, p_1049] (by decide)

def p_440349241 : CertifiedPrime :=
  certify 440349241 (7) [p_2, p_2, p_2, p_3, p_5, p_43, p_61, p_1399] (by decide)

def p_440412197 : CertifiedPrime :=
  certify 440412197 (2) [p_2, p_2, p_7, p_7, p_1499, p_1499] (by decide)

def p_440748037 : CertifiedPrime :=
  certify 440748037 (5) [p_2, p_2, p_3, p_3, p_3499, p_3499] (by decide)

def p_440769031 : CertifiedPrime :=
  certify 440769031 (3) [p_2, p_3, p_5, p_13, p_17, p_19, p_3499] (by decide)

def p_440853013 : CertifiedPrime :=
  certify 440853013 (2) [p_2, p_2, p_3, p_3, p_29, p_181, p_2333] (by decide)

def p_440979001 : CertifiedPrime :=
  certify 440979001 (13) [p_2, p_2, p_2, p_3, p_5, p_5, p_5, p_7, p_11, p_23, p_83] (by decide)

def p_441147013 : CertifiedPrime :=
  certify 441147013 (2) [p_2, p_2, p_3, p_59, p_89, p_7001] (by decide)

def p_441252037 : CertifiedPrime :=
  certify 441252037 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_389, p_389] (by decide)

def p_441609211 : CertifiedPrime :=
  certify 441609211 (2) [p_2, p_3, p_3, p_5, p_7, p_19, p_79, p_467] (by decide)

def p_441840401 : CertifiedPrime :=
  certify 441840401 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_1051, p_1051] (by decide)

def p_442155757 : CertifiedPrime :=
  certify 442155757 (2) [p_2, p_2, p_3, p_7, p_43, p_163, p_751] (by decide)

def p_442260901 : CertifiedPrime :=
  certify 442260901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_701, p_701] (by decide)

def p_442618483 : CertifiedPrime :=
  certify 442618483 (2) [p_2, p_3, p_67, p_157, p_7013] (by decide)

def p_442681601 : CertifiedPrime :=
  certify 442681601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_263, p_263] (by decide)

def p_443270917 : CertifiedPrime :=
  certify 443270917 (2) [p_2, p_2, p_3, p_3, p_11, p_11, p_11, p_11, p_29, p_29] (by decide)

def p_443797423 : CertifiedPrime :=
  certify 443797423 (3) [p_2, p_3, p_3511, p_21067] (by decide)

def p_444008113 : CertifiedPrime :=
  certify 444008113 (11) [p_2, p_2, p_2, p_2, p_3, p_19, p_439, p_1109] (by decide)

def p_444261007 : CertifiedPrime :=
  certify 444261007 (5) [p_2, p_3, p_3, p_7, p_1171, p_3011] (by decide)

def p_444366401 : CertifiedPrime :=
  certify 444366401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_31, p_31] (by decide)

def p_444513973 : CertifiedPrime :=
  certify 444513973 (5) [p_2, p_2, p_3, p_7, p_29, p_251, p_727] (by decide)

def p_444619397 : CertifiedPrime :=
  certify 444619397 (2) [p_2, p_2, p_13, p_13, p_811, p_811] (by decide)

def p_444682657 : CertifiedPrime :=
  certify 444682657 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_11, p_71, p_659] (by decide)

def p_444809191 : CertifiedPrime :=
  certify 444809191 (13) [p_2, p_3, p_5, p_7, p_19, p_23, p_37, p_131] (by decide)

def p_445062313 : CertifiedPrime :=
  certify 445062313 (5) [p_2, p_2, p_2, p_3, p_3, p_17, p_17, p_73, p_293] (by decide)

def p_445188901 : CertifiedPrime :=
  certify 445188901 (2) [p_2, p_2, p_3, p_5, p_5, p_13, p_211, p_541] (by decide)

def p_445695433 : CertifiedPrime :=
  certify 445695433 (5) [p_2, p_2, p_2, p_3, p_7, p_13, p_29, p_31, p_227] (by decide)

def p_445779883 : CertifiedPrime :=
  certify 445779883 (5) [p_2, p_3, p_3, p_3, p_17, p_23, p_43, p_491] (by decide)

def p_446160007 : CertifiedPrime :=
  certify 446160007 (6) [p_2, p_3, p_3, p_59, p_179, p_2347] (by decide)

def p_446286751 : CertifiedPrime :=
  certify 446286751 (3) [p_2, p_3, p_5, p_5, p_5, p_7, p_13, p_13, p_503] (by decide)

def p_446455771 : CertifiedPrime :=
  certify 446455771 (2) [p_2, p_3, p_5, p_2113, p_7043] (by decide)

def p_446582557 : CertifiedPrime :=
  certify 446582557 (5) [p_2, p_2, p_3, p_3, p_7, p_587, p_3019] (by decide)

def p_446645957 : CertifiedPrime :=
  certify 446645957 (2) [p_2, p_2, p_10567, p_10567] (by decide)

def p_446730497 : CertifiedPrime :=
  certify 446730497 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1321, p_1321] (by decide)

def p_446836183 : CertifiedPrime :=
  certify 446836183 (3) [p_2, p_3, p_13, p_271, p_21139] (by decide)

def p_446899601 : CertifiedPrime :=
  certify 446899601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_151, p_151] (by decide)

def p_447047593 : CertifiedPrime :=
  certify 447047593 (5) [p_2, p_2, p_2, p_3, p_881, p_21143] (by decide)

def p_447174463 : CertifiedPrime :=
  certify 447174463 (3) [p_2, p_3, p_7, p_19, p_53, p_97, p_109] (by decide)

def p_447724441 : CertifiedPrime :=
  certify 447724441 (13) [p_2, p_2, p_2, p_3, p_3, p_5, p_23, p_23, p_2351] (by decide)

def p_447936061 : CertifiedPrime :=
  certify 447936061 (6) [p_2, p_2, p_3, p_5, p_11, p_13, p_17, p_37, p_83] (by decide)

def p_448422977 : CertifiedPrime :=
  certify 448422977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2647, p_2647] (by decide)

def p_448592401 : CertifiedPrime :=
  certify 448592401 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_353, p_353] (by decide)

def p_448740673 : CertifiedPrime :=
  certify 448740673 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_23, p_307, p_331] (by decide)

def p_448952533 : CertifiedPrime :=
  certify 448952533 (2) [p_2, p_2, p_3, p_7, p_1009, p_5297] (by decide)

def p_449206831 : CertifiedPrime :=
  certify 449206831 (6) [p_2, p_3, p_3, p_3, p_5, p_157, p_10597] (by decide)

def p_449334007 : CertifiedPrime :=
  certify 449334007 (3) [p_2, p_3, p_11, p_41, p_47, p_3533] (by decide)

def p_449588413 : CertifiedPrime :=
  certify 449588413 (5) [p_2, p_2, p_3, p_3, p_7, p_13, p_19, p_31, p_233] (by decide)

def p_449609617 : CertifiedPrime :=
  certify 449609617 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_19, p_19, p_31, p_31] (by decide)

def p_449630821 : CertifiedPrime :=
  certify 449630821 (17) [p_2, p_2, p_3, p_3, p_5, p_19, p_31, p_4241] (by decide)

def p_449864101 : CertifiedPrime :=
  certify 449864101 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_101, p_101] (by decide)

def p_449885311 : CertifiedPrime :=
  certify 449885311 (3) [p_2, p_3, p_5, p_7, p_101, p_21211] (by decide)

def p_450543077 : CertifiedPrime :=
  certify 450543077 (2) [p_2, p_2, p_10613, p_10613] (by decide)

def p_450967697 : CertifiedPrime :=
  certify 450967697 (3) [p_2, p_2, p_2, p_2, p_5309, p_5309] (by decide)

def p_451031407 : CertifiedPrime :=
  certify 451031407 (3) [p_2, p_3, p_7, p_37, p_41, p_7079] (by decide)

def p_451116361 : CertifiedPrime :=
  certify 451116361 (13) [p_2, p_2, p_2, p_3, p_3, p_5, p_59, p_67, p_317] (by decide)

def p_451413763 : CertifiedPrime :=
  certify 451413763 (3) [p_2, p_3, p_3541, p_21247] (by decide)

def p_451668757 : CertifiedPrime :=
  certify 451668757 (5) [p_2, p_2, p_3, p_7, p_11, p_23, p_53, p_401] (by decide)

def p_451796281 : CertifiedPrime :=
  certify 451796281 (7) [p_2, p_2, p_2, p_3, p_5, p_13, p_109, p_2657] (by decide)

def p_451817537 : CertifiedPrime :=
  certify 451817537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2657, p_2657] (by decide)

def p_452157697 : CertifiedPrime :=
  certify 452157697 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_443, p_443] (by decide)

def p_452242757 : CertifiedPrime :=
  certify 452242757 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_7, p_7, p_31, p_31] (by decide)

def p_452264023 : CertifiedPrime :=
  certify 452264023 (7) [p_2, p_3, p_3, p_7, p_7, p_7, p_17, p_31, p_139] (by decide)

def p_452306557 : CertifiedPrime :=
  certify 452306557 (2) [p_2, p_2, p_3, p_3, p_13, p_17, p_139, p_409] (by decide)

def p_452561803 : CertifiedPrime :=
  certify 452561803 (11) [p_2, p_3, p_7, p_11, p_967, p_1013] (by decide)

def p_453200233 : CertifiedPrime :=
  certify 453200233 (7) [p_2, p_2, p_2, p_3, p_61, p_349, p_887] (by decide)

def p_453434437 : CertifiedPrime :=
  certify 453434437 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_13, p_13, p_13, p_13] (by decide)

def p_453519617 : CertifiedPrime :=
  certify 453519617 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_11, p_11, p_11, p_11] (by decide)

def p_454052173 : CertifiedPrime :=
  certify 454052173 (2) [p_2, p_2, p_3, p_7, p_761, p_7103] (by decide)

def p_454116101 : CertifiedPrime :=
  certify 454116101 (2) [p_2, p_2, p_5, p_5, p_2131, p_2131] (by decide)

def p_454563721 : CertifiedPrime :=
  certify 454563721 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_13, p_23, p_41, p_103] (by decide)

def p_455075557 : CertifiedPrime :=
  certify 455075557 (2) [p_2, p_2, p_3, p_13, p_547, p_5333] (by decide)

def p_455566337 : CertifiedPrime :=
  certify 455566337 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_23, p_23, p_29, p_29] (by decide)

def p_455651717 : CertifiedPrime :=
  certify 455651717 (2) [p_2, p_2, p_13, p_13, p_821, p_821] (by decide)

def p_455822501 : CertifiedPrime :=
  certify 455822501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_61, p_61] (by decide)

def p_455843851 : CertifiedPrime :=
  certify 455843851 (3) [p_2, p_3, p_5, p_5, p_7, p_11, p_61, p_647] (by decide)

def p_456100093 : CertifiedPrime :=
  certify 456100093 (2) [p_2, p_2, p_3, p_3, p_3, p_7, p_19, p_113, p_281] (by decide)

def p_456142807 : CertifiedPrime :=
  certify 456142807 (6) [p_2, p_3, p_3, p_3, p_7, p_59, p_113, p_181] (by decide)

def p_456484591 : CertifiedPrime :=
  certify 456484591 (15) [p_2, p_3, p_3, p_5, p_1187, p_4273] (by decide)

def p_456847877 : CertifiedPrime :=
  certify 456847877 (2) [p_2, p_2, p_10687, p_10687] (by decide)

def p_456869251 : CertifiedPrime :=
  certify 456869251 (2) [p_2, p_3, p_3, p_5, p_5, p_5, p_19, p_10687] (by decide)

def p_456933377 : CertifiedPrime :=
  certify 456933377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_167, p_167] (by decide)

def p_457040263 : CertifiedPrime :=
  certify 457040263 (3) [p_2, p_3, p_7, p_509, p_21379] (by decide)

def p_457275457 : CertifiedPrime :=
  certify 457275457 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_11, p_11] (by decide)

def p_457360997 : CertifiedPrime :=
  certify 457360997 (2) [p_2, p_2, p_17, p_17, p_17, p_17, p_37, p_37] (by decide)

def p_457681843 : CertifiedPrime :=
  certify 457681843 (2) [p_2, p_3, p_3, p_19, p_563, p_2377] (by decide)

def p_457895803 : CertifiedPrime :=
  certify 457895803 (2) [p_2, p_3, p_7, p_13, p_823, p_1019] (by decide)

def p_458388101 : CertifiedPrime :=
  certify 458388101 (2) [p_2, p_2, p_5, p_5, p_2141, p_2141] (by decide)

def p_458537983 : CertifiedPrime :=
  certify 458537983 (3) [p_2, p_3, p_7, p_7, p_19, p_23, p_43, p_83] (by decide)

def p_458966353 : CertifiedPrime :=
  certify 458966353 (7) [p_2, p_2, p_2, p_2, p_3, p_13, p_37, p_103, p_193] (by decide)

def p_459073477 : CertifiedPrime :=
  certify 459073477 (5) [p_2, p_2, p_3, p_3, p_3571, p_3571] (by decide)

def p_459437791 : CertifiedPrime :=
  certify 459437791 (6) [p_2, p_3, p_5, p_7, p_1429, p_1531] (by decide)

def p_459502097 : CertifiedPrime :=
  certify 459502097 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_233, p_233] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
