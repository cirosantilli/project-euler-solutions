import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk036

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_274051471 : CertifiedPrime :=
  certify 274051471 (6) [p_2, p_3, p_5, p_7, p_11, p_31, p_43, p_89] (by decide)

def p_274349533 : CertifiedPrime :=
  certify 274349533 (2) [p_2, p_2, p_3, p_41, p_101, p_5521] (by decide)

def p_274515193 : CertifiedPrime :=
  certify 274515193 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_19, p_109, p_263] (by decide)

def p_274714051 : CertifiedPrime :=
  certify 274714051 (2) [p_2, p_3, p_5, p_5, p_13, p_17, p_8287] (by decide)

def p_275211511 : CertifiedPrime :=
  certify 275211511 (12) [p_2, p_3, p_5, p_7, p_53, p_79, p_313] (by decide)

def p_275228101 : CertifiedPrime :=
  certify 275228101 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_79, p_79] (by decide)

def p_275344243 : CertifiedPrime :=
  certify 275344243 (2) [p_2, p_3, p_5531, p_8297] (by decide)

def p_275443813 : CertifiedPrime :=
  certify 275443813 (5) [p_2, p_2, p_3, p_3, p_7, p_461, p_2371] (by decide)

def p_275809057 : CertifiedPrime :=
  certify 275809057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_173, p_16607] (by decide)

def p_275941933 : CertifiedPrime :=
  certify 275941933 (2) [p_2, p_2, p_3, p_7, p_7, p_113, p_4153] (by decide)

def p_276041611 : CertifiedPrime :=
  certify 276041611 (10) [p_2, p_3, p_3, p_5, p_13, p_71, p_3323] (by decide)

def p_276224401 : CertifiedPrime :=
  certify 276224401 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_277, p_277] (by decide)

def p_276407251 : CertifiedPrime :=
  certify 276407251 (3) [p_2, p_3, p_5, p_5, p_5, p_7, p_17, p_19, p_163] (by decide)

def p_276507013 : CertifiedPrime :=
  certify 276507013 (2) [p_2, p_2, p_3, p_23, p_241, p_4157] (by decide)

def p_276756497 : CertifiedPrime :=
  certify 276756497 (3) [p_2, p_2, p_2, p_2, p_4159, p_4159] (by decide)

def p_276889601 : CertifiedPrime :=
  certify 276889601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13] (by decide)

def p_277222501 : CertifiedPrime :=
  certify 277222501 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_37, p_37] (by decide)

def p_277505623 : CertifiedPrime :=
  certify 277505623 (5) [p_2, p_3, p_3, p_3, p_617, p_8329] (by decide)

def p_277605583 : CertifiedPrime :=
  certify 277605583 (3) [p_2, p_3, p_2777, p_16661] (by decide)

def p_277738891 : CertifiedPrime :=
  certify 277738891 (2) [p_2, p_3, p_5, p_11, p_13, p_101, p_641] (by decide)

def p_277805557 : CertifiedPrime :=
  certify 277805557 (5) [p_2, p_2, p_3, p_3, p_7, p_463, p_2381] (by decide)

def p_278088977 : CertifiedPrime :=
  certify 278088977 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_379, p_379] (by decide)

def p_278205721 : CertifiedPrime :=
  certify 278205721 (7) [p_2, p_2, p_2, p_3, p_5, p_13, p_139, p_1283] (by decide)

def p_278439283 : CertifiedPrime :=
  certify 278439283 (5) [p_2, p_3, p_3, p_3, p_3, p_11, p_37, p_41, p_103] (by decide)

def p_278606173 : CertifiedPrime :=
  certify 278606173 (6) [p_2, p_2, p_3, p_13, p_107, p_16691] (by decide)

def p_278839903 : CertifiedPrime :=
  certify 278839903 (3) [p_2, p_3, p_11, p_11, p_23, p_16699] (by decide)

def p_279023617 : CertifiedPrime :=
  certify 279023617 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_29, p_29] (by decide)

def p_279441373 : CertifiedPrime :=
  certify 279441373 (5) [p_2, p_2, p_3, p_7, p_73, p_199, p_229] (by decide)

def p_279558401 : CertifiedPrime :=
  certify 279558401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_11, p_11, p_19, p_19] (by decide)

def p_280026757 : CertifiedPrime :=
  certify 280026757 (2) [p_2, p_2, p_3, p_3, p_2789, p_2789] (by decide)

def p_280227601 : CertifiedPrime :=
  certify 280227601 (19) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_31, p_31] (by decide)

def p_280428517 : CertifiedPrime :=
  certify 280428517 (6) [p_2, p_2, p_3, p_3, p_2791, p_2791] (by decide)

def p_280545751 : CertifiedPrime :=
  certify 280545751 (6) [p_2, p_3, p_3, p_5, p_5, p_5, p_67, p_1861] (by decide)

def p_280612753 : CertifiedPrime :=
  certify 280612753 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_349, p_2393] (by decide)

def p_280646257 : CertifiedPrime :=
  certify 280646257 (5) [p_2, p_2, p_2, p_2, p_3, p_11, p_349, p_1523] (by decide)

def p_280696517 : CertifiedPrime :=
  certify 280696517 (2) [p_2, p_2, p_8377, p_8377] (by decide)

def p_280713271 : CertifiedPrime :=
  certify 280713271 (3) [p_2, p_3, p_5, p_1117, p_8377] (by decide)

def p_280847323 : CertifiedPrime :=
  certify 280847323 (11) [p_2, p_3, p_3, p_7, p_7, p_19, p_16759] (by decide)

def p_281232901 : CertifiedPrime :=
  certify 281232901 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13, p_43, p_43] (by decide)

def p_281249671 : CertifiedPrime :=
  certify 281249671 (3) [p_2, p_3, p_5, p_13, p_31, p_43, p_541] (by decide)

def p_281316757 : CertifiedPrime :=
  certify 281316757 (2) [p_2, p_2, p_3, p_7, p_599, p_5591] (by decide)

def p_281568401 : CertifiedPrime :=
  certify 281568401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_839, p_839] (by decide)

def p_281702657 : CertifiedPrime :=
  certify 281702657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1049, p_1049] (by decide)

def p_281904101 : CertifiedPrime :=
  certify 281904101 (2) [p_2, p_2, p_5, p_5, p_23, p_23, p_73, p_73] (by decide)

def p_282038437 : CertifiedPrime :=
  certify 282038437 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_311, p_311] (by decide)

def p_282761041 : CertifiedPrime :=
  certify 282761041 (14) [p_2, p_2, p_2, p_2, p_3, p_5, p_19, p_59, p_1051] (by decide)

def p_282777857 : CertifiedPrime :=
  certify 282777857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1051, p_1051] (by decide)

def p_282828307 : CertifiedPrime :=
  certify 282828307 (3) [p_2, p_3, p_67, p_251, p_2803] (by decide)

def p_283366723 : CertifiedPrime :=
  certify 283366723 (2) [p_2, p_3, p_19, p_31, p_181, p_443] (by decide)

def p_283467733 : CertifiedPrime :=
  certify 283467733 (5) [p_2, p_2, p_3, p_23, p_61, p_113, p_149] (by decide)

def p_283787717 : CertifiedPrime :=
  certify 283787717 (2) [p_2, p_2, p_8423, p_8423] (by decide)

def p_284057317 : CertifiedPrime :=
  certify 284057317 (2) [p_2, p_2, p_3, p_3, p_53, p_53, p_53, p_53] (by decide)

def p_284141593 : CertifiedPrime :=
  certify 284141593 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_43, p_1873] (by decide)

def p_284394497 : CertifiedPrime :=
  certify 284394497 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_17, p_17, p_31, p_31] (by decide)

def p_284478823 : CertifiedPrime :=
  certify 284478823 (5) [p_2, p_3, p_3, p_101, p_167, p_937] (by decide)

def p_284546293 : CertifiedPrime :=
  certify 284546293 (2) [p_2, p_2, p_3, p_4217, p_5623] (by decide)

def p_284596901 : CertifiedPrime :=
  certify 284596901 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_241, p_241] (by decide)

def p_284934401 : CertifiedPrime :=
  certify 284934401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_211, p_211] (by decide)

def p_284951281 : CertifiedPrime :=
  certify 284951281 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_17, p_211, p_331] (by decide)

def p_285086341 : CertifiedPrime :=
  certify 285086341 (10) [p_2, p_2, p_3, p_3, p_5, p_7, p_11, p_67, p_307] (by decide)

def p_285187657 : CertifiedPrime :=
  certify 285187657 (5) [p_2, p_2, p_2, p_3, p_13, p_433, p_2111] (by decide)

def p_285474817 : CertifiedPrime :=
  certify 285474817 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11] (by decide)

def p_285660703 : CertifiedPrime :=
  certify 285660703 (11) [p_2, p_3, p_3, p_3, p_313, p_16901] (by decide)

def p_285745217 : CertifiedPrime :=
  certify 285745217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2113, p_2113] (by decide)

def p_285762121 : CertifiedPrime :=
  certify 285762121 (31) [p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_23, p_2113] (by decide)

def p_285812837 : CertifiedPrime :=
  certify 285812837 (2) [p_2, p_2, p_79, p_79, p_107, p_107] (by decide)

def p_286083397 : CertifiedPrime :=
  certify 286083397 (5) [p_2, p_2, p_3, p_3, p_2819, p_2819] (by decide)

def p_286472551 : CertifiedPrime :=
  certify 286472551 (3) [p_2, p_3, p_5, p_5, p_7, p_13, p_31, p_677] (by decide)

def p_286489477 : CertifiedPrime :=
  certify 286489477 (5) [p_2, p_2, p_3, p_3, p_7, p_7, p_13, p_13, p_31, p_31] (by decide)

def p_286709557 : CertifiedPrime :=
  certify 286709557 (6) [p_2, p_2, p_3, p_7, p_17, p_41, p_59, p_83] (by decide)

def p_287014423 : CertifiedPrime :=
  certify 287014423 (3) [p_2, p_3, p_43, p_197, p_5647] (by decide)

def p_287302501 : CertifiedPrime :=
  certify 287302501 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_113, p_113] (by decide)

def p_287319451 : CertifiedPrime :=
  certify 287319451 (3) [p_2, p_3, p_5, p_5, p_11, p_23, p_67, p_113] (by decide)

def p_287387257 : CertifiedPrime :=
  certify 287387257 (5) [p_2, p_2, p_2, p_3, p_13, p_163, p_5651] (by decide)

def p_287421163 : CertifiedPrime :=
  certify 287421163 (2) [p_2, p_3, p_7, p_7, p_173, p_5651] (by decide)

def p_287438117 : CertifiedPrime :=
  certify 287438117 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_173, p_173] (by decide)

def p_287794261 : CertifiedPrime :=
  certify 287794261 (2) [p_2, p_2, p_3, p_3, p_5, p_13, p_29, p_4241] (by decide)

def p_287845157 : CertifiedPrime :=
  certify 287845157 (2) [p_2, p_2, p_17, p_17, p_499, p_499] (by decide)

def p_288116677 : CertifiedPrime :=
  certify 288116677 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_23, p_23, p_41, p_41] (by decide)

def p_288201553 : CertifiedPrime :=
  certify 288201553 (10) [p_2, p_2, p_2, p_2, p_3, p_1061, p_5659] (by decide)

def p_288439273 : CertifiedPrime :=
  certify 288439273 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_11, p_17, p_37, p_193] (by decide)

def p_288507211 : CertifiedPrime :=
  certify 288507211 (3) [p_2, p_3, p_5, p_19, p_43, p_79, p_149] (by decide)

def p_288745057 : CertifiedPrime :=
  certify 288745057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_59, p_16993] (by decide)

def p_288796037 : CertifiedPrime :=
  certify 288796037 (2) [p_2, p_2, p_29, p_29, p_293, p_293] (by decide)

def p_288847021 : CertifiedPrime :=
  certify 288847021 (6) [p_2, p_2, p_3, p_5, p_7, p_11, p_103, p_607] (by decide)

def p_289000001 : CertifiedPrime :=
  certify 289000001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_17, p_17] (by decide)

def p_289153021 : CertifiedPrime :=
  certify 289153021 (7) [p_2, p_2, p_3, p_5, p_13, p_19, p_109, p_179] (by decide)

def p_289221043 : CertifiedPrime :=
  certify 289221043 (2) [p_2, p_3, p_11, p_773, p_5669] (by decide)

def p_289340101 : CertifiedPrime :=
  certify 289340101 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_7] (by decide)

def p_289527241 : CertifiedPrime :=
  certify 289527241 (14) [p_2, p_2, p_2, p_3, p_5, p_41, p_83, p_709] (by decide)

def p_289629343 : CertifiedPrime :=
  certify 289629343 (5) [p_2, p_3, p_3, p_31, p_61, p_67, p_127] (by decide)

def p_289680401 : CertifiedPrime :=
  certify 289680401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_23, p_23, p_37, p_37] (by decide)

def p_289765507 : CertifiedPrime :=
  certify 289765507 (3) [p_2, p_3, p_29, p_587, p_2837] (by decide)

def p_289867651 : CertifiedPrime :=
  certify 289867651 (2) [p_2, p_3, p_5, p_5, p_227, p_8513] (by decide)

def p_290480893 : CertifiedPrime :=
  certify 290480893 (11) [p_2, p_2, p_3, p_13, p_19, p_23, p_4261] (by decide)

def p_290549071 : CertifiedPrime :=
  certify 290549071 (7) [p_2, p_3, p_3, p_5, p_7, p_487, p_947] (by decide)

def p_290651353 : CertifiedPrime :=
  certify 290651353 (5) [p_2, p_2, p_2, p_3, p_2131, p_5683] (by decide)

def p_291043601 : CertifiedPrime :=
  certify 291043601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_853, p_853] (by decide)

def p_291060661 : CertifiedPrime :=
  certify 291060661 (2) [p_2, p_2, p_3, p_5, p_11, p_11, p_47, p_853] (by decide)

def p_291384901 : CertifiedPrime :=
  certify 291384901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_569, p_569] (by decide)

def p_291931397 : CertifiedPrime :=
  certify 291931397 (2) [p_2, p_2, p_8543, p_8543] (by decide)

def p_292221931 : CertifiedPrime :=
  certify 292221931 (10) [p_2, p_3, p_5, p_7, p_11, p_13, p_37, p_263] (by decide)

def p_292324507 : CertifiedPrime :=
  certify 292324507 (3) [p_2, p_3, p_41, p_83, p_103, p_139] (by decide)

def p_292495507 : CertifiedPrime :=
  certify 292495507 (2) [p_2, p_3, p_17, p_503, p_5701] (by decide)

def p_292940341 : CertifiedPrime :=
  certify 292940341 (2) [p_2, p_2, p_3, p_5, p_7, p_11, p_163, p_389] (by decide)

def p_293214253 : CertifiedPrime :=
  certify 293214253 (5) [p_2, p_2, p_3, p_1427, p_17123] (by decide)

def p_293231377 : CertifiedPrime :=
  certify 293231377 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1427, p_1427] (by decide)

def p_293522557 : CertifiedPrime :=
  certify 293522557 (2) [p_2, p_2, p_3, p_4283, p_5711] (by decide)

def p_293625361 : CertifiedPrime :=
  certify 293625361 (33) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_17, p_23, p_149] (by decide)

def p_293642497 : CertifiedPrime :=
  certify 293642497 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_17, p_17] (by decide)

def p_294122501 : CertifiedPrime :=
  certify 294122501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_7, p_7, p_7, p_7] (by decide)

def p_294482761 : CertifiedPrime :=
  certify 294482761 (7) [p_2, p_2, p_2, p_3, p_5, p_11, p_13, p_131, p_131] (by decide)

def p_294551407 : CertifiedPrime :=
  certify 294551407 (5) [p_2, p_3, p_3, p_1907, p_8581] (by decide)

def p_294602897 : CertifiedPrime :=
  certify 294602897 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_613, p_613] (by decide)

def p_294860413 : CertifiedPrime :=
  certify 294860413 (14) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_11, p_53, p_223] (by decide)

def p_294894757 : CertifiedPrime :=
  certify 294894757 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_53, p_1321] (by decide)

def p_295066507 : CertifiedPrime :=
  certify 295066507 (3) [p_2, p_3, p_7, p_89, p_193, p_409] (by decide)

def p_295289857 : CertifiedPrime :=
  certify 295289857 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_179, p_179] (by decide)

def p_295307041 : CertifiedPrime :=
  certify 295307041 (11) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_179, p_491] (by decide)

def p_295375783 : CertifiedPrime :=
  certify 295375783 (3) [p_2, p_3, p_13, p_17, p_337, p_661] (by decide)

def p_295582057 : CertifiedPrime :=
  certify 295582057 (5) [p_2, p_2, p_2, p_3, p_7, p_11, p_307, p_521] (by decide)

def p_295702417 : CertifiedPrime :=
  certify 295702417 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1433, p_1433] (by decide)

def p_296029231 : CertifiedPrime :=
  certify 296029231 (11) [p_2, p_3, p_5, p_7, p_31, p_37, p_1229] (by decide)

def p_296235733 : CertifiedPrime :=
  certify 296235733 (2) [p_2, p_2, p_3, p_13, p_331, p_5737] (by decide)

def p_296339011 : CertifiedPrime :=
  certify 296339011 (10) [p_2, p_3, p_5, p_11, p_19, p_151, p_313] (by decide)

def p_296390657 : CertifiedPrime :=
  certify 296390657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_269, p_269] (by decide)

def p_296407873 : CertifiedPrime :=
  certify 296407873 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_269, p_1913] (by decide)

def p_297700517 : CertifiedPrime :=
  certify 297700517 (2) [p_2, p_2, p_8627, p_8627] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
