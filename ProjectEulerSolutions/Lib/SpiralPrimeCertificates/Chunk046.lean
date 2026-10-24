import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk045

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_520022417 : CertifiedPrime :=
  certify 520022417 (3) [p_2, p_2, p_2, p_2, p_5701, p_5701] (by decide)

def p_520136443 : CertifiedPrime :=
  certify 520136443 (5) [p_2, p_3, p_3, p_7, p_181, p_22807] (by decide)

def p_520410157 : CertifiedPrime :=
  certify 520410157 (5) [p_2, p_2, p_3, p_7, p_1901, p_3259] (by decide)

def p_520775221 : CertifiedPrime :=
  certify 520775221 (2) [p_2, p_2, p_3, p_5, p_7, p_163, p_7607] (by decide)

def p_521208901 : CertifiedPrime :=
  certify 521208901 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_761, p_761] (by decide)

def p_521391557 : CertifiedPrime :=
  certify 521391557 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_233, p_233] (by decide)

def p_521779807 : CertifiedPrime :=
  certify 521779807 (5) [p_2, p_3, p_3, p_3, p_3, p_3, p_47, p_53, p_431] (by decide)

def p_522008257 : CertifiedPrime :=
  certify 522008257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_17, p_31, p_67] (by decide)

def p_522122501 : CertifiedPrime :=
  certify 522122501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_457, p_457] (by decide)

def p_522305317 : CertifiedPrime :=
  certify 522305317 (5) [p_2, p_2, p_3, p_3, p_13, p_13, p_293, p_293] (by decide)

def p_522396737 : CertifiedPrime :=
  certify 522396737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2857, p_2857] (by decide)

def p_522602461 : CertifiedPrime :=
  certify 522602461 (11) [p_2, p_2, p_3, p_3, p_5, p_127, p_22861] (by decide)

def p_523380007 : CertifiedPrime :=
  certify 523380007 (5) [p_2, p_3, p_3, p_31, p_41, p_22877] (by decide)

def p_523677457 : CertifiedPrime :=
  certify 523677457 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1907, p_1907] (by decide)

def p_523952101 : CertifiedPrime :=
  certify 523952101 (26) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_109, p_109] (by decide)

def p_523974991 : CertifiedPrime :=
  certify 523974991 (3) [p_2, p_3, p_5, p_7, p_11, p_109, p_2081] (by decide)

def p_524112343 : CertifiedPrime :=
  certify 524112343 (3) [p_2, p_3, p_13, p_587, p_11447] (by decide)

def p_524341303 : CertifiedPrime :=
  certify 524341303 (3) [p_2, p_3, p_17, p_107, p_107, p_449] (by decide)

def p_524410001 : CertifiedPrime :=
  certify 524410001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_229, p_229] (by decide)

def p_524593217 : CertifiedPrime :=
  certify 524593217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_409, p_409] (by decide)

def p_524661931 : CertifiedPrime :=
  certify 524661931 (2) [p_2, p_3, p_3, p_5, p_13, p_509, p_881] (by decide)

def p_524684837 : CertifiedPrime :=
  certify 524684837 (2) [p_2, p_2, p_13, p_13, p_881, p_881] (by decide)

def p_524753557 : CertifiedPrime :=
  certify 524753557 (5) [p_2, p_2, p_3, p_23, p_83, p_22907] (by decide)

def p_525165973 : CertifiedPrime :=
  certify 525165973 (7) [p_2, p_2, p_3, p_17, p_337, p_7639] (by decide)

def p_525211807 : CertifiedPrime :=
  certify 525211807 (3) [p_2, p_3, p_7, p_1637, p_7639] (by decide)

def p_525486853 : CertifiedPrime :=
  certify 525486853 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_11, p_283, p_521] (by decide)

def p_525624403 : CertifiedPrime :=
  certify 525624403 (3) [p_2, p_3, p_101, p_227, p_3821] (by decide)

def p_525784901 : CertifiedPrime :=
  certify 525784901 (2) [p_2, p_2, p_5, p_5, p_2293, p_2293] (by decide)

def p_525853693 : CertifiedPrime :=
  certify 525853693 (11) [p_2, p_2, p_3, p_3, p_7, p_7, p_13, p_23, p_997] (by decide)

def p_526128907 : CertifiedPrime :=
  certify 526128907 (3) [p_2, p_3, p_3823, p_22937] (by decide)

def p_526541863 : CertifiedPrime :=
  certify 526541863 (3) [p_2, p_3, p_7, p_11, p_149, p_7649] (by decide)

def p_527092723 : CertifiedPrime :=
  certify 527092723 (2) [p_2, p_3, p_3, p_13, p_883, p_2551] (by decide)

def p_527161601 : CertifiedPrime :=
  certify 527161601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_41, p_41] (by decide)

def p_527345297 : CertifiedPrime :=
  certify 527345297 (3) [p_2, p_2, p_2, p_2, p_5741, p_5741] (by decide)

def p_527781703 : CertifiedPrime :=
  certify 527781703 (5) [p_2, p_3, p_7, p_547, p_22973] (by decide)

def p_527804677 : CertifiedPrime :=
  certify 527804677 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_547, p_547] (by decide)

def p_528057421 : CertifiedPrime :=
  certify 528057421 (6) [p_2, p_2, p_3, p_5, p_11, p_383, p_2089] (by decide)

def p_528264257 : CertifiedPrime :=
  certify 528264257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_13, p_13, p_17, p_17] (by decide)

def p_528471133 : CertifiedPrime :=
  certify 528471133 (2) [p_2, p_2, p_3, p_7, p_79, p_97, p_821] (by decide)

def p_528793021 : CertifiedPrime :=
  certify 528793021 (6) [p_2, p_2, p_3, p_3, p_5, p_7, p_73, p_5749] (by decide)

def p_529736257 : CertifiedPrime :=
  certify 529736257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_137, p_137] (by decide)

def p_530035507 : CertifiedPrime :=
  certify 530035507 (5) [p_2, p_3, p_3, p_7, p_11, p_13, p_23, p_1279] (by decide)

def p_530311813 : CertifiedPrime :=
  certify 530311813 (6) [p_2, p_2, p_3, p_19, p_101, p_23029] (by decide)

def p_530542123 : CertifiedPrime :=
  certify 530542123 (3) [p_2, p_3, p_11, p_31, p_349, p_743] (by decide)

def p_530657297 : CertifiedPrime :=
  certify 530657297 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_443, p_443] (by decide)

def p_531002893 : CertifiedPrime :=
  certify 531002893 (6) [p_2, p_2, p_3, p_7, p_823, p_7681] (by decide)

def p_531118117 : CertifiedPrime :=
  certify 531118117 (6) [p_2, p_2, p_3, p_3, p_23, p_23, p_167, p_167] (by decide)

def p_531279451 : CertifiedPrime :=
  certify 531279451 (2) [p_2, p_3, p_3, p_5, p_5, p_13, p_197, p_461] (by decide)

def p_531648307 : CertifiedPrime :=
  certify 531648307 (17) [p_2, p_3, p_3, p_3, p_7, p_61, p_23057] (by decide)

def p_531763601 : CertifiedPrime :=
  certify 531763601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1153, p_1153] (by decide)

def p_532663321 : CertifiedPrime :=
  certify 532663321 (13) [p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_157, p_577] (by decide)

def p_532871057 : CertifiedPrime :=
  certify 532871057 (3) [p_2, p_2, p_2, p_2, p_29, p_29, p_199, p_199] (by decide)

def p_532963397 : CertifiedPrime :=
  certify 532963397 (2) [p_2, p_2, p_7, p_7, p_17, p_17, p_97, p_97] (by decide)

def p_533586901 : CertifiedPrime :=
  certify 533586901 (19) [p_2, p_2, p_3, p_5, p_5, p_7, p_11, p_23099] (by decide)

def p_533610001 : CertifiedPrime :=
  certify 533610001 (34) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_7, p_7, p_11, p_11] (by decide)

def p_534002773 : CertifiedPrime :=
  certify 534002773 (2) [p_2, p_2, p_3, p_53, p_109, p_7703] (by decide)

def p_534072101 : CertifiedPrime :=
  certify 534072101 (2) [p_2, p_2, p_5, p_5, p_2311, p_2311] (by decide)

def p_534187657 : CertifiedPrime :=
  certify 534187657 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_29, p_107, p_797] (by decide)

def p_534534401 : CertifiedPrime :=
  certify 534534401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_17, p_17] (by decide)

def p_534835003 : CertifiedPrime :=
  certify 534835003 (2) [p_2, p_3, p_13, p_31, p_373, p_593] (by decide)

def p_535181957 : CertifiedPrime :=
  certify 535181957 (2) [p_2, p_2, p_43, p_43, p_269, p_269] (by decide)

def p_535274497 : CertifiedPrime :=
  certify 535274497 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_241, p_241] (by decide)

def p_535459601 : CertifiedPrime :=
  certify 535459601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13, p_89, p_89] (by decide)

def p_535529023 : CertifiedPrime :=
  certify 535529023 (6) [p_2, p_3, p_7, p_19, p_29, p_73, p_317] (by decide)

def p_535575307 : CertifiedPrime :=
  certify 535575307 (3) [p_2, p_3, p_7, p_19, p_29, p_23143] (by decide)

def p_535644737 : CertifiedPrime :=
  certify 535644737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_263, p_263] (by decide)

def p_535945651 : CertifiedPrime :=
  certify 535945651 (2) [p_2, p_3, p_5, p_5, p_463, p_7717] (by decide)

def p_536362441 : CertifiedPrime :=
  certify 536362441 (7) [p_2, p_2, p_2, p_3, p_5, p_193, p_23159] (by decide)

def p_536385601 : CertifiedPrime :=
  certify 536385601 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_193, p_193] (by decide)

def p_536547733 : CertifiedPrime :=
  certify 536547733 (2) [p_2, p_2, p_3, p_7, p_1103, p_5791] (by decide)

def p_536570897 : CertifiedPrime :=
  certify 536570897 (3) [p_2, p_2, p_2, p_2, p_5791, p_5791] (by decide)

def p_536663557 : CertifiedPrime :=
  certify 536663557 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_11, p_11, p_13, p_13] (by decide)

def p_537126977 : CertifiedPrime :=
  certify 537126977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2897, p_2897] (by decide)

def p_537242863 : CertifiedPrime :=
  certify 537242863 (3) [p_2, p_3, p_13, p_1783, p_3863] (by decide)

def p_537335581 : CertifiedPrime :=
  certify 537335581 (2) [p_2, p_2, p_3, p_5, p_19, p_61, p_7727] (by decide)

def p_537521041 : CertifiedPrime :=
  certify 537521041 (19) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_23, p_4637] (by decide)

def p_537590597 : CertifiedPrime :=
  certify 537590597 (2) [p_2, p_2, p_11593, p_11593] (by decide)

def p_537892057 : CertifiedPrime :=
  certify 537892057 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_13, p_223, p_859] (by decide)

def p_537961637 : CertifiedPrime :=
  certify 537961637 (2) [p_2, p_2, p_11597, p_11597] (by decide)

def p_538170403 : CertifiedPrime :=
  certify 538170403 (2) [p_2, p_3, p_7, p_11, p_19, p_37, p_1657] (by decide)

def p_538216801 : CertifiedPrime :=
  certify 538216801 (17) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_11, p_19, p_29, p_37] (by decide)

def p_539052307 : CertifiedPrime :=
  certify 539052307 (2) [p_2, p_3, p_13, p_19, p_47, p_71, p_109] (by decide)

def p_539191621 : CertifiedPrime :=
  certify 539191621 (7) [p_2, p_2, p_3, p_3, p_3, p_5, p_11, p_43, p_2111] (by decide)

def p_539354177 : CertifiedPrime :=
  certify 539354177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2903, p_2903] (by decide)

def p_540167323 : CertifiedPrime :=
  certify 540167323 (2) [p_2, p_3, p_61, p_127, p_11621] (by decide)

def p_540260293 : CertifiedPrime :=
  certify 540260293 (5) [p_2, p_2, p_3, p_11, p_13, p_149, p_2113] (by decide)

def p_540678757 : CertifiedPrime :=
  certify 540678757 (2) [p_2, p_2, p_3, p_23, p_337, p_5813] (by decide)

def p_541306757 : CertifiedPrime :=
  certify 541306757 (2) [p_2, p_2, p_11633, p_11633] (by decide)

def p_541562713 : CertifiedPrime :=
  certify 541562713 (5) [p_2, p_2, p_2, p_3, p_2909, p_7757] (by decide)

def p_541679077 : CertifiedPrime :=
  certify 541679077 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_431, p_431] (by decide)

def p_541702351 : CertifiedPrime :=
  certify 541702351 (7) [p_2, p_3, p_3, p_3, p_5, p_5, p_7, p_7, p_19, p_431] (by decide)

def p_542214511 : CertifiedPrime :=
  certify 542214511 (3) [p_2, p_3, p_5, p_3881, p_4657] (by decide)

def p_542773507 : CertifiedPrime :=
  certify 542773507 (3) [p_2, p_3, p_11, p_353, p_23297] (by decide)

def p_542890001 : CertifiedPrime :=
  certify 542890001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_233, p_233] (by decide)

def p_543169637 : CertifiedPrime :=
  certify 543169637 (2) [p_2, p_2, p_43, p_43, p_271, p_271] (by decide)

def p_543379411 : CertifiedPrime :=
  certify 543379411 (11) [p_2, p_3, p_3, p_5, p_7, p_37, p_23311] (by decide)

def p_543659173 : CertifiedPrime :=
  certify 543659173 (6) [p_2, p_2, p_3, p_7, p_29, p_67, p_3331] (by decide)

def p_544008977 : CertifiedPrime :=
  certify 544008977 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_7, p_7, p_7, p_7, p_17, p_17] (by decide)

def p_544102277 : CertifiedPrime :=
  certify 544102277 (2) [p_2, p_2, p_107, p_107, p_109, p_109] (by decide)

def p_544312231 : CertifiedPrime :=
  certify 544312231 (12) [p_2, p_3, p_5, p_7, p_11, p_101, p_2333] (by decide)

def p_544475557 : CertifiedPrime :=
  certify 544475557 (2) [p_2, p_2, p_3, p_3, p_3889, p_3889] (by decide)

def p_544568897 : CertifiedPrime :=
  certify 544568897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2917, p_2917] (by decide)

def p_546133531 : CertifiedPrime :=
  certify 546133531 (12) [p_2, p_3, p_5, p_19, p_41, p_23369] (by decide)

def p_546156901 : CertifiedPrime :=
  certify 546156901 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_19, p_41, p_41] (by decide)

def p_546554263 : CertifiedPrime :=
  certify 546554263 (3) [p_2, p_3, p_7793, p_11689] (by decide)

def p_546811457 : CertifiedPrime :=
  certify 546811457 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_37, p_37, p_79, p_79] (by decide)

def p_547255843 : CertifiedPrime :=
  certify 547255843 (3) [p_2, p_3, p_7, p_149, p_157, p_557] (by decide)

def p_547279237 : CertifiedPrime :=
  certify 547279237 (6) [p_2, p_2, p_3, p_3, p_7, p_7, p_557, p_557] (by decide)

def p_547677007 : CertifiedPrime :=
  certify 547677007 (3) [p_2, p_3, p_29, p_269, p_11701] (by decide)

def p_548215397 : CertifiedPrime :=
  certify 548215397 (2) [p_2, p_2, p_23, p_23, p_509, p_509] (by decide)

def p_548426143 : CertifiedPrime :=
  certify 548426143 (11) [p_2, p_3, p_3, p_11, p_1301, p_2129] (by decide)

def p_548566663 : CertifiedPrime :=
  certify 548566663 (3) [p_2, p_3, p_7, p_7, p_37, p_211, p_239] (by decide)

def p_548660353 : CertifiedPrime :=
  certify 548660353 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_59, p_61, p_397] (by decide)

def p_548683777 : CertifiedPrime :=
  certify 548683777 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_61, p_61] (by decide)

def p_549082057 : CertifiedPrime :=
  certify 549082057 (7) [p_2, p_2, p_2, p_3, p_29, p_73, p_101, p_107] (by decide)

def p_549246097 : CertifiedPrime :=
  certify 549246097 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7, p_31, p_31] (by decide)

def p_549433601 : CertifiedPrime :=
  certify 549433601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_293, p_293] (by decide)

def p_549714917 : CertifiedPrime :=
  certify 549714917 (2) [p_2, p_2, p_19, p_19, p_617, p_617] (by decide)

def p_549925951 : CertifiedPrime :=
  certify 549925951 (6) [p_2, p_3, p_5, p_5, p_7, p_67, p_7817] (by decide)

def p_549972853 : CertifiedPrime :=
  certify 549972853 (2) [p_2, p_2, p_3, p_11, p_13, p_41, p_7817] (by decide)

def p_550207393 : CertifiedPrime :=
  certify 550207393 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_733, p_1117] (by decide)

def p_550629691 : CertifiedPrime :=
  certify 550629691 (10) [p_2, p_3, p_5, p_13, p_19, p_19, p_3911] (by decide)

def p_550653157 : CertifiedPrime :=
  certify 550653157 (5) [p_2, p_2, p_3, p_3, p_3911, p_3911] (by decide)

def p_551122577 : CertifiedPrime :=
  certify 551122577 (3) [p_2, p_2, p_2, p_2, p_5869, p_5869] (by decide)

def p_551474773 : CertifiedPrime :=
  certify 551474773 (5) [p_2, p_2, p_3, p_19, p_23, p_103, p_1021] (by decide)

def p_551803591 : CertifiedPrime :=
  certify 551803591 (11) [p_2, p_3, p_3, p_3, p_3, p_5, p_13, p_13, p_29, p_139] (by decide)

def p_551944543 : CertifiedPrime :=
  certify 551944543 (3) [p_2, p_3, p_17, p_41, p_191, p_691] (by decide)

def p_552367507 : CertifiedPrime :=
  certify 552367507 (3) [p_2, p_3, p_19, p_1237, p_3917] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
