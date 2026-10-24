import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk043

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_459866581 : CertifiedPrime :=
  certify 459866581 (6) [p_2, p_2, p_3, p_5, p_1787, p_4289] (by decide)

def p_459930917 : CertifiedPrime :=
  certify 459930917 (2) [p_2, p_2, p_10723, p_10723] (by decide)

def p_460359937 : CertifiedPrime :=
  certify 460359937 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_149, p_149] (by decide)

def p_460381393 : CertifiedPrime :=
  certify 460381393 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_43, p_149, p_499] (by decide)

def p_460531601 : CertifiedPrime :=
  certify 460531601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_29, p_29, p_37, p_37] (by decide)

def p_460595983 : CertifiedPrime :=
  certify 460595983 (3) [p_2, p_3, p_7, p_7, p_11, p_73, p_1951] (by decide)

def p_460703297 : CertifiedPrime :=
  certify 460703297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2683, p_2683] (by decide)

def p_460767691 : CertifiedPrime :=
  certify 460767691 (2) [p_2, p_3, p_3, p_3, p_3, p_5, p_53, p_10733] (by decide)

def p_460896493 : CertifiedPrime :=
  certify 460896493 (6) [p_2, p_2, p_3, p_7, p_1789, p_3067] (by decide)

def p_461132677 : CertifiedPrime :=
  certify 461132677 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_1193, p_1193] (by decide)

def p_461240053 : CertifiedPrime :=
  certify 461240053 (2) [p_2, p_2, p_3, p_7, p_13, p_59, p_7159] (by decide)

def p_461648197 : CertifiedPrime :=
  certify 461648197 (2) [p_2, p_2, p_3, p_3, p_3581, p_3581] (by decide)

def p_462056521 : CertifiedPrime :=
  certify 462056521 (17) [p_2, p_2, p_2, p_3, p_5, p_1433, p_2687] (by decide)

def p_462443521 : CertifiedPrime :=
  certify 462443521 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_17, p_23] (by decide)

def p_462658591 : CertifiedPrime :=
  certify 462658591 (21) [p_2, p_3, p_3, p_5, p_137, p_157, p_239] (by decide)

def p_462787657 : CertifiedPrime :=
  certify 462787657 (5) [p_2, p_2, p_2, p_3, p_71, p_101, p_2689] (by decide)

def p_462852197 : CertifiedPrime :=
  certify 462852197 (2) [p_2, p_2, p_31, p_31, p_347, p_347] (by decide)

def p_463347151 : CertifiedPrime :=
  certify 463347151 (3) [p_2, p_3, p_5, p_5, p_7, p_41, p_47, p_229] (by decide)

def p_463562431 : CertifiedPrime :=
  certify 463562431 (3) [p_2, p_3, p_5, p_2153, p_7177] (by decide)

def p_463691623 : CertifiedPrime :=
  certify 463691623 (3) [p_2, p_3, p_37, p_61, p_97, p_353] (by decide)

def p_464079307 : CertifiedPrime :=
  certify 464079307 (2) [p_2, p_3, p_43, p_167, p_10771] (by decide)

def p_464402501 : CertifiedPrime :=
  certify 464402501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_431, p_431] (by decide)

def p_464574917 : CertifiedPrime :=
  certify 464574917 (2) [p_2, p_2, p_13, p_13, p_829, p_829] (by decide)

def p_464596471 : CertifiedPrime :=
  certify 464596471 (6) [p_2, p_3, p_3, p_5, p_13, p_479, p_829] (by decide)

def p_464768923 : CertifiedPrime :=
  certify 464768923 (3) [p_2, p_3, p_3593, p_21559] (by decide)

def p_464898283 : CertifiedPrime :=
  certify 464898283 (2) [p_2, p_3, p_7187, p_10781] (by decide)

def p_465092357 : CertifiedPrime :=
  certify 465092357 (2) [p_2, p_2, p_41, p_41, p_263, p_263] (by decide)

def p_465113923 : CertifiedPrime :=
  certify 465113923 (2) [p_2, p_3, p_7, p_13, p_41, p_79, p_263] (by decide)

def p_465157057 : CertifiedPrime :=
  certify 465157057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_79, p_337] (by decide)

def p_465264901 : CertifiedPrime :=
  certify 465264901 (17) [p_2, p_2, p_3, p_3, p_5, p_5, p_719, p_719] (by decide)

def p_465523777 : CertifiedPrime :=
  certify 465523777 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_29, p_29, p_31, p_31] (by decide)

def p_465804307 : CertifiedPrime :=
  certify 465804307 (5) [p_2, p_3, p_3, p_11, p_109, p_113, p_191] (by decide)

def p_466279243 : CertifiedPrime :=
  certify 466279243 (5) [p_2, p_3, p_11, p_13, p_59, p_61, p_151] (by decide)

def p_466322431 : CertifiedPrime :=
  certify 466322431 (3) [p_2, p_3, p_5, p_7, p_59, p_61, p_617] (by decide)

def p_466560001 : CertifiedPrime :=
  certify 466560001 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5] (by decide)

def p_466797631 : CertifiedPrime :=
  certify 466797631 (3) [p_2, p_3, p_5, p_13, p_29, p_149, p_277] (by decide)

def p_466819237 : CertifiedPrime :=
  certify 466819237 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_277, p_277] (by decide)

def p_467316307 : CertifiedPrime :=
  certify 467316307 (5) [p_2, p_3, p_3, p_1201, p_21617] (by decide)

def p_467619001 : CertifiedPrime :=
  certify 467619001 (23) [p_2, p_2, p_2, p_3, p_5, p_5, p_5, p_17, p_53, p_173] (by decide)

def p_467878531 : CertifiedPrime :=
  certify 467878531 (3) [p_2, p_3, p_5, p_7, p_97, p_103, p_223] (by decide)

def p_468116497 : CertifiedPrime :=
  certify 468116497 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_601, p_601] (by decide)

def p_468224683 : CertifiedPrime :=
  certify 468224683 (2) [p_2, p_3, p_31, p_349, p_7213] (by decide)

def p_468484381 : CertifiedPrime :=
  certify 468484381 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_37, p_773] (by decide)

def p_468614257 : CertifiedPrime :=
  certify 468614257 (5) [p_2, p_2, p_2, p_2, p_3, p_11, p_41, p_21647] (by decide)

def p_468744151 : CertifiedPrime :=
  certify 468744151 (3) [p_2, p_3, p_5, p_5, p_7, p_433, p_1031] (by decide)

def p_469047307 : CertifiedPrime :=
  certify 469047307 (2) [p_2, p_3, p_7, p_7, p_13, p_17, p_7219] (by decide)

def p_469133941 : CertifiedPrime :=
  certify 469133941 (6) [p_2, p_2, p_3, p_5, p_11, p_11, p_19, p_19, p_179] (by decide)

def p_469328897 : CertifiedPrime :=
  certify 469328897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_677, p_677] (by decide)

def p_469827301 : CertifiedPrime :=
  certify 469827301 (2) [p_2, p_2, p_3, p_5, p_5, p_17, p_17, p_5419] (by decide)

def p_469914007 : CertifiedPrime :=
  certify 469914007 (3) [p_2, p_3, p_53, p_409, p_3613] (by decide)

def p_470824903 : CertifiedPrime :=
  certify 470824903 (5) [p_2, p_3, p_3, p_19, p_571, p_2411] (by decide)

def p_470868301 : CertifiedPrime :=
  certify 470868301 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_31, p_2411] (by decide)

def p_471215557 : CertifiedPrime :=
  certify 471215557 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_67, p_443] (by decide)

def p_471345811 : CertifiedPrime :=
  certify 471345811 (2) [p_2, p_3, p_5, p_13, p_167, p_7237] (by decide)

def p_471497797 : CertifiedPrime :=
  certify 471497797 (18) [p_2, p_2, p_3, p_3, p_7, p_7, p_11, p_11, p_47, p_47] (by decide)

def p_471519511 : CertifiedPrime :=
  certify 471519511 (6) [p_2, p_3, p_5, p_7, p_11, p_43, p_47, p_101] (by decide)

def p_471780121 : CertifiedPrime :=
  certify 471780121 (17) [p_2, p_2, p_2, p_3, p_5, p_7, p_29, p_107, p_181] (by decide)

def p_472019077 : CertifiedPrime :=
  certify 472019077 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_17, p_17, p_71, p_71] (by decide)

def p_472258093 : CertifiedPrime :=
  certify 472258093 (6) [p_2, p_2, p_3, p_31, p_701, p_1811] (by decide)

def p_472453697 : CertifiedPrime :=
  certify 472453697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_13, p_13, p_19, p_19] (by decide)

def p_472562383 : CertifiedPrime :=
  certify 472562383 (3) [p_2, p_3, p_3623, p_21739] (by decide)

def p_472649341 : CertifiedPrime :=
  certify 472649341 (2) [p_2, p_2, p_3, p_5, p_1087, p_7247] (by decide)

def p_473040751 : CertifiedPrime :=
  certify 473040751 (28) [p_2, p_3, p_5, p_5, p_5, p_7, p_13, p_29, p_239] (by decide)

def p_473475841 : CertifiedPrime :=
  certify 473475841 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_17, p_7253] (by decide)

def p_473562883 : CertifiedPrime :=
  certify 473562883 (14) [p_2, p_3, p_3, p_3, p_13, p_31, p_47, p_463] (by decide)

def p_473954671 : CertifiedPrime :=
  certify 473954671 (6) [p_2, p_3, p_3, p_5, p_7, p_41, p_59, p_311] (by decide)

def p_474085303 : CertifiedPrime :=
  certify 474085303 (5) [p_2, p_3, p_19, p_191, p_21773] (by decide)

def p_474194177 : CertifiedPrime :=
  certify 474194177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1361, p_1361] (by decide)

def p_474542657 : CertifiedPrime :=
  certify 474542657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_389, p_389] (by decide)

def p_474913057 : CertifiedPrime :=
  certify 474913057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_19, p_31, p_37, p_227] (by decide)

def p_475414417 : CertifiedPrime :=
  certify 475414417 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_23, p_23, p_79, p_79] (by decide)

def p_475523443 : CertifiedPrime :=
  certify 475523443 (2) [p_2, p_3, p_3, p_2423, p_10903] (by decide)

def p_475676101 : CertifiedPrime :=
  certify 475676101 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_727, p_727] (by decide)

def p_475850597 : CertifiedPrime :=
  certify 475850597 (2) [p_2, p_2, p_13, p_13, p_839, p_839] (by decide)

def p_475916041 : CertifiedPrime :=
  certify 475916041 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_101, p_4363] (by decide)

def p_476352451 : CertifiedPrime :=
  certify 476352451 (10) [p_2, p_3, p_3, p_5, p_5, p_7, p_97, p_1559] (by decide)

def p_476548901 : CertifiedPrime :=
  certify 476548901 (2) [p_2, p_2, p_5, p_5, p_37, p_37, p_59, p_59] (by decide)

def p_476614393 : CertifiedPrime :=
  certify 476614393 (5) [p_2, p_2, p_2, p_3, p_19, p_383, p_2729] (by decide)

def p_476701723 : CertifiedPrime :=
  certify 476701723 (7) [p_2, p_3, p_3, p_7, p_1213, p_3119] (by decide)

def p_476985601 : CertifiedPrime :=
  certify 476985601 (83) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_13, p_13] (by decide)

def p_477662881 : CertifiedPrime :=
  certify 477662881 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_31, p_47, p_683] (by decide)

def p_477684737 : CertifiedPrime :=
  certify 477684737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_683, p_683] (by decide)

def p_478143823 : CertifiedPrime :=
  certify 478143823 (3) [p_2, p_3, p_13, p_29, p_29, p_37, p_197] (by decide)

def p_478296901 : CertifiedPrime :=
  certify 478296901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_478668763 : CertifiedPrime :=
  certify 478668763 (5) [p_2, p_3, p_3, p_11, p_13, p_17, p_10939] (by decide)

def p_478734401 : CertifiedPrime :=
  certify 478734401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_547, p_547] (by decide)

def p_478996997 : CertifiedPrime :=
  certify 478996997 (2) [p_2, p_2, p_31, p_31, p_353, p_353] (by decide)

def p_479062657 : CertifiedPrime :=
  certify 479062657 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_19, p_43, p_509] (by decide)

def p_479347237 : CertifiedPrime :=
  certify 479347237 (2) [p_2, p_2, p_3, p_3, p_41, p_41, p_89, p_89] (by decide)

def p_479369131 : CertifiedPrime :=
  certify 479369131 (3) [p_2, p_3, p_5, p_29, p_41, p_89, p_151] (by decide)

def p_479434817 : CertifiedPrime :=
  certify 479434817 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_17, p_17, p_23, p_23] (by decide)

def p_479500507 : CertifiedPrime :=
  certify 479500507 (2) [p_2, p_3, p_3, p_3, p_811, p_10949] (by decide)

def p_479719507 : CertifiedPrime :=
  certify 479719507 (2) [p_2, p_3, p_7, p_7, p_47, p_149, p_233] (by decide)

def p_479872837 : CertifiedPrime :=
  certify 479872837 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_1217, p_1217] (by decide)

def p_480223397 : CertifiedPrime :=
  certify 480223397 (2) [p_2, p_2, p_10957, p_10957] (by decide)

def p_480311057 : CertifiedPrime :=
  certify 480311057 (3) [p_2, p_2, p_2, p_2, p_5479, p_5479] (by decide)

def p_480661777 : CertifiedPrime :=
  certify 480661777 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7, p_29, p_29] (by decide)

def p_480683701 : CertifiedPrime :=
  certify 480683701 (10) [p_2, p_2, p_3, p_3, p_3, p_5, p_5, p_7, p_29, p_877] (by decide)

def p_480924901 : CertifiedPrime :=
  certify 480924901 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_17, p_17, p_43, p_43] (by decide)

def p_481341661 : CertifiedPrime :=
  certify 481341661 (2) [p_2, p_2, p_3, p_5, p_71, p_103, p_1097] (by decide)

def p_481363601 : CertifiedPrime :=
  certify 481363601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1097, p_1097] (by decide)

def p_481561081 : CertifiedPrime :=
  certify 481561081 (17) [p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_13, p_19, p_211] (by decide)

def p_481736653 : CertifiedPrime :=
  certify 481736653 (6) [p_2, p_2, p_3, p_31, p_47, p_59, p_467] (by decide)

def p_481824451 : CertifiedPrime :=
  certify 481824451 (2) [p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_271, p_439] (by decide)

def p_481978117 : CertifiedPrime :=
  certify 481978117 (2) [p_2, p_2, p_3, p_3, p_3659, p_3659] (by decide)

def p_482065937 : CertifiedPrime :=
  certify 482065937 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_499, p_499] (by decide)

def p_482351407 : CertifiedPrime :=
  certify 482351407 (3) [p_2, p_3, p_79, p_139, p_7321] (by decide)

def p_482417297 : CertifiedPrime :=
  certify 482417297 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_17, p_17, p_19, p_19] (by decide)

def p_482658931 : CertifiedPrime :=
  certify 482658931 (12) [p_2, p_3, p_3, p_5, p_13, p_13, p_13, p_2441] (by decide)

def p_482878651 : CertifiedPrime :=
  certify 482878651 (2) [p_2, p_3, p_5, p_5, p_293, p_10987] (by decide)

def p_483120401 : CertifiedPrime :=
  certify 483120401 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_157, p_157] (by decide)

def p_483384197 : CertifiedPrime :=
  certify 483384197 (2) [p_2, p_2, p_10993, p_10993] (by decide)

def p_483560101 : CertifiedPrime :=
  certify 483560101 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_733, p_733] (by decide)

def p_483824017 : CertifiedPrime :=
  certify 483824017 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13, p_47, p_47] (by decide)

def p_484638211 : CertifiedPrime :=
  certify 484638211 (7) [p_2, p_3, p_3, p_5, p_7, p_17, p_37, p_1223] (by decide)

def p_484726273 : CertifiedPrime :=
  certify 484726273 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_41, p_43, p_179] (by decide)

def p_484902421 : CertifiedPrime :=
  certify 484902421 (17) [p_2, p_2, p_3, p_5, p_19, p_19, p_61, p_367] (by decide)

def p_485034553 : CertifiedPrime :=
  certify 485034553 (7) [p_2, p_2, p_2, p_3, p_3, p_2447, p_2753] (by decide)

def p_485497157 : CertifiedPrime :=
  certify 485497157 (3) [p_2, p_2, p_23, p_23, p_479, p_479] (by decide)

def p_485937937 : CertifiedPrime :=
  certify 485937937 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11, p_167, p_167] (by decide)

def p_486180451 : CertifiedPrime :=
  certify 486180451 (11) [p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_17, p_1297] (by decide)

def p_486356863 : CertifiedPrime :=
  certify 486356863 (3) [p_2, p_3, p_7351, p_11027] (by decide)

def p_486378917 : CertifiedPrime :=
  certify 486378917 (2) [p_2, p_2, p_11027, p_11027] (by decide)

def p_486489193 : CertifiedPrime :=
  certify 486489193 (5) [p_2, p_2, p_2, p_3, p_7, p_23, p_137, p_919] (by decide)

def p_486643601 : CertifiedPrime :=
  certify 486643601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1103, p_1103] (by decide)

def p_487151113 : CertifiedPrime :=
  certify 487151113 (20) [p_2, p_2, p_2, p_3, p_7, p_31, p_89, p_1051] (by decide)

def p_487261477 : CertifiedPrime :=
  certify 487261477 (7) [p_2, p_2, p_3, p_3, p_13, p_13, p_283, p_283] (by decide)

def p_487283551 : CertifiedPrime :=
  certify 487283551 (3) [p_2, p_3, p_5, p_5, p_13, p_283, p_883] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
