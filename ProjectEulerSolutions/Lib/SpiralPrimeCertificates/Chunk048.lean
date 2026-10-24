import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk047

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_587892763 : CertifiedPrime :=
  certify 587892763 (5) [p_2, p_3, p_3, p_3, p_449, p_24247] (by decide)

def p_588038251 : CertifiedPrime :=
  certify 588038251 (2) [p_2, p_3, p_5, p_5, p_5, p_59, p_97, p_137] (by decide)

def p_588547601 : CertifiedPrime :=
  certify 588547601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1213, p_1213] (by decide)

def p_588571861 : CertifiedPrime :=
  certify 588571861 (2) [p_2, p_2, p_3, p_5, p_1213, p_8087] (by decide)

def p_588717433 : CertifiedPrime :=
  certify 588717433 (5) [p_2, p_2, p_2, p_3, p_3, p_19, p_337, p_1277] (by decide)

def p_589008631 : CertifiedPrime :=
  certify 589008631 (7) [p_2, p_3, p_5, p_7, p_809, p_3467] (by decide)

def p_589032901 : CertifiedPrime :=
  certify 589032901 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_809, p_809] (by decide)

def p_589299901 : CertifiedPrime :=
  certify 589299901 (6) [p_2, p_2, p_3, p_5, p_5, p_7, p_17, p_17, p_971] (by decide)

def p_589591243 : CertifiedPrime :=
  certify 589591243 (13) [p_2, p_3, p_3, p_19, p_71, p_24281] (by decide)

def p_589712657 : CertifiedPrime :=
  certify 589712657 (5) [p_2, p_2, p_2, p_2, p_13, p_13, p_467, p_467] (by decide)

def p_590028391 : CertifiedPrime :=
  certify 590028391 (12) [p_2, p_3, p_3, p_5, p_7, p_347, p_2699] (by decide)

def p_590319913 : CertifiedPrime :=
  certify 590319913 (11) [p_2, p_2, p_2, p_3, p_7, p_13, p_89, p_3037] (by decide)

def p_590368507 : CertifiedPrime :=
  certify 590368507 (3) [p_2, p_3, p_7, p_13, p_89, p_12149] (by decide)

def p_590660113 : CertifiedPrime :=
  certify 590660113 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_31, p_8101] (by decide)

def p_591535363 : CertifiedPrime :=
  certify 591535363 (2) [p_2, p_3, p_11, p_11, p_67, p_12161] (by decide)

def p_591656977 : CertifiedPrime :=
  certify 591656977 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_2027, p_2027] (by decide)

def p_591754277 : CertifiedPrime :=
  certify 591754277 (2) [p_2, p_2, p_12163, p_12163] (by decide)

def p_591973231 : CertifiedPrime :=
  certify 591973231 (3) [p_2, p_3, p_5, p_29, p_811, p_839] (by decide)

def p_592362583 : CertifiedPrime :=
  certify 592362583 (3) [p_2, p_3, p_7, p_19, p_43, p_61, p_283] (by decide)

def p_592557307 : CertifiedPrime :=
  certify 592557307 (3) [p_2, p_3, p_11, p_2213, p_4057] (by decide)

def p_592995553 : CertifiedPrime :=
  certify 592995553 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_761, p_8117] (by decide)

def p_593214737 : CertifiedPrime :=
  certify 593214737 (3) [p_2, p_2, p_2, p_2, p_6089, p_6089] (by decide)

def p_593580133 : CertifiedPrime :=
  certify 593580133 (2) [p_2, p_2, p_3, p_3, p_2707, p_6091] (by decide)

def p_593604497 : CertifiedPrime :=
  certify 593604497 (3) [p_2, p_2, p_2, p_2, p_6091, p_6091] (by decide)

def p_593701957 : CertifiedPrime :=
  certify 593701957 (2) [p_2, p_2, p_3, p_3, p_31, p_31, p_131, p_131] (by decide)

def p_593726323 : CertifiedPrime :=
  certify 593726323 (3) [p_2, p_3, p_7, p_31, p_59, p_59, p_131] (by decide)

def p_594408781 : CertifiedPrime :=
  certify 594408781 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_23, p_43, p_53] (by decide)

def p_594457543 : CertifiedPrime :=
  certify 594457543 (5) [p_2, p_3, p_3, p_3, p_3, p_7, p_43, p_73, p_167] (by decide)

def p_594579457 : CertifiedPrime :=
  certify 594579457 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_127, p_127] (by decide)

def p_595140421 : CertifiedPrime :=
  certify 595140421 (11) [p_2, p_2, p_3, p_5, p_7, p_17, p_19, p_41, p_107] (by decide)

def p_595286803 : CertifiedPrime :=
  certify 595286803 (2) [p_2, p_3, p_3, p_11, p_1109, p_2711] (by decide)

def p_595433203 : CertifiedPrime :=
  certify 595433203 (12) [p_2, p_3, p_7, p_7, p_13, p_83, p_1877] (by decide)

def p_595628431 : CertifiedPrime :=
  certify 595628431 (3) [p_2, p_3, p_5, p_1627, p_12203] (by decide)

def p_596067811 : CertifiedPrime :=
  certify 596067811 (3) [p_2, p_3, p_5, p_13, p_19, p_257, p_313] (by decide)

def p_596898193 : CertifiedPrime :=
  certify 596898193 (13) [p_2, p_2, p_2, p_2, p_3, p_11, p_509, p_2221] (by decide)

def p_597020357 : CertifiedPrime :=
  certify 597020357 (2) [p_2, p_2, p_19, p_19, p_643, p_643] (by decide)

def p_597093661 : CertifiedPrime :=
  certify 597093661 (6) [p_2, p_2, p_3, p_3, p_3, p_5, p_41, p_149, p_181] (by decide)

def p_597191407 : CertifiedPrime :=
  certify 597191407 (3) [p_2, p_3, p_7, p_3491, p_4073] (by decide)

def p_597313601 : CertifiedPrime :=
  certify 597313601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13, p_47, p_47] (by decide)

def p_597509137 : CertifiedPrime :=
  certify 597509137 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_97, p_97] (by decide)

def p_597606917 : CertifiedPrime :=
  certify 597606917 (2) [p_2, p_2, p_17, p_17, p_719, p_719] (by decide)

def p_597924757 : CertifiedPrime :=
  certify 597924757 (5) [p_2, p_2, p_3, p_3, p_11, p_13, p_19, p_6113] (by decide)

def p_598218223 : CertifiedPrime :=
  certify 598218223 (3) [p_2, p_3, p_7, p_31, p_263, p_1747] (by decide)

def p_598267141 : CertifiedPrime :=
  certify 598267141 (2) [p_2, p_2, p_3, p_5, p_31, p_263, p_1223] (by decide)

def p_598291601 : CertifiedPrime :=
  certify 598291601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1223, p_1223] (by decide)

def p_598560691 : CertifiedPrime :=
  certify 598560691 (2) [p_2, p_3, p_5, p_7, p_13, p_233, p_941] (by decide)

def p_598780901 : CertifiedPrime :=
  certify 598780901 (2) [p_2, p_2, p_5, p_5, p_2447, p_2447] (by decide)

def p_598805371 : CertifiedPrime :=
  certify 598805371 (13) [p_2, p_3, p_3, p_5, p_2447, p_2719] (by decide)

def p_598854313 : CertifiedPrime :=
  certify 598854313 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_19, p_23, p_2719] (by decide)

def p_599294881 : CertifiedPrime :=
  certify 599294881 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_17, p_24481] (by decide)

def p_599539711 : CertifiedPrime :=
  certify 599539711 (3) [p_2, p_3, p_5, p_7, p_11, p_53, p_59, p_83] (by decide)

def p_599760101 : CertifiedPrime :=
  certify 599760101 (2) [p_2, p_2, p_5, p_5, p_31, p_31, p_79, p_79] (by decide)

def p_599956037 : CertifiedPrime :=
  certify 599956037 (2) [p_2, p_2, p_37, p_37, p_331, p_331] (by decide)

def p_600054017 : CertifiedPrime :=
  certify 600054017 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1531, p_1531] (by decide)

def p_600764611 : CertifiedPrime :=
  certify 600764611 (3) [p_2, p_3, p_5, p_19, p_43, p_127, p_193] (by decide)

def p_601353007 : CertifiedPrime :=
  certify 601353007 (3) [p_2, p_3, p_61, p_67, p_137, p_179] (by decide)

def p_601426577 : CertifiedPrime :=
  certify 601426577 (3) [p_2, p_2, p_2, p_2, p_6131, p_6131] (by decide)

def p_601451101 : CertifiedPrime :=
  certify 601451101 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_109, p_6131] (by decide)

def p_601647313 : CertifiedPrime :=
  certify 601647313 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_19, p_73, p_1291] (by decide)

def p_601794493 : CertifiedPrime :=
  certify 601794493 (2) [p_2, p_2, p_3, p_13, p_17, p_37, p_6133] (by decide)

def p_601892623 : CertifiedPrime :=
  certify 601892623 (5) [p_2, p_3, p_3, p_29, p_47, p_24533] (by decide)

def p_601917157 : CertifiedPrime :=
  certify 601917157 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_29, p_29, p_47, p_47] (by decide)

def p_602383393 : CertifiedPrime :=
  certify 602383393 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_13, p_59, p_101] (by decide)

def p_602997137 : CertifiedPrime :=
  certify 602997137 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_877, p_877] (by decide)

def p_603267283 : CertifiedPrime :=
  certify 603267283 (2) [p_2, p_3, p_3, p_2729, p_12281] (by decide)

def p_603390097 : CertifiedPrime :=
  certify 603390097 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_23, p_23, p_89, p_89] (by decide)

def p_603955201 : CertifiedPrime :=
  certify 603955201 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_983] (by decide)

def p_604668101 : CertifiedPrime :=
  certify 604668101 (2) [p_2, p_2, p_5, p_5, p_2459, p_2459] (by decide)

def p_604963217 : CertifiedPrime :=
  certify 604963217 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_13, p_13, p_43, p_43] (by decide)

def p_605160001 : CertifiedPrime :=
  certify 605160001 (21) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_41, p_41] (by decide)

def p_605332213 : CertifiedPrime :=
  certify 605332213 (2) [p_2, p_2, p_3, p_59, p_139, p_6151] (by decide)

def p_605455237 : CertifiedPrime :=
  certify 605455237 (11) [p_2, p_2, p_3, p_3, p_3, p_3, p_1367, p_1367] (by decide)

def p_606612271 : CertifiedPrime :=
  certify 606612271 (6) [p_2, p_3, p_5, p_11, p_821, p_2239] (by decide)

def p_606809323 : CertifiedPrime :=
  certify 606809323 (2) [p_2, p_3, p_3, p_7, p_17, p_23, p_109, p_113] (by decide)

def p_607795063 : CertifiedPrime :=
  certify 607795063 (5) [p_2, p_3, p_7, p_89, p_277, p_587] (by decide)

def p_608090941 : CertifiedPrime :=
  certify 608090941 (7) [p_2, p_2, p_3, p_3, p_5, p_137, p_24659] (by decide)

def p_608140261 : CertifiedPrime :=
  certify 608140261 (14) [p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_137, p_271] (by decide)

def p_608806277 : CertifiedPrime :=
  certify 608806277 (2) [p_2, p_2, p_13, p_13, p_13, p_13, p_73, p_73] (by decide)

def p_608830951 : CertifiedPrime :=
  certify 608830951 (3) [p_2, p_3, p_5, p_5, p_7, p_13, p_13, p_47, p_73] (by decide)

def p_608904977 : CertifiedPrime :=
  certify 608904977 (3) [p_2, p_2, p_2, p_2, p_31, p_31, p_199, p_199] (by decide)

def p_609028363 : CertifiedPrime :=
  certify 609028363 (11) [p_2, p_3, p_3, p_3, p_23, p_29, p_37, p_457] (by decide)

def p_609472657 : CertifiedPrime :=
  certify 609472657 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_211, p_1543] (by decide)

def p_609867721 : CertifiedPrime :=
  certify 609867721 (13) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_7, p_7, p_11, p_449] (by decide)

def p_610090001 : CertifiedPrime :=
  certify 610090001 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_13, p_13, p_19, p_19] (by decide)

def p_610509973 : CertifiedPrime :=
  certify 610509973 (6) [p_2, p_2, p_3, p_29, p_71, p_24709] (by decide)

def p_611103121 : CertifiedPrime :=
  certify 611103121 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_59, p_103, p_419] (by decide)

def p_611251453 : CertifiedPrime :=
  certify 611251453 (6) [p_2, p_2, p_3, p_3, p_7, p_41, p_67, p_883] (by decide)

def p_611350351 : CertifiedPrime :=
  certify 611350351 (3) [p_2, p_3, p_5, p_5, p_13, p_23, p_43, p_317] (by decide)

def p_611375077 : CertifiedPrime :=
  certify 611375077 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_317, p_317] (by decide)

def p_611572901 : CertifiedPrime :=
  certify 611572901 (2) [p_2, p_2, p_5, p_5, p_2473, p_2473] (by decide)

def p_611869697 : CertifiedPrime :=
  certify 611869697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_773, p_773] (by decide)

def p_611943907 : CertifiedPrime :=
  certify 611943907 (3) [p_2, p_3, p_7, p_19, p_29, p_31, p_853] (by decide)

def p_612141823 : CertifiedPrime :=
  certify 612141823 (5) [p_2, p_3, p_3, p_89, p_139, p_2749] (by decide)

def p_612983323 : CertifiedPrime :=
  certify 612983323 (2) [p_2, p_3, p_3, p_3, p_7, p_131, p_12379] (by decide)

def p_613032841 : CertifiedPrime :=
  certify 613032841 (13) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_131, p_619] (by decide)

def p_613181407 : CertifiedPrime :=
  certify 613181407 (3) [p_2, p_3, p_4127, p_24763] (by decide)

def p_613627213 : CertifiedPrime :=
  certify 613627213 (2) [p_2, p_2, p_3, p_11, p_23, p_359, p_563] (by decide)

def p_613726303 : CertifiedPrime :=
  certify 613726303 (5) [p_2, p_3, p_7, p_3539, p_4129] (by decide)

def p_614321011 : CertifiedPrime :=
  certify 614321011 (10) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_17, p_4957] (by decide)

def p_614742437 : CertifiedPrime :=
  certify 614742437 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_11, p_11, p_23, p_23] (by decide)

def p_614916007 : CertifiedPrime :=
  certify 614916007 (3) [p_2, p_3, p_137, p_181, p_4133] (by decide)

def p_615040001 : CertifiedPrime :=
  certify 615040001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_31, p_31] (by decide)

def p_615238417 : CertifiedPrime :=
  certify 615238417 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13, p_53, p_53] (by decide)

def p_615809041 : CertifiedPrime :=
  certify 615809041 (62) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_47, p_709] (by decide)

def p_616032401 : CertifiedPrime :=
  certify 616032401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_73, p_73] (by decide)

def p_616156507 : CertifiedPrime :=
  certify 616156507 (5) [p_2, p_3, p_3, p_7, p_103, p_197, p_241] (by decide)

def p_617224337 : CertifiedPrime :=
  certify 617224337 (3) [p_2, p_2, p_2, p_2, p_6211, p_6211] (by decide)

def p_617447953 : CertifiedPrime :=
  certify 617447953 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_251, p_1553] (by decide)

def p_617945023 : CertifiedPrime :=
  certify 617945023 (13) [p_2, p_3, p_3, p_1381, p_24859] (by decide)

def p_618541771 : CertifiedPrime :=
  certify 618541771 (7) [p_2, p_3, p_5, p_7, p_11, p_17, p_19, p_829] (by decide)

def p_618815377 : CertifiedPrime :=
  certify 618815377 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_691, p_691] (by decide)

def p_619089043 : CertifiedPrime :=
  certify 619089043 (3) [p_2, p_3, p_11, p_13, p_29, p_139, p_179] (by decide)

def p_620283931 : CertifiedPrime :=
  certify 620283931 (10) [p_2, p_3, p_5, p_7, p_17, p_293, p_593] (by decide)

def p_620308837 : CertifiedPrime :=
  certify 620308837 (6) [p_2, p_2, p_3, p_3, p_7, p_7, p_593, p_593] (by decide)

def p_620333743 : CertifiedPrime :=
  certify 620333743 (3) [p_2, p_3, p_7, p_593, p_24907] (by decide)

def p_620707397 : CertifiedPrime :=
  certify 620707397 (2) [p_2, p_2, p_12457, p_12457] (by decide)

def p_620732311 : CertifiedPrime :=
  certify 620732311 (3) [p_2, p_3, p_5, p_11, p_151, p_12457] (by decide)

def p_621205777 : CertifiedPrime :=
  certify 621205777 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_31, p_31, p_67, p_67] (by decide)

def p_621330403 : CertifiedPrime :=
  certify 621330403 (2) [p_2, p_3, p_7, p_11, p_11, p_103, p_1187] (by decide)

def p_621504901 : CertifiedPrime :=
  certify 621504901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_277, p_277] (by decide)

def p_621629557 : CertifiedPrime :=
  certify 621629557 (2) [p_2, p_2, p_3, p_23, p_271, p_8311] (by decide)

def p_621804097 : CertifiedPrime :=
  certify 621804097 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_1039, p_1039] (by decide)

def p_622228081 : CertifiedPrime :=
  certify 622228081 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_1559, p_1663] (by decide)

def p_622577353 : CertifiedPrime :=
  certify 622577353 (5) [p_2, p_2, p_2, p_3, p_3119, p_8317] (by decide)

def p_623276191 : CertifiedPrime :=
  certify 623276191 (13) [p_2, p_3, p_3, p_5, p_19, p_73, p_4993] (by decide)

def p_623301157 : CertifiedPrime :=
  certify 623301157 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_19, p_19, p_73, p_73] (by decide)

def p_623700677 : CertifiedPrime :=
  certify 623700677 (2) [p_2, p_2, p_12487, p_12487] (by decide)

def p_623800577 : CertifiedPrime :=
  certify 623800577 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_223, p_223] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
