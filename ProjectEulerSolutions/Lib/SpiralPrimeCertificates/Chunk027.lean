import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk026

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_97347823 : CertifiedPrime :=
  certify 97347823 (3) [p_2, p_3, p_11, p_13, p_23, p_4933] (by decide)

def p_97495877 : CertifiedPrime :=
  certify 97495877 (2) [p_2, p_2, p_4937, p_4937] (by decide)

def p_97525501 : CertifiedPrime :=
  certify 97525501 (6) [p_2, p_2, p_3, p_5, p_5, p_5, p_79, p_823] (by decide)

def p_97535377 : CertifiedPrime :=
  certify 97535377 (15) [p_2, p_2, p_2, p_2, p_3, p_3, p_823, p_823] (by decide)

def p_97614401 : CertifiedPrime :=
  certify 97614401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13, p_19, p_19] (by decide)

def p_97663807 : CertifiedPrime :=
  certify 97663807 (5) [p_2, p_3, p_3, p_3, p_3, p_61, p_9883] (by decide)

def p_97723111 : CertifiedPrime :=
  certify 97723111 (3) [p_2, p_3, p_5, p_659, p_4943] (by decide)

def p_97891237 : CertifiedPrime :=
  certify 97891237 (2) [p_2, p_2, p_3, p_3, p_17, p_17, p_97, p_97] (by decide)

def p_97930817 : CertifiedPrime :=
  certify 97930817 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1237, p_1237] (by decide)

def p_97960507 : CertifiedPrime :=
  certify 97960507 (2) [p_2, p_3, p_7, p_7, p_101, p_3299] (by decide)

def p_98000101 : CertifiedPrime :=
  certify 98000101 (17) [p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_19, p_521] (by decide)

def p_98010001 : CertifiedPrime :=
  certify 98010001 (14) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_11, p_11] (by decide)

def p_98089217 : CertifiedPrime :=
  certify 98089217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_619, p_619] (by decide)

def p_98178373 : CertifiedPrime :=
  certify 98178373 (2) [p_2, p_2, p_3, p_3, p_3, p_367, p_2477] (by decide)

def p_98257657 : CertifiedPrime :=
  certify 98257657 (15) [p_2, p_2, p_2, p_3, p_7, p_23, p_59, p_431] (by decide)

def p_98376643 : CertifiedPrime :=
  certify 98376643 (7) [p_2, p_3, p_3, p_7, p_13, p_19, p_29, p_109] (by decide)

def p_98495701 : CertifiedPrime :=
  certify 98495701 (6) [p_2, p_2, p_3, p_5, p_5, p_397, p_827] (by decide)

def p_98952757 : CertifiedPrime :=
  certify 98952757 (5) [p_2, p_2, p_3, p_7, p_7, p_7, p_29, p_829] (by decide)

def p_99012451 : CertifiedPrime :=
  certify 99012451 (13) [p_2, p_3, p_5, p_5, p_31, p_107, p_199] (by decide)

def p_99121937 : CertifiedPrime :=
  certify 99121937 (3) [p_2, p_2, p_2, p_2, p_19, p_19, p_131, p_131] (by decide)

def p_99151807 : CertifiedPrime :=
  certify 99151807 (3) [p_2, p_3, p_13, p_383, p_3319] (by decide)

def p_99211561 : CertifiedPrime :=
  certify 99211561 (13) [p_2, p_2, p_2, p_3, p_5, p_7, p_83, p_1423] (by decide)

def p_99251407 : CertifiedPrime :=
  certify 99251407 (6) [p_2, p_3, p_3, p_3, p_3, p_3, p_17, p_41, p_293] (by decide)

def p_99331123 : CertifiedPrime :=
  certify 99331123 (3) [p_2, p_3, p_11, p_151, p_9967] (by decide)

def p_99400901 : CertifiedPrime :=
  certify 99400901 (2) [p_2, p_2, p_5, p_5, p_997, p_997] (by decide)

def p_99430813 : CertifiedPrime :=
  certify 99430813 (7) [p_2, p_2, p_3, p_3, p_13, p_13, p_59, p_277] (by decide)

def p_99510601 : CertifiedPrime :=
  certify 99510601 (13) [p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_19, p_29, p_43] (by decide)

def p_99600401 : CertifiedPrime :=
  certify 99600401 (7) [p_2, p_2, p_2, p_2, p_5, p_5, p_499, p_499] (by decide)

def p_99720197 : CertifiedPrime :=
  certify 99720197 (2) [p_2, p_2, p_4993, p_4993] (by decide)

def p_99790111 : CertifiedPrime :=
  certify 99790111 (7) [p_2, p_3, p_3, p_3, p_5, p_7, p_37, p_1427] (by decide)

def p_99800101 : CertifiedPrime :=
  certify 99800101 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_37, p_37] (by decide)

def p_99930013 : CertifiedPrime :=
  certify 99930013 (5) [p_2, p_2, p_3, p_7, p_7, p_13, p_17, p_769] (by decide)

def p_99990001 : CertifiedPrime :=
  certify 99990001 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_11, p_101] (by decide)

def p_100120037 : CertifiedPrime :=
  certify 100120037 (2) [p_2, p_2, p_5003, p_5003] (by decide)

def p_100170073 : CertifiedPrime :=
  certify 100170073 (19) [p_2, p_2, p_2, p_3, p_3, p_139, p_10009] (by decide)

def p_100230133 : CertifiedPrime :=
  certify 100230133 (2) [p_2, p_2, p_3, p_47, p_71, p_2503] (by decide)

def p_100280197 : CertifiedPrime :=
  certify 100280197 (2) [p_2, p_2, p_3, p_3, p_1669, p_1669] (by decide)

def p_100320257 : CertifiedPrime :=
  certify 100320257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_313, p_313] (by decide)

def p_100480577 : CertifiedPrime :=
  certify 100480577 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_179, p_179] (by decide)

def p_100590871 : CertifiedPrime :=
  certify 100590871 (3) [p_2, p_3, p_5, p_17, p_59, p_3343] (by decide)

def p_100691191 : CertifiedPrime :=
  certify 100691191 (6) [p_2, p_3, p_3, p_5, p_29, p_173, p_223] (by decide)

def p_101002501 : CertifiedPrime :=
  certify 101002501 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_67, p_67] (by decide)

def p_101123137 : CertifiedPrime :=
  certify 101123137 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_419, p_419] (by decide)

def p_101193541 : CertifiedPrime :=
  certify 101193541 (2) [p_2, p_2, p_3, p_5, p_7, p_479, p_503] (by decide)

def p_101294161 : CertifiedPrime :=
  certify 101294161 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_11, p_17, p_37, p_61] (by decide)

def p_101485477 : CertifiedPrime :=
  certify 101485477 (5) [p_2, p_2, p_3, p_3, p_23, p_23, p_73, p_73] (by decide)

def p_101687057 : CertifiedPrime :=
  certify 101687057 (3) [p_2, p_2, p_2, p_2, p_2521, p_2521] (by decide)

def p_101727397 : CertifiedPrime :=
  certify 101727397 (2) [p_2, p_2, p_3, p_3, p_41, p_41, p_41, p_41] (by decide)

def p_101737483 : CertifiedPrime :=
  certify 101737483 (3) [p_2, p_3, p_7, p_11, p_41, p_41, p_131] (by decide)

def p_101858557 : CertifiedPrime :=
  certify 101858557 (5) [p_2, p_2, p_3, p_29, p_29, p_10093] (by decide)

def p_102161557 : CertifiedPrime :=
  certify 102161557 (2) [p_2, p_2, p_3, p_3, p_7, p_19, p_19, p_1123] (by decide)

def p_102201991 : CertifiedPrime :=
  certify 102201991 (6) [p_2, p_3, p_5, p_11, p_337, p_919] (by decide)

def p_102333457 : CertifiedPrime :=
  certify 102333457 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_281, p_281] (by decide)

def p_102343573 : CertifiedPrime :=
  certify 102343573 (5) [p_2, p_2, p_3, p_3, p_67, p_151, p_281] (by decide)

def p_102444763 : CertifiedPrime :=
  certify 102444763 (12) [p_2, p_3, p_7, p_29, p_241, p_349] (by decide)

def p_102465007 : CertifiedPrime :=
  certify 102465007 (5) [p_2, p_3, p_7, p_53, p_191, p_241] (by decide)

def p_102525751 : CertifiedPrime :=
  certify 102525751 (6) [p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_61, p_83] (by decide)

def p_102535877 : CertifiedPrime :=
  certify 102535877 (2) [p_2, p_2, p_61, p_61, p_83, p_83] (by decide)

def p_102586513 : CertifiedPrime :=
  certify 102586513 (11) [p_2, p_2, p_2, p_2, p_3, p_7, p_211, p_1447] (by decide)

def p_102616901 : CertifiedPrime :=
  certify 102616901 (2) [p_2, p_2, p_5, p_5, p_1013, p_1013] (by decide)

def p_103022501 : CertifiedPrime :=
  certify 103022501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_29, p_29] (by decide)

def p_103052953 : CertifiedPrime :=
  certify 103052953 (17) [p_2, p_2, p_2, p_3, p_3, p_3, p_47, p_10151] (by decide)

def p_103195123 : CertifiedPrime :=
  certify 103195123 (3) [p_2, p_3, p_1693, p_10159] (by decide)

def p_103225601 : CertifiedPrime :=
  certify 103225601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_127, p_127] (by decide)

def p_103235761 : CertifiedPrime :=
  certify 103235761 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_127, p_1129] (by decide)

def p_103347557 : CertifiedPrime :=
  certify 103347557 (3) [p_2, p_2, p_13, p_13, p_17, p_17, p_23, p_23] (by decide)

def p_104029801 : CertifiedPrime :=
  certify 104029801 (19) [p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_17, p_31, p_47] (by decide)

def p_104091007 : CertifiedPrime :=
  certify 104091007 (3) [p_2, p_3, p_19, p_179, p_5101] (by decide)

def p_104152231 : CertifiedPrime :=
  certify 104152231 (15) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_7, p_13, p_157] (by decide)

def p_104172643 : CertifiedPrime :=
  certify 104172643 (20) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_59, p_173] (by decide)

def p_104295157 : CertifiedPrime :=
  certify 104295157 (5) [p_2, p_2, p_3, p_7, p_23, p_37, p_1459] (by decide)

def p_104366657 : CertifiedPrime :=
  certify 104366657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1277, p_1277] (by decide)

def p_104519953 : CertifiedPrime :=
  certify 104519953 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_71, p_10223] (by decide)

def p_104581303 : CertifiedPrime :=
  certify 104581303 (3) [p_2, p_3, p_7, p_487, p_5113] (by decide)

def p_104847361 : CertifiedPrime :=
  certify 104847361 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_3413] (by decide)

def p_104857601 : CertifiedPrime :=
  certify 104857601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5] (by decide)

def p_104908807 : CertifiedPrime :=
  certify 104908807 (5) [p_2, p_3, p_3, p_569, p_10243] (by decide)

def p_104949781 : CertifiedPrime :=
  certify 104949781 (2) [p_2, p_2, p_3, p_5, p_13, p_197, p_683] (by decide)

def p_104980517 : CertifiedPrime :=
  certify 104980517 (2) [p_2, p_2, p_47, p_47, p_109, p_109] (by decide)

def p_105093253 : CertifiedPrime :=
  certify 105093253 (2) [p_2, p_2, p_3, p_3, p_11, p_17, p_67, p_233] (by decide)

def p_105185537 : CertifiedPrime :=
  certify 105185537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_641, p_641] (by decide)

def p_105339433 : CertifiedPrime :=
  certify 105339433 (7) [p_2, p_2, p_2, p_3, p_11, p_311, p_1283] (by decide)

def p_105380491 : CertifiedPrime :=
  certify 105380491 (3) [p_2, p_3, p_5, p_29, p_59, p_2053] (by decide)

def p_105472901 : CertifiedPrime :=
  certify 105472901 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_79, p_79] (by decide)

def p_105524257 : CertifiedPrime :=
  certify 105524257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_107, p_10273] (by decide)

def p_105596177 : CertifiedPrime :=
  certify 105596177 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_367, p_367] (by decide)

def p_105760657 : CertifiedPrime :=
  certify 105760657 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_857, p_857] (by decide)

def p_105873811 : CertifiedPrime :=
  certify 105873811 (15) [p_2, p_3, p_5, p_7, p_7, p_7, p_10289] (by decide)

def p_105966437 : CertifiedPrime :=
  certify 105966437 (2) [p_2, p_2, p_5147, p_5147] (by decide)

def p_106059103 : CertifiedPrime :=
  certify 106059103 (3) [p_2, p_3, p_19, p_271, p_3433] (by decide)

def p_106141507 : CertifiedPrime :=
  certify 106141507 (3) [p_2, p_3, p_17, p_101, p_10303] (by decide)

def p_106182721 : CertifiedPrime :=
  certify 106182721 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_23, p_229] (by decide)

def p_106244557 : CertifiedPrime :=
  certify 106244557 (5) [p_2, p_2, p_3, p_11, p_859, p_937] (by decide)

def p_106327033 : CertifiedPrime :=
  certify 106327033 (5) [p_2, p_2, p_2, p_3, p_7, p_491, p_1289] (by decide)

def p_106388911 : CertifiedPrime :=
  certify 106388911 (15) [p_2, p_3, p_3, p_3, p_5, p_191, p_2063] (by decide)

def p_106430173 : CertifiedPrime :=
  certify 106430173 (2) [p_2, p_2, p_3, p_19, p_181, p_2579] (by decide)

def p_106492081 : CertifiedPrime :=
  certify 106492081 (13) [p_2, p_2, p_2, p_2, p_3, p_5, p_17, p_43, p_607] (by decide)

def p_106574653 : CertifiedPrime :=
  certify 106574653 (2) [p_2, p_2, p_3, p_3, p_29, p_31, p_37, p_89] (by decide)

def p_106584977 : CertifiedPrime :=
  certify 106584977 (3) [p_2, p_2, p_2, p_2, p_29, p_29, p_89, p_89] (by decide)

def p_106626277 : CertifiedPrime :=
  certify 106626277 (2) [p_2, p_2, p_3, p_3, p_1721, p_1721] (by decide)

def p_106760557 : CertifiedPrime :=
  certify 106760557 (5) [p_2, p_2, p_3, p_3, p_7, p_41, p_10333] (by decide)

def p_107122501 : CertifiedPrime :=
  certify 107122501 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_23, p_23] (by decide)

def p_107298523 : CertifiedPrime :=
  certify 107298523 (2) [p_2, p_3, p_3, p_1151, p_5179] (by decide)

def p_107329601 : CertifiedPrime :=
  certify 107329601 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_37, p_37] (by decide)

def p_107505793 : CertifiedPrime :=
  certify 107505793 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_10369] (by decide)

def p_107661377 : CertifiedPrime :=
  certify 107661377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1297, p_1297] (by decide)

def p_107827457 : CertifiedPrime :=
  certify 107827457 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_59, p_59] (by decide)

def p_107920933 : CertifiedPrime :=
  certify 107920933 (2) [p_2, p_2, p_3, p_7, p_7, p_53, p_3463] (by decide)

def p_108128803 : CertifiedPrime :=
  certify 108128803 (3) [p_2, p_3, p_1733, p_10399] (by decide)

def p_108191203 : CertifiedPrime :=
  certify 108191203 (2) [p_2, p_3, p_7, p_743, p_3467] (by decide)

def p_108451397 : CertifiedPrime :=
  certify 108451397 (2) [p_2, p_2, p_41, p_41, p_127, p_127] (by decide)

def p_108482641 : CertifiedPrime :=
  certify 108482641 (17) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_31, p_2083] (by decide)

def p_108493057 : CertifiedPrime :=
  certify 108493057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_31, p_31] (by decide)

def p_108659777 : CertifiedPrime :=
  certify 108659777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1303, p_1303] (by decide)

def p_108701477 : CertifiedPrime :=
  certify 108701477 (2) [p_2, p_2, p_13, p_13, p_401, p_401] (by decide)

def p_108753613 : CertifiedPrime :=
  certify 108753613 (6) [p_2, p_2, p_3, p_11, p_79, p_10429] (by decide)

def p_108784901 : CertifiedPrime :=
  certify 108784901 (3) [p_2, p_2, p_5, p_5, p_7, p_7, p_149, p_149] (by decide)

def p_108816193 : CertifiedPrime :=
  certify 108816193 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_19, p_61, p_163] (by decide)

def p_108920533 : CertifiedPrime :=
  certify 108920533 (2) [p_2, p_2, p_3, p_7, p_7, p_71, p_2609] (by decide)

def p_108941407 : CertifiedPrime :=
  certify 108941407 (3) [p_2, p_3, p_7, p_7, p_17, p_71, p_307] (by decide)

def p_109066693 : CertifiedPrime :=
  certify 109066693 (2) [p_2, p_2, p_3, p_7, p_59, p_59, p_373] (by decide)

def p_109108471 : CertifiedPrime :=
  certify 109108471 (3) [p_2, p_3, p_5, p_1741, p_2089] (by decide)

def p_109547623 : CertifiedPrime :=
  certify 109547623 (6) [p_2, p_3, p_3, p_1163, p_5233] (by decide)

def p_109694203 : CertifiedPrime :=
  certify 109694203 (2) [p_2, p_3, p_3491, p_5237] (by decide)

def p_109819921 : CertifiedPrime :=
  certify 109819921 (17) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_131, p_499] (by decide)

def p_109882807 : CertifiedPrime :=
  certify 109882807 (3) [p_2, p_3, p_11, p_953, p_1747] (by decide)

def p_110040101 : CertifiedPrime :=
  certify 110040101 (2) [p_2, p_2, p_5, p_5, p_1049, p_1049] (by decide)

def p_110071573 : CertifiedPrime :=
  certify 110071573 (2) [p_2, p_2, p_3, p_13, p_43, p_61, p_269] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
