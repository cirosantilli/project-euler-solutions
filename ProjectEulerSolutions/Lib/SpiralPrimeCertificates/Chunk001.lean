import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk000

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_727 : CertifiedPrime :=
  certify 727 (5) [p_2, p_3, p_11, p_11] (by decide)

def p_733 : CertifiedPrime :=
  certify 733 (6) [p_2, p_2, p_3, p_61] (by decide)

def p_739 : CertifiedPrime :=
  certify 739 (3) [p_2, p_3, p_3, p_41] (by decide)

def p_743 : CertifiedPrime :=
  certify 743 (5) [p_2, p_7, p_53] (by decide)

def p_751 : CertifiedPrime :=
  certify 751 (3) [p_2, p_3, p_5, p_5, p_5] (by decide)

def p_757 : CertifiedPrime :=
  certify 757 (2) [p_2, p_2, p_3, p_3, p_3, p_7] (by decide)

def p_761 : CertifiedPrime :=
  certify 761 (6) [p_2, p_2, p_2, p_5, p_19] (by decide)

def p_769 : CertifiedPrime :=
  certify 769 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3] (by decide)

def p_773 : CertifiedPrime :=
  certify 773 (2) [p_2, p_2, p_193] (by decide)

def p_787 : CertifiedPrime :=
  certify 787 (2) [p_2, p_3, p_131] (by decide)

def p_797 : CertifiedPrime :=
  certify 797 (2) [p_2, p_2, p_199] (by decide)

def p_809 : CertifiedPrime :=
  certify 809 (3) [p_2, p_2, p_2, p_101] (by decide)

def p_811 : CertifiedPrime :=
  certify 811 (3) [p_2, p_3, p_3, p_3, p_3, p_5] (by decide)

def p_821 : CertifiedPrime :=
  certify 821 (2) [p_2, p_2, p_5, p_41] (by decide)

def p_823 : CertifiedPrime :=
  certify 823 (3) [p_2, p_3, p_137] (by decide)

def p_827 : CertifiedPrime :=
  certify 827 (2) [p_2, p_7, p_59] (by decide)

def p_829 : CertifiedPrime :=
  certify 829 (2) [p_2, p_2, p_3, p_3, p_23] (by decide)

def p_839 : CertifiedPrime :=
  certify 839 (11) [p_2, p_419] (by decide)

def p_853 : CertifiedPrime :=
  certify 853 (2) [p_2, p_2, p_3, p_71] (by decide)

def p_857 : CertifiedPrime :=
  certify 857 (3) [p_2, p_2, p_2, p_107] (by decide)

def p_859 : CertifiedPrime :=
  certify 859 (2) [p_2, p_3, p_11, p_13] (by decide)

def p_863 : CertifiedPrime :=
  certify 863 (5) [p_2, p_431] (by decide)

def p_877 : CertifiedPrime :=
  certify 877 (2) [p_2, p_2, p_3, p_73] (by decide)

def p_881 : CertifiedPrime :=
  certify 881 (3) [p_2, p_2, p_2, p_2, p_5, p_11] (by decide)

def p_883 : CertifiedPrime :=
  certify 883 (2) [p_2, p_3, p_3, p_7, p_7] (by decide)

def p_887 : CertifiedPrime :=
  certify 887 (5) [p_2, p_443] (by decide)

def p_907 : CertifiedPrime :=
  certify 907 (2) [p_2, p_3, p_151] (by decide)

def p_911 : CertifiedPrime :=
  certify 911 (17) [p_2, p_5, p_7, p_13] (by decide)

def p_919 : CertifiedPrime :=
  certify 919 (7) [p_2, p_3, p_3, p_3, p_17] (by decide)

def p_929 : CertifiedPrime :=
  certify 929 (3) [p_2, p_2, p_2, p_2, p_2, p_29] (by decide)

def p_937 : CertifiedPrime :=
  certify 937 (5) [p_2, p_2, p_2, p_3, p_3, p_13] (by decide)

def p_941 : CertifiedPrime :=
  certify 941 (2) [p_2, p_2, p_5, p_47] (by decide)

def p_947 : CertifiedPrime :=
  certify 947 (2) [p_2, p_11, p_43] (by decide)

def p_953 : CertifiedPrime :=
  certify 953 (3) [p_2, p_2, p_2, p_7, p_17] (by decide)

def p_967 : CertifiedPrime :=
  certify 967 (5) [p_2, p_3, p_7, p_23] (by decide)

def p_971 : CertifiedPrime :=
  certify 971 (6) [p_2, p_5, p_97] (by decide)

def p_977 : CertifiedPrime :=
  certify 977 (3) [p_2, p_2, p_2, p_2, p_61] (by decide)

def p_983 : CertifiedPrime :=
  certify 983 (5) [p_2, p_491] (by decide)

def p_991 : CertifiedPrime :=
  certify 991 (6) [p_2, p_3, p_3, p_5, p_11] (by decide)

def p_997 : CertifiedPrime :=
  certify 997 (7) [p_2, p_2, p_3, p_83] (by decide)

def p_1009 : CertifiedPrime :=
  certify 1009 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_7] (by decide)

def p_1013 : CertifiedPrime :=
  certify 1013 (3) [p_2, p_2, p_11, p_23] (by decide)

def p_1019 : CertifiedPrime :=
  certify 1019 (2) [p_2, p_509] (by decide)

def p_1021 : CertifiedPrime :=
  certify 1021 (10) [p_2, p_2, p_3, p_5, p_17] (by decide)

def p_1031 : CertifiedPrime :=
  certify 1031 (14) [p_2, p_5, p_103] (by decide)

def p_1033 : CertifiedPrime :=
  certify 1033 (5) [p_2, p_2, p_2, p_3, p_43] (by decide)

def p_1039 : CertifiedPrime :=
  certify 1039 (3) [p_2, p_3, p_173] (by decide)

def p_1049 : CertifiedPrime :=
  certify 1049 (3) [p_2, p_2, p_2, p_131] (by decide)

def p_1051 : CertifiedPrime :=
  certify 1051 (7) [p_2, p_3, p_5, p_5, p_7] (by decide)

def p_1061 : CertifiedPrime :=
  certify 1061 (2) [p_2, p_2, p_5, p_53] (by decide)

def p_1063 : CertifiedPrime :=
  certify 1063 (3) [p_2, p_3, p_3, p_59] (by decide)

def p_1069 : CertifiedPrime :=
  certify 1069 (6) [p_2, p_2, p_3, p_89] (by decide)

def p_1087 : CertifiedPrime :=
  certify 1087 (3) [p_2, p_3, p_181] (by decide)

def p_1091 : CertifiedPrime :=
  certify 1091 (2) [p_2, p_5, p_109] (by decide)

def p_1093 : CertifiedPrime :=
  certify 1093 (5) [p_2, p_2, p_3, p_7, p_13] (by decide)

def p_1097 : CertifiedPrime :=
  certify 1097 (3) [p_2, p_2, p_2, p_137] (by decide)

def p_1103 : CertifiedPrime :=
  certify 1103 (5) [p_2, p_19, p_29] (by decide)

def p_1109 : CertifiedPrime :=
  certify 1109 (2) [p_2, p_2, p_277] (by decide)

def p_1117 : CertifiedPrime :=
  certify 1117 (2) [p_2, p_2, p_3, p_3, p_31] (by decide)

def p_1123 : CertifiedPrime :=
  certify 1123 (2) [p_2, p_3, p_11, p_17] (by decide)

def p_1129 : CertifiedPrime :=
  certify 1129 (11) [p_2, p_2, p_2, p_3, p_47] (by decide)

def p_1151 : CertifiedPrime :=
  certify 1151 (17) [p_2, p_5, p_5, p_23] (by decide)

def p_1153 : CertifiedPrime :=
  certify 1153 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3] (by decide)

def p_1163 : CertifiedPrime :=
  certify 1163 (5) [p_2, p_7, p_83] (by decide)

def p_1171 : CertifiedPrime :=
  certify 1171 (2) [p_2, p_3, p_3, p_5, p_13] (by decide)

def p_1181 : CertifiedPrime :=
  certify 1181 (7) [p_2, p_2, p_5, p_59] (by decide)

def p_1187 : CertifiedPrime :=
  certify 1187 (2) [p_2, p_593] (by decide)

def p_1193 : CertifiedPrime :=
  certify 1193 (3) [p_2, p_2, p_2, p_149] (by decide)

def p_1201 : CertifiedPrime :=
  certify 1201 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_5] (by decide)

def p_1213 : CertifiedPrime :=
  certify 1213 (2) [p_2, p_2, p_3, p_101] (by decide)

def p_1217 : CertifiedPrime :=
  certify 1217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19] (by decide)

def p_1223 : CertifiedPrime :=
  certify 1223 (5) [p_2, p_13, p_47] (by decide)

def p_1229 : CertifiedPrime :=
  certify 1229 (2) [p_2, p_2, p_307] (by decide)

def p_1231 : CertifiedPrime :=
  certify 1231 (3) [p_2, p_3, p_5, p_41] (by decide)

def p_1237 : CertifiedPrime :=
  certify 1237 (2) [p_2, p_2, p_3, p_103] (by decide)

def p_1249 : CertifiedPrime :=
  certify 1249 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_13] (by decide)

def p_1259 : CertifiedPrime :=
  certify 1259 (2) [p_2, p_17, p_37] (by decide)

def p_1277 : CertifiedPrime :=
  certify 1277 (2) [p_2, p_2, p_11, p_29] (by decide)

def p_1279 : CertifiedPrime :=
  certify 1279 (3) [p_2, p_3, p_3, p_71] (by decide)

def p_1283 : CertifiedPrime :=
  certify 1283 (2) [p_2, p_641] (by decide)

def p_1289 : CertifiedPrime :=
  certify 1289 (6) [p_2, p_2, p_2, p_7, p_23] (by decide)

def p_1291 : CertifiedPrime :=
  certify 1291 (2) [p_2, p_3, p_5, p_43] (by decide)

def p_1297 : CertifiedPrime :=
  certify 1297 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3] (by decide)

def p_1301 : CertifiedPrime :=
  certify 1301 (2) [p_2, p_2, p_5, p_5, p_13] (by decide)

def p_1303 : CertifiedPrime :=
  certify 1303 (6) [p_2, p_3, p_7, p_31] (by decide)

def p_1307 : CertifiedPrime :=
  certify 1307 (2) [p_2, p_653] (by decide)

def p_1319 : CertifiedPrime :=
  certify 1319 (13) [p_2, p_659] (by decide)

def p_1321 : CertifiedPrime :=
  certify 1321 (13) [p_2, p_2, p_2, p_3, p_5, p_11] (by decide)

def p_1327 : CertifiedPrime :=
  certify 1327 (3) [p_2, p_3, p_13, p_17] (by decide)

def p_1361 : CertifiedPrime :=
  certify 1361 (3) [p_2, p_2, p_2, p_2, p_5, p_17] (by decide)

def p_1367 : CertifiedPrime :=
  certify 1367 (5) [p_2, p_683] (by decide)

def p_1373 : CertifiedPrime :=
  certify 1373 (2) [p_2, p_2, p_7, p_7, p_7] (by decide)

def p_1381 : CertifiedPrime :=
  certify 1381 (2) [p_2, p_2, p_3, p_5, p_23] (by decide)

def p_1399 : CertifiedPrime :=
  certify 1399 (13) [p_2, p_3, p_233] (by decide)

def p_1409 : CertifiedPrime :=
  certify 1409 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11] (by decide)

def p_1423 : CertifiedPrime :=
  certify 1423 (3) [p_2, p_3, p_3, p_79] (by decide)

def p_1427 : CertifiedPrime :=
  certify 1427 (2) [p_2, p_23, p_31] (by decide)

def p_1429 : CertifiedPrime :=
  certify 1429 (6) [p_2, p_2, p_3, p_7, p_17] (by decide)

def p_1433 : CertifiedPrime :=
  certify 1433 (3) [p_2, p_2, p_2, p_179] (by decide)

def p_1439 : CertifiedPrime :=
  certify 1439 (7) [p_2, p_719] (by decide)

def p_1447 : CertifiedPrime :=
  certify 1447 (3) [p_2, p_3, p_241] (by decide)

def p_1451 : CertifiedPrime :=
  certify 1451 (2) [p_2, p_5, p_5, p_29] (by decide)

def p_1453 : CertifiedPrime :=
  certify 1453 (2) [p_2, p_2, p_3, p_11, p_11] (by decide)

def p_1459 : CertifiedPrime :=
  certify 1459 (3) [p_2, p_3, p_3, p_3, p_3, p_3, p_3] (by decide)

def p_1471 : CertifiedPrime :=
  certify 1471 (6) [p_2, p_3, p_5, p_7, p_7] (by decide)

def p_1481 : CertifiedPrime :=
  certify 1481 (3) [p_2, p_2, p_2, p_5, p_37] (by decide)

def p_1483 : CertifiedPrime :=
  certify 1483 (2) [p_2, p_3, p_13, p_19] (by decide)

def p_1487 : CertifiedPrime :=
  certify 1487 (5) [p_2, p_743] (by decide)

def p_1489 : CertifiedPrime :=
  certify 1489 (14) [p_2, p_2, p_2, p_2, p_3, p_31] (by decide)

def p_1493 : CertifiedPrime :=
  certify 1493 (2) [p_2, p_2, p_373] (by decide)

def p_1499 : CertifiedPrime :=
  certify 1499 (2) [p_2, p_7, p_107] (by decide)

def p_1511 : CertifiedPrime :=
  certify 1511 (11) [p_2, p_5, p_151] (by decide)

def p_1523 : CertifiedPrime :=
  certify 1523 (2) [p_2, p_761] (by decide)

def p_1531 : CertifiedPrime :=
  certify 1531 (2) [p_2, p_3, p_3, p_5, p_17] (by decide)

def p_1543 : CertifiedPrime :=
  certify 1543 (5) [p_2, p_3, p_257] (by decide)

def p_1549 : CertifiedPrime :=
  certify 1549 (2) [p_2, p_2, p_3, p_3, p_43] (by decide)

def p_1553 : CertifiedPrime :=
  certify 1553 (3) [p_2, p_2, p_2, p_2, p_97] (by decide)

def p_1559 : CertifiedPrime :=
  certify 1559 (19) [p_2, p_19, p_41] (by decide)

def p_1567 : CertifiedPrime :=
  certify 1567 (3) [p_2, p_3, p_3, p_3, p_29] (by decide)

def p_1571 : CertifiedPrime :=
  certify 1571 (2) [p_2, p_5, p_157] (by decide)

def p_1579 : CertifiedPrime :=
  certify 1579 (3) [p_2, p_3, p_263] (by decide)

def p_1583 : CertifiedPrime :=
  certify 1583 (5) [p_2, p_7, p_113] (by decide)

def p_1597 : CertifiedPrime :=
  certify 1597 (11) [p_2, p_2, p_3, p_7, p_19] (by decide)

def p_1601 : CertifiedPrime :=
  certify 1601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5] (by decide)

def p_1607 : CertifiedPrime :=
  certify 1607 (5) [p_2, p_11, p_73] (by decide)

def p_1609 : CertifiedPrime :=
  certify 1609 (7) [p_2, p_2, p_2, p_3, p_67] (by decide)

def p_1613 : CertifiedPrime :=
  certify 1613 (3) [p_2, p_2, p_13, p_31] (by decide)

def p_1619 : CertifiedPrime :=
  certify 1619 (2) [p_2, p_809] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
