import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk011

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_981091 : CertifiedPrime :=
  certify 981091 (14) [p_2, p_3, p_3, p_5, p_11, p_991] (by decide)

def p_985057 : CertifiedPrime :=
  certify 985057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_31, p_331] (by decide)

def p_987043 : CertifiedPrime :=
  certify 987043 (2) [p_2, p_3, p_7, p_71, p_331] (by decide)

def p_1003003 : CertifiedPrime :=
  certify 1003003 (7) [p_2, p_3, p_7, p_11, p_13, p_167] (by decide)

def p_1005007 : CertifiedPrime :=
  certify 1005007 (3) [p_2, p_3, p_17, p_59, p_167] (by decide)

def p_1008017 : CertifiedPrime :=
  certify 1008017 (3) [p_2, p_2, p_2, p_2, p_251, p_251] (by decide)

def p_1015057 : CertifiedPrime :=
  certify 1015057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_19, p_53] (by decide)

def p_1020101 : CertifiedPrime :=
  certify 1020101 (2) [p_2, p_2, p_5, p_5, p_101, p_101] (by decide)

def p_1023133 : CertifiedPrime :=
  certify 1023133 (2) [p_2, p_2, p_3, p_11, p_23, p_337] (by decide)

def p_1033273 : CertifiedPrime :=
  certify 1033273 (10) [p_2, p_2, p_2, p_3, p_3, p_113, p_127] (by decide)

def p_1041421 : CertifiedPrime :=
  certify 1041421 (11) [p_2, p_2, p_3, p_5, p_17, p_1021] (by decide)

def p_1045507 : CertifiedPrime :=
  certify 1045507 (5) [p_2, p_3, p_7, p_11, p_31, p_73] (by decide)

def p_1059871 : CertifiedPrime :=
  certify 1059871 (3) [p_2, p_3, p_5, p_7, p_7, p_7, p_103] (by decide)

def p_1073297 : CertifiedPrime :=
  certify 1073297 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_37, p_37] (by decide)

def p_1084723 : CertifiedPrime :=
  certify 1084723 (2) [p_2, p_3, p_347, p_521] (by decide)

def p_1110917 : CertifiedPrime :=
  certify 1110917 (3) [p_2, p_2, p_17, p_17, p_31, p_31] (by decide)

def p_1120423 : CertifiedPrime :=
  certify 1120423 (3) [p_2, p_3, p_23, p_23, p_353] (by decide)

def p_1123601 : CertifiedPrime :=
  certify 1123601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_53, p_53] (by decide)

def p_1135291 : CertifiedPrime :=
  certify 1135291 (2) [p_2, p_3, p_5, p_13, p_41, p_71] (by decide)

def p_1136357 : CertifiedPrime :=
  certify 1136357 (2) [p_2, p_2, p_13, p_13, p_41, p_41] (by decide)

def p_1139557 : CertifiedPrime :=
  certify 1139557 (5) [p_2, p_2, p_3, p_11, p_89, p_97] (by decide)

def p_1144901 : CertifiedPrime :=
  certify 1144901 (2) [p_2, p_2, p_5, p_5, p_107, p_107] (by decide)

def p_1145971 : CertifiedPrime :=
  certify 1145971 (2) [p_2, p_3, p_3, p_5, p_7, p_17, p_107] (by decide)

def p_1154551 : CertifiedPrime :=
  certify 1154551 (3) [p_2, p_3, p_5, p_5, p_43, p_179] (by decide)

def p_1161007 : CertifiedPrime :=
  certify 1161007 (3) [p_2, p_3, p_7, p_7, p_11, p_359] (by decide)

def p_1166401 : CertifiedPrime :=
  certify 1166401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_1191373 : CertifiedPrime :=
  certify 1191373 (6) [p_2, p_2, p_3, p_7, p_13, p_1091] (by decide)

def p_1193557 : CertifiedPrime :=
  certify 1193557 (6) [p_2, p_2, p_3, p_7, p_13, p_1093] (by decide)

def p_1196837 : CertifiedPrime :=
  certify 1196837 (2) [p_2, p_2, p_547, p_547] (by decide)

def p_1201217 : CertifiedPrime :=
  certify 1201217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_137, p_137] (by decide)

def p_1204507 : CertifiedPrime :=
  certify 1204507 (5) [p_2, p_3, p_3, p_61, p_1097] (by decide)

def p_1206703 : CertifiedPrime :=
  certify 1206703 (7) [p_2, p_3, p_3, p_7, p_61, p_157] (by decide)

def p_1223237 : CertifiedPrime :=
  certify 1223237 (2) [p_2, p_2, p_7, p_7, p_79, p_79] (by decide)

def p_1226557 : CertifiedPrime :=
  certify 1226557 (2) [p_2, p_2, p_3, p_3, p_3, p_41, p_277] (by decide)

def p_1246573 : CertifiedPrime :=
  certify 1246573 (5) [p_2, p_2, p_3, p_3, p_31, p_1117] (by decide)

def p_1251043 : CertifiedPrime :=
  certify 1251043 (2) [p_2, p_3, p_13, p_43, p_373] (by decide)

def p_1263377 : CertifiedPrime :=
  certify 1263377 (3) [p_2, p_2, p_2, p_2, p_281, p_281] (by decide)

def p_1266751 : CertifiedPrime :=
  certify 1266751 (12) [p_2, p_3, p_3, p_5, p_5, p_5, p_563] (by decide)

def p_1278031 : CertifiedPrime :=
  certify 1278031 (6) [p_2, p_3, p_5, p_13, p_29, p_113] (by decide)

def p_1284823 : CertifiedPrime :=
  certify 1284823 (5) [p_2, p_3, p_3, p_3, p_3, p_7, p_11, p_103] (by decide)

def p_1299601 : CertifiedPrime :=
  certify 1299601 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_19] (by decide)

def p_1305307 : CertifiedPrime :=
  certify 1305307 (2) [p_2, p_3, p_3, p_127, p_571] (by decide)

def p_1308737 : CertifiedPrime :=
  certify 1308737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_13, p_13] (by decide)

def p_1313317 : CertifiedPrime :=
  certify 1313317 (2) [p_2, p_2, p_3, p_3, p_191, p_191] (by decide)

def p_1314463 : CertifiedPrime :=
  certify 1314463 (3) [p_2, p_3, p_31, p_37, p_191] (by decide)

def p_1319053 : CertifiedPrime :=
  certify 1319053 (2) [p_2, p_2, p_3, p_7, p_41, p_383] (by decide)

def p_1321351 : CertifiedPrime :=
  certify 1321351 (3) [p_2, p_3, p_5, p_5, p_23, p_383] (by decide)

def p_1322501 : CertifiedPrime :=
  certify 1322501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_23, p_23] (by decide)

def p_1336337 : CertifiedPrime :=
  certify 1336337 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_17, p_17] (by decide)

def p_1353733 : CertifiedPrime :=
  certify 1353733 (5) [p_2, p_2, p_3, p_97, p_1163] (by decide)

def p_1378277 : CertifiedPrime :=
  certify 1378277 (2) [p_2, p_2, p_587, p_587] (by decide)

def p_1382977 : CertifiedPrime :=
  certify 1382977 (17) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_7, p_7] (by decide)

def p_1395943 : CertifiedPrime :=
  certify 1395943 (3) [p_2, p_3, p_197, p_1181] (by decide)

def p_1398307 : CertifiedPrime :=
  certify 1398307 (3) [p_2, p_3, p_7, p_13, p_13, p_197] (by decide)

def p_1401857 : CertifiedPrime :=
  certify 1401857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_37, p_37] (by decide)

def p_1419673 : CertifiedPrime :=
  certify 1419673 (5) [p_2, p_2, p_2, p_3, p_149, p_397] (by decide)

def p_1424443 : CertifiedPrime :=
  certify 1424443 (3) [p_2, p_3, p_199, p_1193] (by decide)

def p_1431613 : CertifiedPrime :=
  certify 1431613 (2) [p_2, p_2, p_3, p_3, p_7, p_13, p_19, p_23] (by decide)

def p_1441201 : CertifiedPrime :=
  certify 1441201 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_1201] (by decide)

def p_1446007 : CertifiedPrime :=
  certify 1446007 (3) [p_2, p_3, p_401, p_601] (by decide)

def p_1462891 : CertifiedPrime :=
  certify 1462891 (2) [p_2, p_3, p_5, p_11, p_11, p_13, p_31] (by decide)

def p_1464101 : CertifiedPrime :=
  certify 1464101 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_11, p_11] (by decide)

def p_1482307 : CertifiedPrime :=
  certify 1482307 (3) [p_2, p_3, p_7, p_29, p_1217] (by decide)

def p_1492063 : CertifiedPrime :=
  certify 1492063 (3) [p_2, p_3, p_11, p_13, p_37, p_47] (by decide)

def p_1514131 : CertifiedPrime :=
  certify 1514131 (3) [p_2, p_3, p_5, p_41, p_1231] (by decide)

def p_1547537 : CertifiedPrime :=
  certify 1547537 (3) [p_2, p_2, p_2, p_2, p_311, p_311] (by decide)

def p_1552517 : CertifiedPrime :=
  certify 1552517 (2) [p_2, p_2, p_7, p_7, p_89, p_89] (by decide)

def p_1573771 : CertifiedPrime :=
  certify 1573771 (3) [p_2, p_3, p_5, p_11, p_19, p_251] (by decide)

def p_1578793 : CertifiedPrime :=
  certify 1578793 (7) [p_2, p_2, p_2, p_3, p_157, p_419] (by decide)

def p_1588861 : CertifiedPrime :=
  certify 1588861 (17) [p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_97] (by decide)

def p_1596433 : CertifiedPrime :=
  certify 1596433 (7) [p_2, p_2, p_2, p_2, p_3, p_79, p_421] (by decide)

def p_1619257 : CertifiedPrime :=
  certify 1619257 (5) [p_2, p_2, p_2, p_3, p_19, p_53, p_67] (by decide)

def p_1623077 : CertifiedPrime :=
  certify 1623077 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_13, p_13] (by decide)

def p_1624351 : CertifiedPrime :=
  certify 1624351 (3) [p_2, p_3, p_5, p_5, p_7, p_7, p_13, p_17] (by decide)

def p_1626901 : CertifiedPrime :=
  certify 1626901 (2) [p_2, p_2, p_3, p_5, p_5, p_11, p_17, p_29] (by decide)

def p_1628177 : CertifiedPrime :=
  certify 1628177 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_29, p_29] (by decide)

def p_1642243 : CertifiedPrime :=
  certify 1642243 (2) [p_2, p_3, p_7, p_61, p_641] (by decide)

def p_1664101 : CertifiedPrime :=
  certify 1664101 (11) [p_2, p_2, p_3, p_3, p_5, p_5, p_43, p_43] (by decide)

def p_1674437 : CertifiedPrime :=
  certify 1674437 (2) [p_2, p_2, p_647, p_647] (by decide)

def p_1678321 : CertifiedPrime :=
  certify 1678321 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_37] (by decide)

def p_1705637 : CertifiedPrime :=
  certify 1705637 (2) [p_2, p_2, p_653, p_653] (by decide)

def p_1712173 : CertifiedPrime :=
  certify 1712173 (6) [p_2, p_2, p_3, p_7, p_11, p_17, p_109] (by decide)

def p_1726597 : CertifiedPrime :=
  certify 1726597 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_73, p_73] (by decide)

def p_1727911 : CertifiedPrime :=
  certify 1727911 (11) [p_2, p_3, p_3, p_5, p_73, p_263] (by decide)

def p_1731857 : CertifiedPrime :=
  certify 1731857 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_47, p_47] (by decide)

def p_1735807 : CertifiedPrime :=
  certify 1735807 (3) [p_2, p_3, p_439, p_659] (by decide)

def p_1742401 : CertifiedPrime :=
  certify 1742401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_11] (by decide)

def p_1751653 : CertifiedPrime :=
  certify 1751653 (2) [p_2, p_2, p_3, p_3, p_3, p_7, p_7, p_331] (by decide)

def p_1752977 : CertifiedPrime :=
  certify 1752977 (3) [p_2, p_2, p_2, p_2, p_331, p_331] (by decide)

def p_1772893 : CertifiedPrime :=
  certify 1772893 (17) [p_2, p_2, p_3, p_3, p_11, p_11, p_11, p_37] (by decide)

def p_1795601 : CertifiedPrime :=
  certify 1795601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_67, p_67] (by decide)

def p_1796941 : CertifiedPrime :=
  certify 1796941 (6) [p_2, p_2, p_3, p_3, p_5, p_67, p_149] (by decide)

def p_1804993 : CertifiedPrime :=
  certify 1804993 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_17, p_79] (by decide)

def p_1821151 : CertifiedPrime :=
  certify 1821151 (21) [p_2, p_3, p_3, p_3, p_5, p_5, p_19, p_71] (by decide)

def p_1822501 : CertifiedPrime :=
  certify 1822501 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5] (by decide)

def p_1829257 : CertifiedPrime :=
  certify 1829257 (5) [p_2, p_2, p_2, p_3, p_11, p_13, p_13, p_41] (by decide)

def p_1833317 : CertifiedPrime :=
  certify 1833317 (2) [p_2, p_2, p_677, p_677] (by decide)

def p_1848241 : CertifiedPrime :=
  certify 1848241 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_17, p_151] (by decide)

def p_1861861 : CertifiedPrime :=
  certify 1861861 (2) [p_2, p_2, p_3, p_5, p_7, p_11, p_13, p_31] (by decide)

def p_1864591 : CertifiedPrime :=
  certify 1864591 (3) [p_2, p_3, p_5, p_7, p_13, p_683] (by decide)

def p_1865957 : CertifiedPrime :=
  certify 1865957 (2) [p_2, p_2, p_683, p_683] (by decide)

def p_1886503 : CertifiedPrime :=
  certify 1886503 (3) [p_2, p_3, p_229, p_1373] (by decide)

def p_1887877 : CertifiedPrime :=
  certify 1887877 (2) [p_2, p_2, p_3, p_3, p_229, p_229] (by decide)

def p_1893377 : CertifiedPrime :=
  certify 1893377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_43, p_43] (by decide)

def p_1905781 : CertifiedPrime :=
  certify 1905781 (6) [p_2, p_2, p_3, p_5, p_23, p_1381] (by decide)

def p_1922383 : CertifiedPrime :=
  certify 1922383 (5) [p_2, p_3, p_3, p_7, p_11, p_19, p_73] (by decide)

def p_1939057 : CertifiedPrime :=
  certify 1939057 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_29, p_199] (by decide)

def p_1943237 : CertifiedPrime :=
  certify 1943237 (2) [p_2, p_2, p_17, p_17, p_41, p_41] (by decide)

def p_1964203 : CertifiedPrime :=
  certify 1964203 (2) [p_2, p_3, p_467, p_701] (by decide)

def p_1976837 : CertifiedPrime :=
  certify 1976837 (2) [p_2, p_2, p_19, p_19, p_37, p_37] (by decide)

def p_1988101 : CertifiedPrime :=
  certify 1988101 (13) [p_2, p_2, p_3, p_3, p_5, p_5, p_47, p_47] (by decide)

def p_2003641 : CertifiedPrime :=
  certify 2003641 (11) [p_2, p_2, p_2, p_3, p_5, p_59, p_283] (by decide)

def p_2005057 : CertifiedPrime :=
  certify 2005057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_59, p_59] (by decide)

def p_2016401 : CertifiedPrime :=
  certify 2016401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_71, p_71] (by decide)

def p_2020663 : CertifiedPrime :=
  certify 2020663 (7) [p_2, p_3, p_3, p_7, p_7, p_29, p_79] (by decide)

def p_2023507 : CertifiedPrime :=
  certify 2023507 (5) [p_2, p_3, p_3, p_79, p_1423] (by decide)

def p_2037757 : CertifiedPrime :=
  certify 2037757 (13) [p_2, p_2, p_3, p_7, p_17, p_1427] (by decide)

def p_2044901 : CertifiedPrime :=
  certify 2044901 (7) [p_2, p_2, p_5, p_5, p_11, p_11, p_13, p_13] (by decide)

def p_2056357 : CertifiedPrime :=
  certify 2056357 (2) [p_2, p_2, p_3, p_3, p_239, p_239] (by decide)

def p_2057791 : CertifiedPrime :=
  certify 2057791 (3) [p_2, p_3, p_5, p_7, p_41, p_239] (by decide)

def p_2073601 : CertifiedPrime :=
  certify 2073601 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_2075041 : CertifiedPrime :=
  certify 2075041 (19) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_11, p_131] (by decide)

def p_2080807 : CertifiedPrime :=
  certify 2080807 (3) [p_2, p_3, p_7, p_13, p_37, p_103] (by decide)

def p_2083693 : CertifiedPrime :=
  certify 2083693 (2) [p_2, p_2, p_3, p_13, p_19, p_19, p_37] (by decide)

def p_2098153 : CertifiedPrime :=
  certify 2098153 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_23, p_181] (by decide)

def p_2101051 : CertifiedPrime :=
  certify 2101051 (2) [p_2, p_3, p_3, p_5, p_5, p_7, p_23, p_29] (by decide)

def p_2115571 : CertifiedPrime :=
  certify 2115571 (2) [p_2, p_3, p_5, p_97, p_727] (by decide)

def p_2119937 : CertifiedPrime :=
  certify 2119937 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_13, p_13] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
