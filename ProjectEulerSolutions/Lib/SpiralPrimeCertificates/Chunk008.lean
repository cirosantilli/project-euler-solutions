import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk007

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_11621 : CertifiedPrime :=
  certify 11621 (2) [p_2, p_2, p_5, p_7, p_83] (by decide)

def p_11633 : CertifiedPrime :=
  certify 11633 (3) [p_2, p_2, p_2, p_2, p_727] (by decide)

def p_11677 : CertifiedPrime :=
  certify 11677 (2) [p_2, p_2, p_3, p_7, p_139] (by decide)

def p_11689 : CertifiedPrime :=
  certify 11689 (7) [p_2, p_2, p_2, p_3, p_487] (by decide)

def p_11701 : CertifiedPrime :=
  certify 11701 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_13] (by decide)

def p_11717 : CertifiedPrime :=
  certify 11717 (2) [p_2, p_2, p_29, p_101] (by decide)

def p_11777 : CertifiedPrime :=
  certify 11777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_23] (by decide)

def p_11789 : CertifiedPrime :=
  certify 11789 (2) [p_2, p_2, p_7, p_421] (by decide)

def p_11821 : CertifiedPrime :=
  certify 11821 (2) [p_2, p_2, p_3, p_5, p_197] (by decide)

def p_11839 : CertifiedPrime :=
  certify 11839 (3) [p_2, p_3, p_1973] (by decide)

def p_11863 : CertifiedPrime :=
  certify 11863 (3) [p_2, p_3, p_3, p_659] (by decide)

def p_11903 : CertifiedPrime :=
  certify 11903 (5) [p_2, p_11, p_541] (by decide)

def p_11923 : CertifiedPrime :=
  certify 11923 (5) [p_2, p_3, p_1987] (by decide)

def p_11927 : CertifiedPrime :=
  certify 11927 (5) [p_2, p_67, p_89] (by decide)

def p_11933 : CertifiedPrime :=
  certify 11933 (2) [p_2, p_2, p_19, p_157] (by decide)

def p_11987 : CertifiedPrime :=
  certify 11987 (2) [p_2, p_13, p_461] (by decide)

def p_12007 : CertifiedPrime :=
  certify 12007 (13) [p_2, p_3, p_3, p_23, p_29] (by decide)

def p_12037 : CertifiedPrime :=
  certify 12037 (5) [p_2, p_2, p_3, p_17, p_59] (by decide)

def p_12041 : CertifiedPrime :=
  certify 12041 (3) [p_2, p_2, p_2, p_5, p_7, p_43] (by decide)

def p_12043 : CertifiedPrime :=
  certify 12043 (2) [p_2, p_3, p_3, p_3, p_223] (by decide)

def p_12073 : CertifiedPrime :=
  certify 12073 (7) [p_2, p_2, p_2, p_3, p_503] (by decide)

def p_12101 : CertifiedPrime :=
  certify 12101 (3) [p_2, p_2, p_5, p_5, p_11, p_11] (by decide)

def p_12109 : CertifiedPrime :=
  certify 12109 (6) [p_2, p_2, p_3, p_1009] (by decide)

def p_12113 : CertifiedPrime :=
  certify 12113 (3) [p_2, p_2, p_2, p_2, p_757] (by decide)

def p_12119 : CertifiedPrime :=
  certify 12119 (7) [p_2, p_73, p_83] (by decide)

def p_12143 : CertifiedPrime :=
  certify 12143 (10) [p_2, p_13, p_467] (by decide)

def p_12149 : CertifiedPrime :=
  certify 12149 (2) [p_2, p_2, p_3037] (by decide)

def p_12161 : CertifiedPrime :=
  certify 12161 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_19] (by decide)

def p_12163 : CertifiedPrime :=
  certify 12163 (5) [p_2, p_3, p_2027] (by decide)

def p_12203 : CertifiedPrime :=
  certify 12203 (2) [p_2, p_6101] (by decide)

def p_12211 : CertifiedPrime :=
  certify 12211 (2) [p_2, p_3, p_5, p_11, p_37] (by decide)

def p_12227 : CertifiedPrime :=
  certify 12227 (2) [p_2, p_6113] (by decide)

def p_12251 : CertifiedPrime :=
  certify 12251 (2) [p_2, p_5, p_5, p_5, p_7, p_7] (by decide)

def p_12269 : CertifiedPrime :=
  certify 12269 (2) [p_2, p_2, p_3067] (by decide)

def p_12277 : CertifiedPrime :=
  certify 12277 (2) [p_2, p_2, p_3, p_3, p_11, p_31] (by decide)

def p_12281 : CertifiedPrime :=
  certify 12281 (3) [p_2, p_2, p_2, p_5, p_307] (by decide)

def p_12329 : CertifiedPrime :=
  certify 12329 (3) [p_2, p_2, p_2, p_23, p_67] (by decide)

def p_12379 : CertifiedPrime :=
  certify 12379 (2) [p_2, p_3, p_2063] (by decide)

def p_12433 : CertifiedPrime :=
  certify 12433 (13) [p_2, p_2, p_2, p_2, p_3, p_7, p_37] (by decide)

def p_12457 : CertifiedPrime :=
  certify 12457 (10) [p_2, p_2, p_2, p_3, p_3, p_173] (by decide)

def p_12473 : CertifiedPrime :=
  certify 12473 (3) [p_2, p_2, p_2, p_1559] (by decide)

def p_12487 : CertifiedPrime :=
  certify 12487 (3) [p_2, p_3, p_2081] (by decide)

def p_12497 : CertifiedPrime :=
  certify 12497 (3) [p_2, p_2, p_2, p_2, p_11, p_71] (by decide)

def p_12541 : CertifiedPrime :=
  certify 12541 (14) [p_2, p_2, p_3, p_5, p_11, p_19] (by decide)

def p_12547 : CertifiedPrime :=
  certify 12547 (2) [p_2, p_3, p_3, p_17, p_41] (by decide)

def p_12569 : CertifiedPrime :=
  certify 12569 (3) [p_2, p_2, p_2, p_1571] (by decide)

def p_12583 : CertifiedPrime :=
  certify 12583 (5) [p_2, p_3, p_3, p_3, p_233] (by decide)

def p_12637 : CertifiedPrime :=
  certify 12637 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_13] (by decide)

def p_12641 : CertifiedPrime :=
  certify 12641 (3) [p_2, p_2, p_2, p_2, p_2, p_5, p_79] (by decide)

def p_12647 : CertifiedPrime :=
  certify 12647 (5) [p_2, p_6323] (by decide)

def p_12659 : CertifiedPrime :=
  certify 12659 (2) [p_2, p_6329] (by decide)

def p_12697 : CertifiedPrime :=
  certify 12697 (7) [p_2, p_2, p_2, p_3, p_23, p_23] (by decide)

def p_12721 : CertifiedPrime :=
  certify 12721 (13) [p_2, p_2, p_2, p_2, p_3, p_5, p_53] (by decide)

def p_12781 : CertifiedPrime :=
  certify 12781 (2) [p_2, p_2, p_3, p_3, p_5, p_71] (by decide)

def p_12791 : CertifiedPrime :=
  certify 12791 (7) [p_2, p_5, p_1279] (by decide)

def p_12799 : CertifiedPrime :=
  certify 12799 (13) [p_2, p_3, p_3, p_3, p_3, p_79] (by decide)

def p_12821 : CertifiedPrime :=
  certify 12821 (2) [p_2, p_2, p_5, p_641] (by decide)

def p_12823 : CertifiedPrime :=
  certify 12823 (3) [p_2, p_3, p_2137] (by decide)

def p_12841 : CertifiedPrime :=
  certify 12841 (21) [p_2, p_2, p_2, p_3, p_5, p_107] (by decide)

def p_12853 : CertifiedPrime :=
  certify 12853 (5) [p_2, p_2, p_3, p_3, p_3, p_7, p_17] (by decide)

def p_12889 : CertifiedPrime :=
  certify 12889 (13) [p_2, p_2, p_2, p_3, p_3, p_179] (by decide)

def p_12893 : CertifiedPrime :=
  certify 12893 (3) [p_2, p_2, p_11, p_293] (by decide)

def p_12967 : CertifiedPrime :=
  certify 12967 (3) [p_2, p_3, p_2161] (by decide)

def p_12973 : CertifiedPrime :=
  certify 12973 (14) [p_2, p_2, p_3, p_23, p_47] (by decide)

def p_13007 : CertifiedPrime :=
  certify 13007 (5) [p_2, p_7, p_929] (by decide)

def p_13033 : CertifiedPrime :=
  certify 13033 (5) [p_2, p_2, p_2, p_3, p_3, p_181] (by decide)

def p_13037 : CertifiedPrime :=
  certify 13037 (2) [p_2, p_2, p_3259] (by decide)

def p_13043 : CertifiedPrime :=
  certify 13043 (2) [p_2, p_6521] (by decide)

def p_13093 : CertifiedPrime :=
  certify 13093 (6) [p_2, p_2, p_3, p_1091] (by decide)

def p_13109 : CertifiedPrime :=
  certify 13109 (2) [p_2, p_2, p_29, p_113] (by decide)

def p_13159 : CertifiedPrime :=
  certify 13159 (3) [p_2, p_3, p_3, p_17, p_43] (by decide)

def p_13163 : CertifiedPrime :=
  certify 13163 (2) [p_2, p_6581] (by decide)

def p_13229 : CertifiedPrime :=
  certify 13229 (2) [p_2, p_2, p_3307] (by decide)

def p_13267 : CertifiedPrime :=
  certify 13267 (3) [p_2, p_3, p_3, p_11, p_67] (by decide)

def p_13297 : CertifiedPrime :=
  certify 13297 (5) [p_2, p_2, p_2, p_2, p_3, p_277] (by decide)

def p_13331 : CertifiedPrime :=
  certify 13331 (2) [p_2, p_5, p_31, p_43] (by decide)

def p_13339 : CertifiedPrime :=
  certify 13339 (2) [p_2, p_3, p_3, p_3, p_13, p_19] (by decide)

def p_13399 : CertifiedPrime :=
  certify 13399 (3) [p_2, p_3, p_7, p_11, p_29] (by decide)

def p_13411 : CertifiedPrime :=
  certify 13411 (2) [p_2, p_3, p_3, p_5, p_149] (by decide)

def p_13457 : CertifiedPrime :=
  certify 13457 (3) [p_2, p_2, p_2, p_2, p_29, p_29] (by decide)

def p_13469 : CertifiedPrime :=
  certify 13469 (2) [p_2, p_2, p_7, p_13, p_37] (by decide)

def p_13477 : CertifiedPrime :=
  certify 13477 (2) [p_2, p_2, p_3, p_1123] (by decide)

def p_13487 : CertifiedPrime :=
  certify 13487 (5) [p_2, p_11, p_613] (by decide)

def p_13499 : CertifiedPrime :=
  certify 13499 (6) [p_2, p_17, p_397] (by decide)

def p_13553 : CertifiedPrime :=
  certify 13553 (3) [p_2, p_2, p_2, p_2, p_7, p_11, p_11] (by decide)

def p_13613 : CertifiedPrime :=
  certify 13613 (2) [p_2, p_2, p_41, p_83] (by decide)

def p_13633 : CertifiedPrime :=
  certify 13633 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_71] (by decide)

def p_13687 : CertifiedPrime :=
  certify 13687 (3) [p_2, p_3, p_2281] (by decide)

def p_13729 : CertifiedPrime :=
  certify 13729 (23) [p_2, p_2, p_2, p_2, p_2, p_3, p_11, p_13] (by decide)

def p_13807 : CertifiedPrime :=
  certify 13807 (5) [p_2, p_3, p_3, p_13, p_59] (by decide)

def p_13903 : CertifiedPrime :=
  certify 13903 (3) [p_2, p_3, p_7, p_331] (by decide)

def p_13931 : CertifiedPrime :=
  certify 13931 (2) [p_2, p_5, p_7, p_199] (by decide)

def p_14029 : CertifiedPrime :=
  certify 14029 (6) [p_2, p_2, p_3, p_7, p_167] (by decide)

def p_14087 : CertifiedPrime :=
  certify 14087 (5) [p_2, p_7043] (by decide)

def p_14149 : CertifiedPrime :=
  certify 14149 (6) [p_2, p_2, p_3, p_3, p_3, p_131] (by decide)

def p_14281 : CertifiedPrime :=
  certify 14281 (19) [p_2, p_2, p_2, p_3, p_5, p_7, p_17] (by decide)

def p_14347 : CertifiedPrime :=
  certify 14347 (3) [p_2, p_3, p_3, p_797] (by decide)

def p_14369 : CertifiedPrime :=
  certify 14369 (3) [p_2, p_2, p_2, p_2, p_2, p_449] (by decide)

def p_14401 : CertifiedPrime :=
  certify 14401 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5] (by decide)

def p_14461 : CertifiedPrime :=
  certify 14461 (2) [p_2, p_2, p_3, p_5, p_241] (by decide)

def p_14519 : CertifiedPrime :=
  certify 14519 (13) [p_2, p_7, p_17, p_61] (by decide)

def p_14549 : CertifiedPrime :=
  certify 14549 (2) [p_2, p_2, p_3637] (by decide)

def p_14653 : CertifiedPrime :=
  certify 14653 (2) [p_2, p_2, p_3, p_3, p_11, p_37] (by decide)

def p_14767 : CertifiedPrime :=
  certify 14767 (3) [p_2, p_3, p_23, p_107] (by decide)

def p_14821 : CertifiedPrime :=
  certify 14821 (2) [p_2, p_2, p_3, p_5, p_13, p_19] (by decide)

def p_14867 : CertifiedPrime :=
  certify 14867 (2) [p_2, p_7433] (by decide)

def p_14939 : CertifiedPrime :=
  certify 14939 (2) [p_2, p_7, p_11, p_97] (by decide)

def p_14951 : CertifiedPrime :=
  certify 14951 (19) [p_2, p_5, p_5, p_13, p_23] (by decide)

def p_15073 : CertifiedPrime :=
  certify 15073 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_157] (by decide)

def p_15077 : CertifiedPrime :=
  certify 15077 (2) [p_2, p_2, p_3769] (by decide)

def p_15091 : CertifiedPrime :=
  certify 15091 (2) [p_2, p_3, p_5, p_503] (by decide)

def p_15107 : CertifiedPrime :=
  certify 15107 (2) [p_2, p_7, p_13, p_83] (by decide)

def p_15137 : CertifiedPrime :=
  certify 15137 (3) [p_2, p_2, p_2, p_2, p_2, p_11, p_43] (by decide)

def p_15149 : CertifiedPrime :=
  certify 15149 (2) [p_2, p_2, p_7, p_541] (by decide)

def p_15241 : CertifiedPrime :=
  certify 15241 (11) [p_2, p_2, p_2, p_3, p_5, p_127] (by decide)

def p_15377 : CertifiedPrime :=
  certify 15377 (3) [p_2, p_2, p_2, p_2, p_31, p_31] (by decide)

def p_15413 : CertifiedPrime :=
  certify 15413 (2) [p_2, p_2, p_3853] (by decide)

def p_15439 : CertifiedPrime :=
  certify 15439 (3) [p_2, p_3, p_31, p_83] (by decide)

def p_15451 : CertifiedPrime :=
  certify 15451 (3) [p_2, p_3, p_5, p_5, p_103] (by decide)

def p_15511 : CertifiedPrime :=
  certify 15511 (3) [p_2, p_3, p_5, p_11, p_47] (by decide)

def p_15527 : CertifiedPrime :=
  certify 15527 (5) [p_2, p_7, p_1109] (by decide)

def p_15569 : CertifiedPrime :=
  certify 15569 (3) [p_2, p_2, p_2, p_2, p_7, p_139] (by decide)

def p_15641 : CertifiedPrime :=
  certify 15641 (3) [p_2, p_2, p_2, p_5, p_17, p_23] (by decide)

def p_15737 : CertifiedPrime :=
  certify 15737 (3) [p_2, p_2, p_2, p_7, p_281] (by decide)

def p_15797 : CertifiedPrime :=
  certify 15797 (2) [p_2, p_2, p_11, p_359] (by decide)

def p_15877 : CertifiedPrime :=
  certify 15877 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7] (by decide)

def p_15881 : CertifiedPrime :=
  certify 15881 (3) [p_2, p_2, p_2, p_5, p_397] (by decide)

def p_15901 : CertifiedPrime :=
  certify 15901 (10) [p_2, p_2, p_3, p_5, p_5, p_53] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
