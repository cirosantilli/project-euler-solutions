import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk009

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_24083 : CertifiedPrime :=
  certify 24083 (2) [p_2, p_12041] (by decide)

def p_24103 : CertifiedPrime :=
  certify 24103 (5) [p_2, p_3, p_3, p_13, p_103] (by decide)

def p_24107 : CertifiedPrime :=
  certify 24107 (2) [p_2, p_17, p_709] (by decide)

def p_24181 : CertifiedPrime :=
  certify 24181 (17) [p_2, p_2, p_3, p_5, p_13, p_31] (by decide)

def p_24239 : CertifiedPrime :=
  certify 24239 (13) [p_2, p_12119] (by decide)

def p_24247 : CertifiedPrime :=
  certify 24247 (3) [p_2, p_3, p_3, p_3, p_449] (by decide)

def p_24281 : CertifiedPrime :=
  certify 24281 (3) [p_2, p_2, p_2, p_5, p_607] (by decide)

def p_24337 : CertifiedPrime :=
  certify 24337 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13] (by decide)

def p_24481 : CertifiedPrime :=
  certify 24481 (11) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_17] (by decide)

def p_24533 : CertifiedPrime :=
  certify 24533 (2) [p_2, p_2, p_6133] (by decide)

def p_24659 : CertifiedPrime :=
  certify 24659 (2) [p_2, p_12329] (by decide)

def p_24709 : CertifiedPrime :=
  certify 24709 (2) [p_2, p_2, p_3, p_29, p_71] (by decide)

def p_24763 : CertifiedPrime :=
  certify 24763 (2) [p_2, p_3, p_4127] (by decide)

def p_24859 : CertifiedPrime :=
  certify 24859 (2) [p_2, p_3, p_3, p_1381] (by decide)

def p_24907 : CertifiedPrime :=
  certify 24907 (2) [p_2, p_3, p_7, p_593] (by decide)

def p_25037 : CertifiedPrime :=
  certify 25037 (2) [p_2, p_2, p_11, p_569] (by decide)

def p_25057 : CertifiedPrime :=
  certify 25057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_29] (by decide)

def p_25153 : CertifiedPrime :=
  certify 25153 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_131] (by decide)

def p_25409 : CertifiedPrime :=
  certify 25409 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_397] (by decide)

def p_25523 : CertifiedPrime :=
  certify 25523 (2) [p_2, p_7, p_1823] (by decide)

def p_25537 : CertifiedPrime :=
  certify 25537 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_19] (by decide)

def p_25583 : CertifiedPrime :=
  certify 25583 (5) [p_2, p_12791] (by decide)

def p_25601 : CertifiedPrime :=
  certify 25601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5] (by decide)

def p_25621 : CertifiedPrime :=
  certify 25621 (10) [p_2, p_2, p_3, p_5, p_7, p_61] (by decide)

def p_25657 : CertifiedPrime :=
  certify 25657 (5) [p_2, p_2, p_2, p_3, p_1069] (by decide)

def p_25747 : CertifiedPrime :=
  certify 25747 (2) [p_2, p_3, p_7, p_613] (by decide)

def p_25873 : CertifiedPrime :=
  certify 25873 (10) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_11] (by decide)

def p_25943 : CertifiedPrime :=
  certify 25943 (5) [p_2, p_7, p_17, p_109] (by decide)

def p_25969 : CertifiedPrime :=
  certify 25969 (7) [p_2, p_2, p_2, p_2, p_3, p_541] (by decide)

def p_25997 : CertifiedPrime :=
  certify 25997 (2) [p_2, p_2, p_67, p_97] (by decide)

def p_25999 : CertifiedPrime :=
  certify 25999 (7) [p_2, p_3, p_7, p_619] (by decide)

def p_26083 : CertifiedPrime :=
  certify 26083 (7) [p_2, p_3, p_3, p_3, p_3, p_7, p_23] (by decide)

def p_26119 : CertifiedPrime :=
  certify 26119 (3) [p_2, p_3, p_3, p_1451] (by decide)

def p_26171 : CertifiedPrime :=
  certify 26171 (2) [p_2, p_5, p_2617] (by decide)

def p_26183 : CertifiedPrime :=
  certify 26183 (5) [p_2, p_13, p_19, p_53] (by decide)

def p_26407 : CertifiedPrime :=
  certify 26407 (5) [p_2, p_3, p_3, p_3, p_3, p_163] (by decide)

def p_27061 : CertifiedPrime :=
  certify 27061 (2) [p_2, p_2, p_3, p_5, p_11, p_41] (by decide)

def p_28057 : CertifiedPrime :=
  certify 28057 (5) [p_2, p_2, p_2, p_3, p_7, p_167] (by decide)

def p_28393 : CertifiedPrime :=
  certify 28393 (15) [p_2, p_2, p_2, p_3, p_7, p_13, p_13] (by decide)

def p_28901 : CertifiedPrime :=
  certify 28901 (3) [p_2, p_2, p_5, p_5, p_17, p_17] (by decide)

def p_30103 : CertifiedPrime :=
  certify 30103 (3) [p_2, p_3, p_29, p_173] (by decide)

def p_30977 : CertifiedPrime :=
  certify 30977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11] (by decide)

def p_31153 : CertifiedPrime :=
  certify 31153 (10) [p_2, p_2, p_2, p_2, p_3, p_11, p_59] (by decide)

def p_32401 : CertifiedPrime :=
  certify 32401 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_33857 : CertifiedPrime :=
  certify 33857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_23, p_23] (by decide)

def p_35533 : CertifiedPrime :=
  certify 35533 (2) [p_2, p_2, p_3, p_3, p_3, p_7, p_47] (by decide)

def p_35911 : CertifiedPrime :=
  certify 35911 (12) [p_2, p_3, p_3, p_3, p_5, p_7, p_19] (by decide)

def p_37057 : CertifiedPrime :=
  certify 37057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_193] (by decide)

def p_37831 : CertifiedPrime :=
  certify 37831 (3) [p_2, p_3, p_5, p_13, p_97] (by decide)

def p_41413 : CertifiedPrime :=
  certify 41413 (6) [p_2, p_2, p_3, p_7, p_17, p_29] (by decide)

def p_41617 : CertifiedPrime :=
  certify 41617 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_17, p_17] (by decide)

def p_42437 : CertifiedPrime :=
  certify 42437 (2) [p_2, p_2, p_103, p_103] (by decide)

def p_42643 : CertifiedPrime :=
  certify 42643 (2) [p_2, p_3, p_3, p_23, p_103] (by decide)

def p_43891 : CertifiedPrime :=
  certify 43891 (3) [p_2, p_3, p_5, p_7, p_11, p_19] (by decide)

def p_44101 : CertifiedPrime :=
  certify 44101 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7] (by decide)

def p_46441 : CertifiedPrime :=
  certify 46441 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_43] (by decide)

def p_47743 : CertifiedPrime :=
  certify 47743 (3) [p_2, p_3, p_73, p_109] (by decide)

def p_50177 : CertifiedPrime :=
  certify 50177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7] (by decide)

def p_52901 : CertifiedPrime :=
  certify 52901 (3) [p_2, p_2, p_5, p_5, p_23, p_23] (by decide)

def p_53593 : CertifiedPrime :=
  certify 53593 (5) [p_2, p_2, p_2, p_3, p_7, p_11, p_29] (by decide)

def p_55697 : CertifiedPrime :=
  certify 55697 (3) [p_2, p_2, p_2, p_2, p_59, p_59] (by decide)

def p_55933 : CertifiedPrime :=
  certify 55933 (2) [p_2, p_2, p_3, p_59, p_79] (by decide)

def p_57601 : CertifiedPrime :=
  certify 57601 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5] (by decide)

def p_60271 : CertifiedPrime :=
  certify 60271 (3) [p_2, p_3, p_5, p_7, p_7, p_41] (by decide)

def p_60763 : CertifiedPrime :=
  certify 60763 (3) [p_2, p_3, p_13, p_19, p_41] (by decide)

def p_62501 : CertifiedPrime :=
  certify 62501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5] (by decide)

def p_65537 : CertifiedPrime :=
  certify 65537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2] (by decide)

def p_67601 : CertifiedPrime :=
  certify 67601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13] (by decide)

def p_69697 : CertifiedPrime :=
  certify 69697 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11] (by decide)

def p_71023 : CertifiedPrime :=
  certify 71023 (6) [p_2, p_3, p_7, p_19, p_89] (by decide)

def p_72901 : CertifiedPrime :=
  certify 72901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_74257 : CertifiedPrime :=
  certify 74257 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_17] (by decide)

def p_77563 : CertifiedPrime :=
  certify 77563 (2) [p_2, p_3, p_3, p_31, p_139] (by decide)

def p_78121 : CertifiedPrime :=
  certify 78121 (11) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_31] (by decide)

def p_78401 : CertifiedPrime :=
  certify 78401 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7] (by decide)

def p_80657 : CertifiedPrime :=
  certify 80657 (3) [p_2, p_2, p_2, p_2, p_71, p_71] (by decide)

def p_82657 : CertifiedPrime :=
  certify 82657 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_41] (by decide)

def p_83233 : CertifiedPrime :=
  certify 83233 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_17, p_17] (by decide)

def p_84391 : CertifiedPrime :=
  certify 84391 (3) [p_2, p_3, p_5, p_29, p_97] (by decide)

def p_86143 : CertifiedPrime :=
  certify 86143 (3) [p_2, p_3, p_7, p_7, p_293] (by decide)

def p_90001 : CertifiedPrime :=
  certify 90001 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5] (by decide)

def p_93637 : CertifiedPrime :=
  certify 93637 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_17, p_17] (by decide)

def p_95791 : CertifiedPrime :=
  certify 95791 (3) [p_2, p_3, p_5, p_31, p_103] (by decide)

def p_98597 : CertifiedPrime :=
  certify 98597 (2) [p_2, p_2, p_157, p_157] (by decide)

def p_98911 : CertifiedPrime :=
  certify 98911 (12) [p_2, p_3, p_3, p_5, p_7, p_157] (by decide)

def p_106277 : CertifiedPrime :=
  certify 106277 (2) [p_2, p_2, p_163, p_163] (by decide)

def p_108571 : CertifiedPrime :=
  certify 108571 (3) [p_2, p_3, p_5, p_7, p_11, p_47] (by decide)

def p_110557 : CertifiedPrime :=
  certify 110557 (2) [p_2, p_2, p_3, p_3, p_37, p_83] (by decide)

def p_113233 : CertifiedPrime :=
  certify 113233 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_337] (by decide)

def p_115601 : CertifiedPrime :=
  certify 115601 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17] (by decide)

def p_117307 : CertifiedPrime :=
  certify 117307 (5) [p_2, p_3, p_3, p_7, p_7, p_7, p_19] (by decide)

def p_118681 : CertifiedPrime :=
  certify 118681 (14) [p_2, p_2, p_2, p_3, p_5, p_23, p_43] (by decide)

def p_121453 : CertifiedPrime :=
  certify 121453 (6) [p_2, p_2, p_3, p_29, p_349] (by decide)

def p_122501 : CertifiedPrime :=
  certify 122501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7] (by decide)

def p_123553 : CertifiedPrime :=
  certify 123553 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_11, p_13] (by decide)

def p_127807 : CertifiedPrime :=
  certify 127807 (3) [p_2, p_3, p_7, p_17, p_179] (by decide)

def p_136531 : CertifiedPrime :=
  certify 136531 (2) [p_2, p_3, p_3, p_5, p_37, p_41] (by decide)

def p_143263 : CertifiedPrime :=
  certify 143263 (29) [p_2, p_3, p_3, p_3, p_7, p_379] (by decide)

def p_145543 : CertifiedPrime :=
  certify 145543 (3) [p_2, p_3, p_127, p_191] (by decide)

def p_147073 : CertifiedPrime :=
  certify 147073 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_383] (by decide)

def p_147457 : CertifiedPrime :=
  certify 147457 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3] (by decide)

def p_148997 : CertifiedPrime :=
  certify 148997 (3) [p_2, p_2, p_193, p_193] (by decide)

def p_154057 : CertifiedPrime :=
  certify 154057 (5) [p_2, p_2, p_2, p_3, p_7, p_7, p_131] (by decide)

def p_156421 : CertifiedPrime :=
  certify 156421 (7) [p_2, p_2, p_3, p_3, p_5, p_11, p_79] (by decide)

def p_156817 : CertifiedPrime :=
  certify 156817 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_11, p_11] (by decide)

def p_158803 : CertifiedPrime :=
  certify 158803 (2) [p_2, p_3, p_7, p_19, p_199] (by decide)

def p_160001 : CertifiedPrime :=
  certify 160001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5] (by decide)

def p_162007 : CertifiedPrime :=
  certify 162007 (3) [p_2, p_3, p_13, p_31, p_67] (by decide)

def p_163621 : CertifiedPrime :=
  certify 163621 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_101] (by decide)

def p_164431 : CertifiedPrime :=
  certify 164431 (6) [p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_29] (by decide)

def p_164837 : CertifiedPrime :=
  certify 164837 (2) [p_2, p_2, p_7, p_7, p_29, p_29] (by decide)

def p_171811 : CertifiedPrime :=
  certify 171811 (15) [p_2, p_3, p_3, p_5, p_23, p_83] (by decide)

def p_173473 : CertifiedPrime :=
  certify 173473 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_13, p_139] (by decide)

def p_176401 : CertifiedPrime :=
  certify 176401 (23) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7] (by decide)

def p_181903 : CertifiedPrime :=
  certify 181903 (3) [p_2, p_3, p_7, p_61, p_71] (by decide)

def p_184901 : CertifiedPrime :=
  certify 184901 (2) [p_2, p_2, p_5, p_5, p_43, p_43] (by decide)

def p_188791 : CertifiedPrime :=
  certify 188791 (7) [p_2, p_3, p_5, p_7, p_29, p_31] (by decide)

def p_189661 : CertifiedPrime :=
  certify 189661 (2) [p_2, p_2, p_3, p_5, p_29, p_109] (by decide)

def p_190097 : CertifiedPrime :=
  certify 190097 (3) [p_2, p_2, p_2, p_2, p_109, p_109] (by decide)

def p_193601 : CertifiedPrime :=
  certify 193601 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_11, p_11] (by decide)

def p_197137 : CertifiedPrime :=
  certify 197137 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_37, p_37] (by decide)

def p_200257 : CertifiedPrime :=
  certify 200257 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_149] (by decide)

def p_205663 : CertifiedPrime :=
  certify 205663 (3) [p_2, p_3, p_151, p_227] (by decide)

def p_207481 : CertifiedPrime :=
  certify 207481 (31) [p_2, p_2, p_2, p_3, p_5, p_7, p_13, p_19] (by decide)

def p_208393 : CertifiedPrime :=
  certify 208393 (5) [p_2, p_2, p_2, p_3, p_19, p_457] (by decide)

def p_215297 : CertifiedPrime :=
  certify 215297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_29, p_29] (by decide)

def p_217157 : CertifiedPrime :=
  certify 217157 (2) [p_2, p_2, p_233, p_233] (by decide)

def p_220901 : CertifiedPrime :=
  certify 220901 (2) [p_2, p_2, p_5, p_5, p_47, p_47] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
