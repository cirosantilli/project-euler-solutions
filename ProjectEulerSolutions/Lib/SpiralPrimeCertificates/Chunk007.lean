import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk006

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_8689 : CertifiedPrime :=
  certify 8689 (13) [p_2, p_2, p_2, p_2, p_3, p_181] (by decide)

def p_8693 : CertifiedPrime :=
  certify 8693 (2) [p_2, p_2, p_41, p_53] (by decide)

def p_8713 : CertifiedPrime :=
  certify 8713 (5) [p_2, p_2, p_2, p_3, p_3, p_11, p_11] (by decide)

def p_8719 : CertifiedPrime :=
  certify 8719 (3) [p_2, p_3, p_1453] (by decide)

def p_8731 : CertifiedPrime :=
  certify 8731 (2) [p_2, p_3, p_3, p_5, p_97] (by decide)

def p_8741 : CertifiedPrime :=
  certify 8741 (2) [p_2, p_2, p_5, p_19, p_23] (by decide)

def p_8753 : CertifiedPrime :=
  certify 8753 (3) [p_2, p_2, p_2, p_2, p_547] (by decide)

def p_8761 : CertifiedPrime :=
  certify 8761 (23) [p_2, p_2, p_2, p_3, p_5, p_73] (by decide)

def p_8803 : CertifiedPrime :=
  certify 8803 (2) [p_2, p_3, p_3, p_3, p_163] (by decide)

def p_8807 : CertifiedPrime :=
  certify 8807 (5) [p_2, p_7, p_17, p_37] (by decide)

def p_8837 : CertifiedPrime :=
  certify 8837 (2) [p_2, p_2, p_47, p_47] (by decide)

def p_8839 : CertifiedPrime :=
  certify 8839 (3) [p_2, p_3, p_3, p_491] (by decide)

def p_8887 : CertifiedPrime :=
  certify 8887 (3) [p_2, p_3, p_1481] (by decide)

def p_8923 : CertifiedPrime :=
  certify 8923 (2) [p_2, p_3, p_1487] (by decide)

def p_8929 : CertifiedPrime :=
  certify 8929 (11) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_31] (by decide)

def p_8963 : CertifiedPrime :=
  certify 8963 (2) [p_2, p_4481] (by decide)

def p_9001 : CertifiedPrime :=
  certify 9001 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5] (by decide)

def p_9007 : CertifiedPrime :=
  certify 9007 (3) [p_2, p_3, p_19, p_79] (by decide)

def p_9043 : CertifiedPrime :=
  certify 9043 (3) [p_2, p_3, p_11, p_137] (by decide)

def p_9059 : CertifiedPrime :=
  certify 9059 (2) [p_2, p_7, p_647] (by decide)

def p_9127 : CertifiedPrime :=
  certify 9127 (3) [p_2, p_3, p_3, p_3, p_13, p_13] (by decide)

def p_9137 : CertifiedPrime :=
  certify 9137 (3) [p_2, p_2, p_2, p_2, p_571] (by decide)

def p_9157 : CertifiedPrime :=
  certify 9157 (6) [p_2, p_2, p_3, p_7, p_109] (by decide)

def p_9173 : CertifiedPrime :=
  certify 9173 (2) [p_2, p_2, p_2293] (by decide)

def p_9181 : CertifiedPrime :=
  certify 9181 (2) [p_2, p_2, p_3, p_3, p_3, p_5, p_17] (by decide)

def p_9199 : CertifiedPrime :=
  certify 9199 (3) [p_2, p_3, p_3, p_7, p_73] (by decide)

def p_9203 : CertifiedPrime :=
  certify 9203 (2) [p_2, p_43, p_107] (by decide)

def p_9241 : CertifiedPrime :=
  certify 9241 (13) [p_2, p_2, p_2, p_3, p_5, p_7, p_11] (by decide)

def p_9283 : CertifiedPrime :=
  certify 9283 (2) [p_2, p_3, p_7, p_13, p_17] (by decide)

def p_9293 : CertifiedPrime :=
  certify 9293 (2) [p_2, p_2, p_23, p_101] (by decide)

def p_9319 : CertifiedPrime :=
  certify 9319 (3) [p_2, p_3, p_1553] (by decide)

def p_9323 : CertifiedPrime :=
  certify 9323 (2) [p_2, p_59, p_79] (by decide)

def p_9337 : CertifiedPrime :=
  certify 9337 (5) [p_2, p_2, p_2, p_3, p_389] (by decide)

def p_9377 : CertifiedPrime :=
  certify 9377 (3) [p_2, p_2, p_2, p_2, p_2, p_293] (by decide)

def p_9397 : CertifiedPrime :=
  certify 9397 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_29] (by decide)

def p_9419 : CertifiedPrime :=
  certify 9419 (2) [p_2, p_17, p_277] (by decide)

def p_9463 : CertifiedPrime :=
  certify 9463 (3) [p_2, p_3, p_19, p_83] (by decide)

def p_9473 : CertifiedPrime :=
  certify 9473 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_37] (by decide)

def p_9497 : CertifiedPrime :=
  certify 9497 (3) [p_2, p_2, p_2, p_1187] (by decide)

def p_9521 : CertifiedPrime :=
  certify 9521 (3) [p_2, p_2, p_2, p_2, p_5, p_7, p_17] (by decide)

def p_9533 : CertifiedPrime :=
  certify 9533 (2) [p_2, p_2, p_2383] (by decide)

def p_9601 : CertifiedPrime :=
  certify 9601 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5] (by decide)

def p_9613 : CertifiedPrime :=
  certify 9613 (2) [p_2, p_2, p_3, p_3, p_3, p_89] (by decide)

def p_9661 : CertifiedPrime :=
  certify 9661 (2) [p_2, p_2, p_3, p_5, p_7, p_23] (by decide)

def p_9689 : CertifiedPrime :=
  certify 9689 (3) [p_2, p_2, p_2, p_7, p_173] (by decide)

def p_9719 : CertifiedPrime :=
  certify 9719 (17) [p_2, p_43, p_113] (by decide)

def p_9733 : CertifiedPrime :=
  certify 9733 (2) [p_2, p_2, p_3, p_811] (by decide)

def p_9743 : CertifiedPrime :=
  certify 9743 (5) [p_2, p_4871] (by decide)

def p_9749 : CertifiedPrime :=
  certify 9749 (2) [p_2, p_2, p_2437] (by decide)

def p_9769 : CertifiedPrime :=
  certify 9769 (13) [p_2, p_2, p_2, p_3, p_11, p_37] (by decide)

def p_9781 : CertifiedPrime :=
  certify 9781 (6) [p_2, p_2, p_3, p_5, p_163] (by decide)

def p_9803 : CertifiedPrime :=
  certify 9803 (2) [p_2, p_13, p_13, p_29] (by decide)

def p_9811 : CertifiedPrime :=
  certify 9811 (3) [p_2, p_3, p_3, p_5, p_109] (by decide)

def p_9817 : CertifiedPrime :=
  certify 9817 (5) [p_2, p_2, p_2, p_3, p_409] (by decide)

def p_9829 : CertifiedPrime :=
  certify 9829 (10) [p_2, p_2, p_3, p_3, p_3, p_7, p_13] (by decide)

def p_9833 : CertifiedPrime :=
  certify 9833 (3) [p_2, p_2, p_2, p_1229] (by decide)

def p_9839 : CertifiedPrime :=
  certify 9839 (7) [p_2, p_4919] (by decide)

def p_9857 : CertifiedPrime :=
  certify 9857 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_11] (by decide)

def p_9883 : CertifiedPrime :=
  certify 9883 (2) [p_2, p_3, p_3, p_3, p_3, p_61] (by decide)

def p_9901 : CertifiedPrime :=
  certify 9901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_11] (by decide)

def p_9967 : CertifiedPrime :=
  certify 9967 (3) [p_2, p_3, p_11, p_151] (by decide)

def p_9973 : CertifiedPrime :=
  certify 9973 (11) [p_2, p_2, p_3, p_3, p_277] (by decide)

def p_10007 : CertifiedPrime :=
  certify 10007 (5) [p_2, p_5003] (by decide)

def p_10009 : CertifiedPrime :=
  certify 10009 (11) [p_2, p_2, p_2, p_3, p_3, p_139] (by decide)

def p_10061 : CertifiedPrime :=
  certify 10061 (3) [p_2, p_2, p_5, p_503] (by decide)

def p_10091 : CertifiedPrime :=
  certify 10091 (2) [p_2, p_5, p_1009] (by decide)

def p_10093 : CertifiedPrime :=
  certify 10093 (2) [p_2, p_2, p_3, p_29, p_29] (by decide)

def p_10103 : CertifiedPrime :=
  certify 10103 (5) [p_2, p_5051] (by decide)

def p_10133 : CertifiedPrime :=
  certify 10133 (2) [p_2, p_2, p_17, p_149] (by decide)

def p_10139 : CertifiedPrime :=
  certify 10139 (2) [p_2, p_37, p_137] (by decide)

def p_10141 : CertifiedPrime :=
  certify 10141 (2) [p_2, p_2, p_3, p_5, p_13, p_13] (by decide)

def p_10151 : CertifiedPrime :=
  certify 10151 (7) [p_2, p_5, p_5, p_7, p_29] (by decide)

def p_10159 : CertifiedPrime :=
  certify 10159 (3) [p_2, p_3, p_1693] (by decide)

def p_10177 : CertifiedPrime :=
  certify 10177 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_53] (by decide)

def p_10193 : CertifiedPrime :=
  certify 10193 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_13] (by decide)

def p_10223 : CertifiedPrime :=
  certify 10223 (5) [p_2, p_19, p_269] (by decide)

def p_10243 : CertifiedPrime :=
  certify 10243 (7) [p_2, p_3, p_3, p_569] (by decide)

def p_10271 : CertifiedPrime :=
  certify 10271 (7) [p_2, p_5, p_13, p_79] (by decide)

def p_10273 : CertifiedPrime :=
  certify 10273 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_107] (by decide)

def p_10289 : CertifiedPrime :=
  certify 10289 (3) [p_2, p_2, p_2, p_2, p_643] (by decide)

def p_10303 : CertifiedPrime :=
  certify 10303 (3) [p_2, p_3, p_17, p_101] (by decide)

def p_10333 : CertifiedPrime :=
  certify 10333 (5) [p_2, p_2, p_3, p_3, p_7, p_41] (by decide)

def p_10369 : CertifiedPrime :=
  certify 10369 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3] (by decide)

def p_10391 : CertifiedPrime :=
  certify 10391 (19) [p_2, p_5, p_1039] (by decide)

def p_10399 : CertifiedPrime :=
  certify 10399 (6) [p_2, p_3, p_1733] (by decide)

def p_10429 : CertifiedPrime :=
  certify 10429 (7) [p_2, p_2, p_3, p_11, p_79] (by decide)

def p_10477 : CertifiedPrime :=
  certify 10477 (2) [p_2, p_2, p_3, p_3, p_3, p_97] (by decide)

def p_10567 : CertifiedPrime :=
  certify 10567 (6) [p_2, p_3, p_3, p_587] (by decide)

def p_10589 : CertifiedPrime :=
  certify 10589 (2) [p_2, p_2, p_2647] (by decide)

def p_10597 : CertifiedPrime :=
  certify 10597 (5) [p_2, p_2, p_3, p_883] (by decide)

def p_10613 : CertifiedPrime :=
  certify 10613 (2) [p_2, p_2, p_7, p_379] (by decide)

def p_10631 : CertifiedPrime :=
  certify 10631 (11) [p_2, p_5, p_1063] (by decide)

def p_10687 : CertifiedPrime :=
  certify 10687 (5) [p_2, p_3, p_13, p_137] (by decide)

def p_10723 : CertifiedPrime :=
  certify 10723 (2) [p_2, p_3, p_1787] (by decide)

def p_10733 : CertifiedPrime :=
  certify 10733 (2) [p_2, p_2, p_2683] (by decide)

def p_10753 : CertifiedPrime :=
  certify 10753 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7] (by decide)

def p_10771 : CertifiedPrime :=
  certify 10771 (3) [p_2, p_3, p_5, p_359] (by decide)

def p_10781 : CertifiedPrime :=
  certify 10781 (10) [p_2, p_2, p_5, p_7, p_7, p_11] (by decide)

def p_10789 : CertifiedPrime :=
  certify 10789 (2) [p_2, p_2, p_3, p_29, p_31] (by decide)

def p_10903 : CertifiedPrime :=
  certify 10903 (3) [p_2, p_3, p_23, p_79] (by decide)

def p_10939 : CertifiedPrime :=
  certify 10939 (3) [p_2, p_3, p_1823] (by decide)

def p_10949 : CertifiedPrime :=
  certify 10949 (2) [p_2, p_2, p_7, p_17, p_23] (by decide)

def p_10957 : CertifiedPrime :=
  certify 10957 (5) [p_2, p_2, p_3, p_11, p_83] (by decide)

def p_10987 : CertifiedPrime :=
  certify 10987 (2) [p_2, p_3, p_1831] (by decide)

def p_10993 : CertifiedPrime :=
  certify 10993 (7) [p_2, p_2, p_2, p_2, p_3, p_229] (by decide)

def p_11027 : CertifiedPrime :=
  certify 11027 (2) [p_2, p_37, p_149] (by decide)

def p_11069 : CertifiedPrime :=
  certify 11069 (2) [p_2, p_2, p_2767] (by decide)

def p_11083 : CertifiedPrime :=
  certify 11083 (2) [p_2, p_3, p_1847] (by decide)

def p_11087 : CertifiedPrime :=
  certify 11087 (5) [p_2, p_23, p_241] (by decide)

def p_11113 : CertifiedPrime :=
  certify 11113 (13) [p_2, p_2, p_2, p_3, p_463] (by decide)

def p_11117 : CertifiedPrime :=
  certify 11117 (3) [p_2, p_2, p_7, p_397] (by decide)

def p_11131 : CertifiedPrime :=
  certify 11131 (2) [p_2, p_3, p_5, p_7, p_53] (by decide)

def p_11161 : CertifiedPrime :=
  certify 11161 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_31] (by decide)

def p_11171 : CertifiedPrime :=
  certify 11171 (2) [p_2, p_5, p_1117] (by decide)

def p_11173 : CertifiedPrime :=
  certify 11173 (5) [p_2, p_2, p_3, p_7, p_7, p_19] (by decide)

def p_11213 : CertifiedPrime :=
  certify 11213 (2) [p_2, p_2, p_2803] (by decide)

def p_11239 : CertifiedPrime :=
  certify 11239 (3) [p_2, p_3, p_1873] (by decide)

def p_11243 : CertifiedPrime :=
  certify 11243 (5) [p_2, p_7, p_11, p_73] (by decide)

def p_11257 : CertifiedPrime :=
  certify 11257 (10) [p_2, p_2, p_2, p_3, p_7, p_67] (by decide)

def p_11321 : CertifiedPrime :=
  certify 11321 (3) [p_2, p_2, p_2, p_5, p_283] (by decide)

def p_11329 : CertifiedPrime :=
  certify 11329 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_59] (by decide)

def p_11383 : CertifiedPrime :=
  certify 11383 (5) [p_2, p_3, p_7, p_271] (by decide)

def p_11447 : CertifiedPrime :=
  certify 11447 (5) [p_2, p_59, p_97] (by decide)

def p_11467 : CertifiedPrime :=
  certify 11467 (5) [p_2, p_3, p_3, p_7, p_7, p_13] (by decide)

def p_11549 : CertifiedPrime :=
  certify 11549 (2) [p_2, p_2, p_2887] (by decide)

def p_11579 : CertifiedPrime :=
  certify 11579 (2) [p_2, p_7, p_827] (by decide)

def p_11593 : CertifiedPrime :=
  certify 11593 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_23] (by decide)

def p_11597 : CertifiedPrime :=
  certify 11597 (3) [p_2, p_2, p_13, p_223] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
