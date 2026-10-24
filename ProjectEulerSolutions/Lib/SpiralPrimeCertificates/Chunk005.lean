import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk004

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_5113 : CertifiedPrime :=
  certify 5113 (19) [p_2, p_2, p_2, p_3, p_3, p_71] (by decide)

def p_5119 : CertifiedPrime :=
  certify 5119 (3) [p_2, p_3, p_853] (by decide)

def p_5147 : CertifiedPrime :=
  certify 5147 (2) [p_2, p_31, p_83] (by decide)

def p_5153 : CertifiedPrime :=
  certify 5153 (5) [p_2, p_2, p_2, p_2, p_2, p_7, p_23] (by decide)

def p_5167 : CertifiedPrime :=
  certify 5167 (6) [p_2, p_3, p_3, p_7, p_41] (by decide)

def p_5171 : CertifiedPrime :=
  certify 5171 (2) [p_2, p_5, p_11, p_47] (by decide)

def p_5179 : CertifiedPrime :=
  certify 5179 (2) [p_2, p_3, p_863] (by decide)

def p_5197 : CertifiedPrime :=
  certify 5197 (7) [p_2, p_2, p_3, p_433] (by decide)

def p_5209 : CertifiedPrime :=
  certify 5209 (17) [p_2, p_2, p_2, p_3, p_7, p_31] (by decide)

def p_5231 : CertifiedPrime :=
  certify 5231 (7) [p_2, p_5, p_523] (by decide)

def p_5233 : CertifiedPrime :=
  certify 5233 (10) [p_2, p_2, p_2, p_2, p_3, p_109] (by decide)

def p_5237 : CertifiedPrime :=
  certify 5237 (3) [p_2, p_2, p_7, p_11, p_17] (by decide)

def p_5261 : CertifiedPrime :=
  certify 5261 (2) [p_2, p_2, p_5, p_263] (by decide)

def p_5281 : CertifiedPrime :=
  certify 5281 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_11] (by decide)

def p_5297 : CertifiedPrime :=
  certify 5297 (3) [p_2, p_2, p_2, p_2, p_331] (by decide)

def p_5303 : CertifiedPrime :=
  certify 5303 (5) [p_2, p_11, p_241] (by decide)

def p_5309 : CertifiedPrime :=
  certify 5309 (2) [p_2, p_2, p_1327] (by decide)

def p_5323 : CertifiedPrime :=
  certify 5323 (5) [p_2, p_3, p_887] (by decide)

def p_5333 : CertifiedPrime :=
  certify 5333 (2) [p_2, p_2, p_31, p_43] (by decide)

def p_5347 : CertifiedPrime :=
  certify 5347 (3) [p_2, p_3, p_3, p_3, p_3, p_3, p_11] (by decide)

def p_5351 : CertifiedPrime :=
  certify 5351 (11) [p_2, p_5, p_5, p_107] (by decide)

def p_5399 : CertifiedPrime :=
  certify 5399 (7) [p_2, p_2699] (by decide)

def p_5407 : CertifiedPrime :=
  certify 5407 (3) [p_2, p_3, p_17, p_53] (by decide)

def p_5419 : CertifiedPrime :=
  certify 5419 (3) [p_2, p_3, p_3, p_7, p_43] (by decide)

def p_5437 : CertifiedPrime :=
  certify 5437 (5) [p_2, p_2, p_3, p_3, p_151] (by decide)

def p_5441 : CertifiedPrime :=
  certify 5441 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_17] (by decide)

def p_5443 : CertifiedPrime :=
  certify 5443 (2) [p_2, p_3, p_907] (by decide)

def p_5477 : CertifiedPrime :=
  certify 5477 (2) [p_2, p_2, p_37, p_37] (by decide)

def p_5479 : CertifiedPrime :=
  certify 5479 (3) [p_2, p_3, p_11, p_83] (by decide)

def p_5483 : CertifiedPrime :=
  certify 5483 (2) [p_2, p_2741] (by decide)

def p_5501 : CertifiedPrime :=
  certify 5501 (2) [p_2, p_2, p_5, p_5, p_5, p_11] (by decide)

def p_5519 : CertifiedPrime :=
  certify 5519 (13) [p_2, p_31, p_89] (by decide)

def p_5521 : CertifiedPrime :=
  certify 5521 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_23] (by decide)

def p_5527 : CertifiedPrime :=
  certify 5527 (5) [p_2, p_3, p_3, p_307] (by decide)

def p_5531 : CertifiedPrime :=
  certify 5531 (10) [p_2, p_5, p_7, p_79] (by decide)

def p_5557 : CertifiedPrime :=
  certify 5557 (2) [p_2, p_2, p_3, p_463] (by decide)

def p_5569 : CertifiedPrime :=
  certify 5569 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_29] (by decide)

def p_5591 : CertifiedPrime :=
  certify 5591 (11) [p_2, p_5, p_13, p_43] (by decide)

def p_5623 : CertifiedPrime :=
  certify 5623 (5) [p_2, p_3, p_937] (by decide)

def p_5639 : CertifiedPrime :=
  certify 5639 (7) [p_2, p_2819] (by decide)

def p_5641 : CertifiedPrime :=
  certify 5641 (14) [p_2, p_2, p_2, p_3, p_5, p_47] (by decide)

def p_5647 : CertifiedPrime :=
  certify 5647 (3) [p_2, p_3, p_941] (by decide)

def p_5651 : CertifiedPrime :=
  certify 5651 (2) [p_2, p_5, p_5, p_113] (by decide)

def p_5659 : CertifiedPrime :=
  certify 5659 (2) [p_2, p_3, p_23, p_41] (by decide)

def p_5669 : CertifiedPrime :=
  certify 5669 (3) [p_2, p_2, p_13, p_109] (by decide)

def p_5683 : CertifiedPrime :=
  certify 5683 (2) [p_2, p_3, p_947] (by decide)

def p_5689 : CertifiedPrime :=
  certify 5689 (11) [p_2, p_2, p_2, p_3, p_3, p_79] (by decide)

def p_5701 : CertifiedPrime :=
  certify 5701 (2) [p_2, p_2, p_3, p_5, p_5, p_19] (by decide)

def p_5711 : CertifiedPrime :=
  certify 5711 (19) [p_2, p_5, p_571] (by decide)

def p_5737 : CertifiedPrime :=
  certify 5737 (5) [p_2, p_2, p_2, p_3, p_239] (by decide)

def p_5741 : CertifiedPrime :=
  certify 5741 (2) [p_2, p_2, p_5, p_7, p_41] (by decide)

def p_5743 : CertifiedPrime :=
  certify 5743 (10) [p_2, p_3, p_3, p_11, p_29] (by decide)

def p_5749 : CertifiedPrime :=
  certify 5749 (2) [p_2, p_2, p_3, p_479] (by decide)

def p_5779 : CertifiedPrime :=
  certify 5779 (2) [p_2, p_3, p_3, p_3, p_107] (by decide)

def p_5783 : CertifiedPrime :=
  certify 5783 (7) [p_2, p_7, p_7, p_59] (by decide)

def p_5791 : CertifiedPrime :=
  certify 5791 (6) [p_2, p_3, p_5, p_193] (by decide)

def p_5807 : CertifiedPrime :=
  certify 5807 (5) [p_2, p_2903] (by decide)

def p_5813 : CertifiedPrime :=
  certify 5813 (2) [p_2, p_2, p_1453] (by decide)

def p_5821 : CertifiedPrime :=
  certify 5821 (6) [p_2, p_2, p_3, p_5, p_97] (by decide)

def p_5839 : CertifiedPrime :=
  certify 5839 (6) [p_2, p_3, p_7, p_139] (by decide)

def p_5849 : CertifiedPrime :=
  certify 5849 (3) [p_2, p_2, p_2, p_17, p_43] (by decide)

def p_5851 : CertifiedPrime :=
  certify 5851 (2) [p_2, p_3, p_3, p_5, p_5, p_13] (by decide)

def p_5857 : CertifiedPrime :=
  certify 5857 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_61] (by decide)

def p_5867 : CertifiedPrime :=
  certify 5867 (5) [p_2, p_7, p_419] (by decide)

def p_5869 : CertifiedPrime :=
  certify 5869 (2) [p_2, p_2, p_3, p_3, p_163] (by decide)

def p_5879 : CertifiedPrime :=
  certify 5879 (11) [p_2, p_2939] (by decide)

def p_5897 : CertifiedPrime :=
  certify 5897 (3) [p_2, p_2, p_2, p_11, p_67] (by decide)

def p_5903 : CertifiedPrime :=
  certify 5903 (5) [p_2, p_13, p_227] (by decide)

def p_5923 : CertifiedPrime :=
  certify 5923 (2) [p_2, p_3, p_3, p_7, p_47] (by decide)

def p_5927 : CertifiedPrime :=
  certify 5927 (5) [p_2, p_2963] (by decide)

def p_5939 : CertifiedPrime :=
  certify 5939 (2) [p_2, p_2969] (by decide)

def p_5987 : CertifiedPrime :=
  certify 5987 (2) [p_2, p_41, p_73] (by decide)

def p_6007 : CertifiedPrime :=
  certify 6007 (3) [p_2, p_3, p_7, p_11, p_13] (by decide)

def p_6011 : CertifiedPrime :=
  certify 6011 (2) [p_2, p_5, p_601] (by decide)

def p_6043 : CertifiedPrime :=
  certify 6043 (5) [p_2, p_3, p_19, p_53] (by decide)

def p_6047 : CertifiedPrime :=
  certify 6047 (5) [p_2, p_3023] (by decide)

def p_6053 : CertifiedPrime :=
  certify 6053 (2) [p_2, p_2, p_17, p_89] (by decide)

def p_6067 : CertifiedPrime :=
  certify 6067 (2) [p_2, p_3, p_3, p_337] (by decide)

def p_6089 : CertifiedPrime :=
  certify 6089 (3) [p_2, p_2, p_2, p_761] (by decide)

def p_6091 : CertifiedPrime :=
  certify 6091 (7) [p_2, p_3, p_5, p_7, p_29] (by decide)

def p_6101 : CertifiedPrime :=
  certify 6101 (2) [p_2, p_2, p_5, p_5, p_61] (by decide)

def p_6113 : CertifiedPrime :=
  certify 6113 (3) [p_2, p_2, p_2, p_2, p_2, p_191] (by decide)

def p_6131 : CertifiedPrime :=
  certify 6131 (2) [p_2, p_5, p_613] (by decide)

def p_6133 : CertifiedPrime :=
  certify 6133 (5) [p_2, p_2, p_3, p_7, p_73] (by decide)

def p_6151 : CertifiedPrime :=
  certify 6151 (3) [p_2, p_3, p_5, p_5, p_41] (by decide)

def p_6163 : CertifiedPrime :=
  certify 6163 (3) [p_2, p_3, p_13, p_79] (by decide)

def p_6173 : CertifiedPrime :=
  certify 6173 (2) [p_2, p_2, p_1543] (by decide)

def p_6197 : CertifiedPrime :=
  certify 6197 (2) [p_2, p_2, p_1549] (by decide)

def p_6199 : CertifiedPrime :=
  certify 6199 (3) [p_2, p_3, p_1033] (by decide)

def p_6211 : CertifiedPrime :=
  certify 6211 (2) [p_2, p_3, p_3, p_3, p_5, p_23] (by decide)

def p_6217 : CertifiedPrime :=
  certify 6217 (5) [p_2, p_2, p_2, p_3, p_7, p_37] (by decide)

def p_6257 : CertifiedPrime :=
  certify 6257 (3) [p_2, p_2, p_2, p_2, p_17, p_23] (by decide)

def p_6263 : CertifiedPrime :=
  certify 6263 (5) [p_2, p_31, p_101] (by decide)

def p_6271 : CertifiedPrime :=
  certify 6271 (11) [p_2, p_3, p_5, p_11, p_19] (by decide)

def p_6277 : CertifiedPrime :=
  certify 6277 (2) [p_2, p_2, p_3, p_523] (by decide)

def p_6317 : CertifiedPrime :=
  certify 6317 (2) [p_2, p_2, p_1579] (by decide)

def p_6323 : CertifiedPrime :=
  certify 6323 (2) [p_2, p_29, p_109] (by decide)

def p_6329 : CertifiedPrime :=
  certify 6329 (3) [p_2, p_2, p_2, p_7, p_113] (by decide)

def p_6337 : CertifiedPrime :=
  certify 6337 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11] (by decide)

def p_6353 : CertifiedPrime :=
  certify 6353 (3) [p_2, p_2, p_2, p_2, p_397] (by decide)

def p_6359 : CertifiedPrime :=
  certify 6359 (13) [p_2, p_11, p_17, p_17] (by decide)

def p_6361 : CertifiedPrime :=
  certify 6361 (19) [p_2, p_2, p_2, p_3, p_5, p_53] (by decide)

def p_6379 : CertifiedPrime :=
  certify 6379 (2) [p_2, p_3, p_1063] (by decide)

def p_6389 : CertifiedPrime :=
  certify 6389 (2) [p_2, p_2, p_1597] (by decide)

def p_6397 : CertifiedPrime :=
  certify 6397 (2) [p_2, p_2, p_3, p_13, p_41] (by decide)

def p_6421 : CertifiedPrime :=
  certify 6421 (6) [p_2, p_2, p_3, p_5, p_107] (by decide)

def p_6427 : CertifiedPrime :=
  certify 6427 (3) [p_2, p_3, p_3, p_3, p_7, p_17] (by decide)

def p_6449 : CertifiedPrime :=
  certify 6449 (3) [p_2, p_2, p_2, p_2, p_13, p_31] (by decide)

def p_6451 : CertifiedPrime :=
  certify 6451 (3) [p_2, p_3, p_5, p_5, p_43] (by decide)

def p_6469 : CertifiedPrime :=
  certify 6469 (2) [p_2, p_2, p_3, p_7, p_7, p_11] (by decide)

def p_6481 : CertifiedPrime :=
  certify 6481 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5] (by decide)

def p_6521 : CertifiedPrime :=
  certify 6521 (6) [p_2, p_2, p_2, p_5, p_163] (by decide)

def p_6551 : CertifiedPrime :=
  certify 6551 (17) [p_2, p_5, p_5, p_131] (by decide)

def p_6553 : CertifiedPrime :=
  certify 6553 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_13] (by decide)

def p_6569 : CertifiedPrime :=
  certify 6569 (3) [p_2, p_2, p_2, p_821] (by decide)

def p_6571 : CertifiedPrime :=
  certify 6571 (3) [p_2, p_3, p_3, p_5, p_73] (by decide)

def p_6577 : CertifiedPrime :=
  certify 6577 (5) [p_2, p_2, p_2, p_2, p_3, p_137] (by decide)

def p_6581 : CertifiedPrime :=
  certify 6581 (14) [p_2, p_2, p_5, p_7, p_47] (by decide)

def p_6599 : CertifiedPrime :=
  certify 6599 (13) [p_2, p_3299] (by decide)

def p_6607 : CertifiedPrime :=
  certify 6607 (3) [p_2, p_3, p_3, p_367] (by decide)

def p_6619 : CertifiedPrime :=
  certify 6619 (2) [p_2, p_3, p_1103] (by decide)

def p_6637 : CertifiedPrime :=
  certify 6637 (2) [p_2, p_2, p_3, p_7, p_79] (by decide)

def p_6653 : CertifiedPrime :=
  certify 6653 (2) [p_2, p_2, p_1663] (by decide)

def p_6659 : CertifiedPrime :=
  certify 6659 (2) [p_2, p_3329] (by decide)

def p_6661 : CertifiedPrime :=
  certify 6661 (6) [p_2, p_2, p_3, p_3, p_5, p_37] (by decide)

def p_6689 : CertifiedPrime :=
  certify 6689 (3) [p_2, p_2, p_2, p_2, p_2, p_11, p_19] (by decide)

def p_6701 : CertifiedPrime :=
  certify 6701 (2) [p_2, p_2, p_5, p_5, p_67] (by decide)

def p_6703 : CertifiedPrime :=
  certify 6703 (5) [p_2, p_3, p_1117] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
