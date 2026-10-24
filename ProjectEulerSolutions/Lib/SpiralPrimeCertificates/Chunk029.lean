import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk028

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_123087931 : CertifiedPrime :=
  certify 123087931 (7) [p_2, p_3, p_5, p_7, p_43, p_43, p_317] (by decide)

def p_123121217 : CertifiedPrime :=
  certify 123121217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_73, p_73] (by decide)

def p_123265507 : CertifiedPrime :=
  certify 123265507 (3) [p_2, p_3, p_7, p_13, p_61, p_3701] (by decide)

def p_123487657 : CertifiedPrime :=
  certify 123487657 (5) [p_2, p_2, p_2, p_3, p_463, p_11113] (by decide)

def p_123532111 : CertifiedPrime :=
  certify 123532111 (6) [p_2, p_3, p_3, p_5, p_13, p_19, p_5557] (by decide)

def p_123565457 : CertifiedPrime :=
  certify 123565457 (5) [p_2, p_2, p_2, p_2, p_7, p_7, p_397, p_397] (by decide)

def p_123687763 : CertifiedPrime :=
  certify 123687763 (2) [p_2, p_3, p_11, p_67, p_83, p_337] (by decide)

def p_123865771 : CertifiedPrime :=
  certify 123865771 (12) [p_2, p_3, p_5, p_7, p_31, p_53, p_359] (by decide)

def p_123876901 : CertifiedPrime :=
  certify 123876901 (22) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_53, p_53] (by decide)

def p_124066183 : CertifiedPrime :=
  certify 124066183 (3) [p_2, p_3, p_47, p_79, p_5569] (by decide)

def p_124411717 : CertifiedPrime :=
  certify 124411717 (6) [p_2, p_2, p_3, p_3, p_11, p_11, p_13, p_13, p_13, p_13] (by decide)

def p_124690723 : CertifiedPrime :=
  certify 124690723 (3) [p_2, p_3, p_13, p_859, p_1861] (by decide)

def p_124768901 : CertifiedPrime :=
  certify 124768901 (2) [p_2, p_2, p_5, p_5, p_1117, p_1117] (by decide)

def p_124802413 : CertifiedPrime :=
  certify 124802413 (6) [p_2, p_2, p_3, p_7, p_7, p_19, p_11171] (by decide)

def p_124824757 : CertifiedPrime :=
  certify 124824757 (5) [p_2, p_2, p_3, p_7, p_7, p_19, p_11173] (by decide)

def p_124958863 : CertifiedPrime :=
  certify 124958863 (7) [p_2, p_3, p_3, p_3, p_3, p_3, p_7, p_23, p_1597] (by decide)

def p_125160157 : CertifiedPrime :=
  certify 125160157 (5) [p_2, p_2, p_3, p_3, p_11, p_113, p_2797] (by decide)

def p_125428801 : CertifiedPrime :=
  certify 125428801 (31) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_3733] (by decide)

def p_125440001 : CertifiedPrime :=
  certify 125440001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7] (by decide)

def p_125529617 : CertifiedPrime :=
  certify 125529617 (3) [p_2, p_2, p_2, p_2, p_2801, p_2801] (by decide)

def p_125608057 : CertifiedPrime :=
  certify 125608057 (5) [p_2, p_2, p_2, p_3, p_7, p_467, p_1601] (by decide)

def p_125742583 : CertifiedPrime :=
  certify 125742583 (5) [p_2, p_3, p_3, p_7, p_89, p_11213] (by decide)

def p_125765011 : CertifiedPrime :=
  certify 125765011 (13) [p_2, p_3, p_3, p_5, p_7, p_89, p_2243] (by decide)

def p_125877181 : CertifiedPrime :=
  certify 125877181 (18) [p_2, p_2, p_3, p_5, p_11, p_13, p_17, p_863] (by decide)

def p_125899621 : CertifiedPrime :=
  certify 125899621 (6) [p_2, p_2, p_3, p_5, p_7, p_7, p_11, p_17, p_229] (by decide)

def p_126213991 : CertifiedPrime :=
  certify 126213991 (15) [p_2, p_3, p_5, p_7, p_41, p_107, p_137] (by decide)

def p_126247697 : CertifiedPrime :=
  certify 126247697 (3) [p_2, p_2, p_2, p_2, p_53, p_53, p_53, p_53] (by decide)

def p_126416293 : CertifiedPrime :=
  certify 126416293 (5) [p_2, p_2, p_3, p_937, p_11243] (by decide)

def p_126427537 : CertifiedPrime :=
  certify 126427537 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_937, p_937] (by decide)

def p_126472517 : CertifiedPrime :=
  certify 126472517 (2) [p_2, p_2, p_5623, p_5623] (by decide)

def p_126697537 : CertifiedPrime :=
  certify 126697537 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_67, p_67] (by decide)

def p_126708793 : CertifiedPrime :=
  certify 126708793 (5) [p_2, p_2, p_2, p_3, p_7, p_67, p_11257] (by decide)

def p_126888961 : CertifiedPrime :=
  certify 126888961 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_11, p_751] (by decide)

def p_127012901 : CertifiedPrime :=
  certify 127012901 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_7, p_7, p_23, p_23] (by decide)

def p_127159453 : CertifiedPrime :=
  certify 127159453 (2) [p_2, p_2, p_3, p_3, p_7, p_179, p_2819] (by decide)

def p_127294807 : CertifiedPrime :=
  certify 127294807 (3) [p_2, p_3, p_3761, p_5641] (by decide)

def p_127373797 : CertifiedPrime :=
  certify 127373797 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_11, p_11, p_19, p_19] (by decide)

def p_127769113 : CertifiedPrime :=
  certify 127769113 (5) [p_2, p_2, p_2, p_3, p_3, p_89, p_127, p_157] (by decide)

def p_127791721 : CertifiedPrime :=
  certify 127791721 (13) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_17, p_19, p_157] (by decide)

def p_128142401 : CertifiedPrime :=
  certify 128142401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_283, p_283] (by decide)

def p_128244301 : CertifiedPrime :=
  certify 128244301 (7) [p_2, p_2, p_3, p_5, p_5, p_19, p_149, p_151] (by decide)

def p_128266951 : CertifiedPrime :=
  certify 128266951 (7) [p_2, p_3, p_5, p_5, p_7, p_151, p_809] (by decide)

def p_128368901 : CertifiedPrime :=
  certify 128368901 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_103, p_103] (by decide)

def p_128504897 : CertifiedPrime :=
  certify 128504897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_109, p_109] (by decide)

def p_128516233 : CertifiedPrime :=
  certify 128516233 (7) [p_2, p_2, p_2, p_3, p_13, p_109, p_3779] (by decide)

def p_128674993 : CertifiedPrime :=
  certify 128674993 (7) [p_2, p_2, p_2, p_2, p_3, p_19, p_199, p_709] (by decide)

def p_128731717 : CertifiedPrime :=
  certify 128731717 (5) [p_2, p_2, p_3, p_3, p_31, p_31, p_61, p_61] (by decide)

def p_128822501 : CertifiedPrime :=
  certify 128822501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_227, p_227] (by decide)

def p_129083683 : CertifiedPrime :=
  certify 129083683 (2) [p_2, p_3, p_7, p_13, p_19, p_23, p_541] (by decide)

def p_129140497 : CertifiedPrime :=
  certify 129140497 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_947, p_947] (by decide)

def p_129220057 : CertifiedPrime :=
  certify 129220057 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_7, p_7, p_29, p_421] (by decide)

def p_129367877 : CertifiedPrime :=
  certify 129367877 (3) [p_2, p_2, p_11, p_11, p_11, p_11, p_47, p_47] (by decide)

def p_129402001 : CertifiedPrime :=
  certify 129402001 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_7, p_13, p_79] (by decide)

def p_129960001 : CertifiedPrime :=
  certify 129960001 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_19, p_19] (by decide)

def p_130039813 : CertifiedPrime :=
  certify 130039813 (2) [p_2, p_2, p_3, p_3, p_7, p_181, p_2851] (by decide)

def p_130051217 : CertifiedPrime :=
  certify 130051217 (3) [p_2, p_2, p_2, p_2, p_2851, p_2851] (by decide)

def p_130176691 : CertifiedPrime :=
  certify 130176691 (2) [p_2, p_3, p_5, p_7, p_163, p_3803] (by decide)

def p_130313641 : CertifiedPrime :=
  certify 130313641 (11) [p_2, p_2, p_2, p_3, p_5, p_761, p_1427] (by decide)

def p_130325057 : CertifiedPrime :=
  certify 130325057 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1427, p_1427] (by decide)

def p_130519201 : CertifiedPrime :=
  certify 130519201 (19) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_17, p_457] (by decide)

def p_130633471 : CertifiedPrime :=
  certify 130633471 (17) [p_2, p_3, p_3, p_5, p_11, p_127, p_1039] (by decide)

def p_130782097 : CertifiedPrime :=
  certify 130782097 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_953, p_953] (by decide)

def p_130839283 : CertifiedPrime :=
  certify 130839283 (2) [p_2, p_3, p_3, p_7, p_19, p_31, p_41, p_43] (by decide)

def p_130907923 : CertifiedPrime :=
  certify 130907923 (3) [p_2, p_3, p_17, p_673, p_1907] (by decide)

def p_131113951 : CertifiedPrime :=
  certify 131113951 (13) [p_2, p_3, p_5, p_5, p_11, p_229, p_347] (by decide)

def p_131239937 : CertifiedPrime :=
  certify 131239937 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_179, p_179] (by decide)

def p_131274307 : CertifiedPrime :=
  certify 131274307 (5) [p_2, p_3, p_3, p_17, p_19, p_67, p_337] (by decide)

def p_131457691 : CertifiedPrime :=
  certify 131457691 (10) [p_2, p_3, p_3, p_5, p_7, p_7, p_13, p_2293] (by decide)

def p_131480623 : CertifiedPrime :=
  certify 131480623 (5) [p_2, p_3, p_3, p_7, p_7, p_13, p_11467] (by decide)

def p_131618257 : CertifiedPrime :=
  certify 131618257 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_149, p_239] (by decide)

def p_131790401 : CertifiedPrime :=
  certify 131790401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_41, p_41] (by decide)

def p_132238501 : CertifiedPrime :=
  certify 132238501 (2) [p_2, p_2, p_3, p_5, p_5, p_5, p_23, p_3833] (by decide)

def p_132514633 : CertifiedPrime :=
  certify 132514633 (5) [p_2, p_2, p_2, p_3, p_3, p_1279, p_1439] (by decide)

def p_132572197 : CertifiedPrime :=
  certify 132572197 (2) [p_2, p_2, p_3, p_3, p_19, p_19, p_101, p_101] (by decide)

def p_132710401 : CertifiedPrime :=
  certify 132710401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_132721921 : CertifiedPrime :=
  certify 132721921 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_41, p_281] (by decide)

def p_132768007 : CertifiedPrime :=
  certify 132768007 (3) [p_2, p_3, p_7, p_23, p_167, p_823] (by decide)

def p_132791053 : CertifiedPrime :=
  certify 132791053 (2) [p_2, p_2, p_3, p_23, p_43, p_67, p_167] (by decide)

def p_133206223 : CertifiedPrime :=
  certify 133206223 (3) [p_2, p_3, p_29, p_199, p_3847] (by decide)

def p_133252393 : CertifiedPrime :=
  certify 133252393 (5) [p_2, p_2, p_2, p_3, p_7, p_13, p_17, p_37, p_97] (by decide)

def p_133321663 : CertifiedPrime :=
  certify 133321663 (6) [p_2, p_3, p_3, p_23, p_251, p_1283] (by decide)

def p_133390951 : CertifiedPrime :=
  certify 133390951 (23) [p_2, p_3, p_5, p_5, p_7, p_11, p_11549] (by decide)

def p_133529581 : CertifiedPrime :=
  certify 133529581 (7) [p_2, p_2, p_3, p_3, p_3, p_5, p_107, p_2311] (by decide)

def p_133633601 : CertifiedPrime :=
  certify 133633601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_17, p_17] (by decide)

def p_133691407 : CertifiedPrime :=
  certify 133691407 (3) [p_2, p_3, p_31, p_41, p_47, p_373] (by decide)

def p_133760791 : CertifiedPrime :=
  certify 133760791 (6) [p_2, p_3, p_3, p_5, p_257, p_5783] (by decide)

def p_133772357 : CertifiedPrime :=
  certify 133772357 (2) [p_2, p_2, p_5783, p_5783] (by decide)

def p_133807057 : CertifiedPrime :=
  certify 133807057 (5) [p_2, p_2, p_2, p_2, p_3, p_43, p_241, p_269] (by decide)

def p_133969051 : CertifiedPrime :=
  certify 133969051 (10) [p_2, p_3, p_3, p_5, p_5, p_463, p_643] (by decide)

def p_134038507 : CertifiedPrime :=
  certify 134038507 (2) [p_2, p_3, p_7, p_17, p_227, p_827] (by decide)

def p_134235397 : CertifiedPrime :=
  certify 134235397 (5) [p_2, p_2, p_3, p_3, p_1931, p_1931] (by decide)

def p_134467217 : CertifiedPrime :=
  certify 134467217 (5) [p_2, p_2, p_2, p_2, p_13, p_13, p_223, p_223] (by decide)

def p_134560001 : CertifiedPrime :=
  certify 134560001 (15) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_29, p_29] (by decide)

def p_134571601 : CertifiedPrime :=
  certify 134571601 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_29, p_1289] (by decide)

def p_134780491 : CertifiedPrime :=
  certify 134780491 (14) [p_2, p_3, p_3, p_3, p_5, p_13, p_19, p_43, p_47] (by decide)

def p_134792101 : CertifiedPrime :=
  certify 134792101 (10) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_43, p_43] (by decide)

def p_134873383 : CertifiedPrime :=
  certify 134873383 (3) [p_2, p_3, p_7, p_7, p_79, p_5807] (by decide)

def p_135059263 : CertifiedPrime :=
  certify 135059263 (3) [p_2, p_3, p_13, p_149, p_11621] (by decide)

def p_135163877 : CertifiedPrime :=
  certify 135163877 (2) [p_2, p_2, p_5813, p_5813] (by decide)

def p_135198757 : CertifiedPrime :=
  certify 135198757 (5) [p_2, p_2, p_3, p_3, p_7, p_11, p_17, p_19, p_151] (by decide)

def p_135501241 : CertifiedPrime :=
  certify 135501241 (11) [p_2, p_2, p_2, p_3, p_5, p_7, p_97, p_1663] (by decide)

def p_135547807 : CertifiedPrime :=
  certify 135547807 (3) [p_2, p_3, p_3881, p_5821] (by decide)

def p_135710851 : CertifiedPrime :=
  certify 135710851 (3) [p_2, p_3, p_5, p_5, p_11, p_233, p_353] (by decide)

def p_135722501 : CertifiedPrime :=
  certify 135722501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_233, p_233] (by decide)

def p_135850681 : CertifiedPrime :=
  certify 135850681 (22) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_31, p_37, p_47] (by decide)

def p_136130557 : CertifiedPrime :=
  certify 136130557 (2) [p_2, p_2, p_3, p_2917, p_3889] (by decide)

def p_136282277 : CertifiedPrime :=
  certify 136282277 (2) [p_2, p_2, p_13, p_13, p_449, p_449] (by decide)

def p_136340653 : CertifiedPrime :=
  certify 136340653 (5) [p_2, p_2, p_3, p_7, p_139, p_11677] (by decide)

def p_136422401 : CertifiedPrime :=
  certify 136422401 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_73, p_73] (by decide)

def p_136527541 : CertifiedPrime :=
  certify 136527541 (2) [p_2, p_2, p_3, p_5, p_19, p_23, p_41, p_127] (by decide)

def p_136691173 : CertifiedPrime :=
  certify 136691173 (2) [p_2, p_2, p_3, p_3, p_3, p_37, p_79, p_433] (by decide)

def p_136737943 : CertifiedPrime :=
  certify 136737943 (3) [p_2, p_3, p_11, p_1063, p_1949] (by decide)

def p_136901701 : CertifiedPrime :=
  certify 136901701 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_11701] (by decide)

def p_137018731 : CertifiedPrime :=
  certify 137018731 (3) [p_2, p_3, p_5, p_1951, p_2341] (by decide)

def p_137229511 : CertifiedPrime :=
  certify 137229511 (3) [p_2, p_3, p_5, p_11, p_71, p_5857] (by decide)

def p_137299807 : CertifiedPrime :=
  certify 137299807 (5) [p_2, p_3, p_3, p_3, p_7, p_31, p_11717] (by decide)

def p_137358401 : CertifiedPrime :=
  certify 137358401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_293, p_293] (by decide)

def p_137463901 : CertifiedPrime :=
  certify 137463901 (6) [p_2, p_2, p_3, p_5, p_5, p_7, p_67, p_977] (by decide)

def p_137792383 : CertifiedPrime :=
  certify 137792383 (5) [p_2, p_3, p_7, p_13, p_43, p_5869] (by decide)

def p_138062501 : CertifiedPrime :=
  certify 138062501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_47, p_47] (by decide)

def p_138215293 : CertifiedPrime :=
  certify 138215293 (2) [p_2, p_2, p_3, p_2939, p_3919] (by decide)

def p_138309361 : CertifiedPrime :=
  certify 138309361 (26) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_19, p_619] (by decide)

def p_138379933 : CertifiedPrime :=
  certify 138379933 (2) [p_2, p_2, p_3, p_3, p_17, p_173, p_1307] (by decide)

def p_138438757 : CertifiedPrime :=
  certify 138438757 (2) [p_2, p_2, p_3, p_3, p_37, p_37, p_53, p_53] (by decide)

def p_138450523 : CertifiedPrime :=
  certify 138450523 (3) [p_2, p_3, p_7, p_37, p_41, p_41, p_53] (by decide)

def p_138709507 : CertifiedPrime :=
  certify 138709507 (3) [p_2, p_3, p_13, p_151, p_11777] (by decide)

def p_138780181 : CertifiedPrime :=
  certify 138780181 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_11, p_17, p_19, p_31] (by decide)

def p_138874441 : CertifiedPrime :=
  certify 138874441 (11) [p_2, p_2, p_2, p_3, p_5, p_491, p_2357] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
