import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk049

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_660515701 : CertifiedPrime :=
  certify 660515701 (2) [p_2, p_2, p_3, p_5, p_5, p_13, p_257, p_659] (by decide)

def p_660695617 : CertifiedPrime :=
  certify 660695617 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7, p_17, p_17] (by decide)

def p_660978391 : CertifiedPrime :=
  certify 660978391 (13) [p_2, p_3, p_5, p_47, p_547, p_857] (by decide)

def p_661132657 : CertifiedPrime :=
  certify 661132657 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1607, p_2857] (by decide)

def p_661209797 : CertifiedPrime :=
  certify 661209797 (2) [p_2, p_2, p_13, p_13, p_23, p_23, p_43, p_43] (by decide)

def p_661492681 : CertifiedPrime :=
  certify 661492681 (7) [p_2, p_2, p_2, p_3, p_5, p_643, p_8573] (by decide)

def p_661724177 : CertifiedPrime :=
  certify 661724177 (3) [p_2, p_2, p_2, p_2, p_59, p_59, p_109, p_109] (by decide)

def p_662341697 : CertifiedPrime :=
  certify 662341697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_3217, p_3217] (by decide)

def p_662521861 : CertifiedPrime :=
  certify 662521861 (14) [p_2, p_2, p_3, p_3, p_5, p_7, p_11, p_13, p_3677] (by decide)

def p_662753537 : CertifiedPrime :=
  certify 662753537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1609, p_1609] (by decide)

def p_662882263 : CertifiedPrime :=
  certify 662882263 (3) [p_2, p_3, p_7, p_613, p_25747] (by decide)

def p_663268517 : CertifiedPrime :=
  certify 663268517 (2) [p_2, p_2, p_79, p_79, p_163, p_163] (by decide)

def p_663371537 : CertifiedPrime :=
  certify 663371537 (3) [p_2, p_2, p_2, p_2, p_47, p_47, p_137, p_137] (by decide)

def p_664221757 : CertifiedPrime :=
  certify 664221757 (2) [p_2, p_2, p_3, p_11, p_11, p_17, p_71, p_379] (by decide)

def p_664582621 : CertifiedPrime :=
  certify 664582621 (2) [p_2, p_2, p_3, p_5, p_13, p_661, p_1289] (by decide)

def p_664917797 : CertifiedPrime :=
  certify 664917797 (2) [p_2, p_2, p_12893, p_12893] (by decide)

def p_665124101 : CertifiedPrime :=
  certify 665124101 (2) [p_2, p_2, p_5, p_5, p_2579, p_2579] (by decide)

def p_665949637 : CertifiedPrime :=
  certify 665949637 (6) [p_2, p_2, p_3, p_3, p_11, p_11, p_17, p_17, p_23, p_23] (by decide)

def p_667008103 : CertifiedPrime :=
  certify 667008103 (3) [p_2, p_3, p_37, p_349, p_8609] (by decide)

def p_667188901 : CertifiedPrime :=
  certify 667188901 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_7, p_41, p_41] (by decide)

def p_667473061 : CertifiedPrime :=
  certify 667473061 (6) [p_2, p_2, p_3, p_5, p_2153, p_5167] (by decide)

def p_667834807 : CertifiedPrime :=
  certify 667834807 (3) [p_2, p_3, p_43, p_59, p_73, p_601] (by decide)

def p_667938181 : CertifiedPrime :=
  certify 667938181 (6) [p_2, p_2, p_3, p_5, p_7, p_13, p_71, p_1723] (by decide)

def p_668429317 : CertifiedPrime :=
  certify 668429317 (2) [p_2, p_2, p_3, p_3, p_31, p_31, p_139, p_139] (by decide)

def p_668455171 : CertifiedPrime :=
  certify 668455171 (3) [p_2, p_3, p_5, p_31, p_139, p_5171] (by decide)

def p_668610307 : CertifiedPrime :=
  certify 668610307 (5) [p_2, p_3, p_3, p_7, p_13, p_13, p_17, p_1847] (by decide)

def p_669024091 : CertifiedPrime :=
  certify 669024091 (7) [p_2, p_3, p_3, p_3, p_5, p_7, p_479, p_739] (by decide)

def p_669334513 : CertifiedPrime :=
  certify 669334513 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_11, p_41, p_631] (by decide)

def p_669386257 : CertifiedPrime :=
  certify 669386257 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_11, p_25873] (by decide)

def p_669541501 : CertifiedPrime :=
  certify 669541501 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_23, p_6469] (by decide)

def p_669981457 : CertifiedPrime :=
  certify 669981457 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_719, p_719] (by decide)

def p_670084997 : CertifiedPrime :=
  certify 670084997 (2) [p_2, p_2, p_7, p_7, p_43, p_43, p_43, p_43] (by decide)

def p_670602817 : CertifiedPrime :=
  certify 670602817 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13, p_83, p_83] (by decide)

def p_670732303 : CertifiedPrime :=
  certify 670732303 (3) [p_2, p_3, p_23, p_89, p_97, p_563] (by decide)

def p_670810001 : CertifiedPrime :=
  certify 670810001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_37, p_37] (by decide)

def p_670887703 : CertifiedPrime :=
  certify 670887703 (13) [p_2, p_3, p_3, p_59, p_439, p_1439] (by decide)

def p_672053777 : CertifiedPrime :=
  certify 672053777 (3) [p_2, p_2, p_2, p_2, p_6481, p_6481] (by decide)

def p_672287113 : CertifiedPrime :=
  certify 672287113 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_43, p_67, p_463] (by decide)

def p_672572357 : CertifiedPrime :=
  certify 672572357 (2) [p_2, p_2, p_12967, p_12967] (by decide)

def p_673065193 : CertifiedPrime :=
  certify 673065193 (5) [p_2, p_2, p_2, p_3, p_23, p_47, p_25943] (by decide)

def p_673402501 : CertifiedPrime :=
  certify 673402501 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_173, p_173] (by decide)

def p_673687981 : CertifiedPrime :=
  certify 673687981 (10) [p_2, p_2, p_3, p_3, p_5, p_7, p_29, p_103, p_179] (by decide)

def p_673739893 : CertifiedPrime :=
  certify 673739893 (13) [p_2, p_2, p_3, p_3, p_7, p_101, p_103, p_257] (by decide)

def p_674362993 : CertifiedPrime :=
  certify 674362993 (11) [p_2, p_2, p_2, p_2, p_3, p_541, p_25969] (by decide)

def p_675246211 : CertifiedPrime :=
  certify 675246211 (3) [p_2, p_3, p_5, p_61, p_71, p_5197] (by decide)

def p_675454111 : CertifiedPrime :=
  certify 675454111 (3) [p_2, p_3, p_5, p_23, p_113, p_8663] (by decide)

def p_675870007 : CertifiedPrime :=
  certify 675870007 (3) [p_2, p_3, p_7, p_619, p_25997] (by decide)

def p_675922003 : CertifiedPrime :=
  certify 675922003 (3) [p_2, p_3, p_7, p_619, p_25999] (by decide)

def p_676546111 : CertifiedPrime :=
  certify 676546111 (43) [p_2, p_3, p_3, p_5, p_17, p_17, p_19, p_37, p_37] (by decide)

def p_676702183 : CertifiedPrime :=
  certify 676702183 (3) [p_2, p_3, p_13, p_23, p_29, p_13007] (by decide)

def p_676832257 : CertifiedPrime :=
  certify 676832257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_271, p_271] (by decide)

def p_677014381 : CertifiedPrime :=
  certify 677014381 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_7, p_59, p_1301] (by decide)

def p_677248577 : CertifiedPrime :=
  certify 677248577 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_3253, p_3253] (by decide)

def p_677899333 : CertifiedPrime :=
  certify 677899333 (2) [p_2, p_2, p_3, p_3, p_11, p_23, p_263, p_283] (by decide)

def p_678602501 : CertifiedPrime :=
  certify 678602501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_521, p_521] (by decide)

def p_679332097 : CertifiedPrime :=
  certify 679332097 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_181, p_181] (by decide)

def p_679436357 : CertifiedPrime :=
  certify 679436357 (2) [p_2, p_2, p_13033, p_13033] (by decide)

def p_679827403 : CertifiedPrime :=
  certify 679827403 (2) [p_2, p_3, p_3, p_2897, p_13037] (by decide)

def p_680088163 : CertifiedPrime :=
  certify 680088163 (2) [p_2, p_3, p_13, p_17, p_59, p_8693] (by decide)

def p_680453311 : CertifiedPrime :=
  certify 680453311 (3) [p_2, p_3, p_5, p_37, p_47, p_13043] (by decide)

def p_680609833 : CertifiedPrime :=
  certify 680609833 (23) [p_2, p_2, p_2, p_3, p_7, p_1087, p_3727] (by decide)

def p_680896837 : CertifiedPrime :=
  certify 680896837 (2) [p_2, p_2, p_3, p_3, p_4349, p_4349] (by decide)

def p_681210001 : CertifiedPrime :=
  certify 681210001 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_29, p_29] (by decide)

def p_681340507 : CertifiedPrime :=
  certify 681340507 (2) [p_2, p_3, p_7, p_11, p_31, p_113, p_421] (by decide)

def p_681705991 : CertifiedPrime :=
  certify 681705991 (6) [p_2, p_3, p_3, p_3, p_5, p_7, p_373, p_967] (by decide)

def p_681940997 : CertifiedPrime :=
  certify 681940997 (2) [p_2, p_2, p_11, p_11, p_1187, p_1187] (by decide)

def p_682176043 : CertifiedPrime :=
  certify 682176043 (14) [p_2, p_3, p_3, p_1451, p_26119] (by decide)

def p_682254401 : CertifiedPrime :=
  certify 682254401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_653, p_653] (by decide)

def p_682463377 : CertifiedPrime :=
  certify 682463377 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_311, p_311] (by decide)

def p_682489501 : CertifiedPrime :=
  certify 682489501 (6) [p_2, p_2, p_3, p_5, p_5, p_5, p_7, p_11, p_19, p_311] (by decide)

def p_682750771 : CertifiedPrime :=
  certify 682750771 (12) [p_2, p_3, p_5, p_13, p_17, p_29, p_53, p_67] (by decide)

def p_682985957 : CertifiedPrime :=
  certify 682985957 (2) [p_2, p_2, p_73, p_73, p_179, p_179] (by decide)

def p_683299601 : CertifiedPrime :=
  certify 683299601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1307, p_1307] (by decide)

def p_683508737 : CertifiedPrime :=
  certify 683508737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_43, p_43] (by decide)

def p_683587171 : CertifiedPrime :=
  certify 683587171 (2) [p_2, p_3, p_3, p_5, p_7, p_17, p_83, p_769] (by decide)

def p_684031717 : CertifiedPrime :=
  certify 684031717 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_1453, p_1453] (by decide)

def p_684345601 : CertifiedPrime :=
  certify 684345601 (17) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_109, p_109] (by decide)

def p_684476407 : CertifiedPrime :=
  certify 684476407 (5) [p_2, p_3, p_3, p_3, p_3, p_17, p_19, p_103, p_127] (by decide)

def p_684947413 : CertifiedPrime :=
  certify 684947413 (7) [p_2, p_2, p_3, p_3, p_727, p_26171] (by decide)

def p_685575673 : CertifiedPrime :=
  certify 685575673 (5) [p_2, p_2, p_2, p_3, p_1091, p_26183] (by decide)

def p_685732783 : CertifiedPrime :=
  certify 685732783 (3) [p_2, p_3, p_7, p_29, p_43, p_13093] (by decide)

def p_686125637 : CertifiedPrime :=
  certify 686125637 (2) [p_2, p_2, p_7, p_7, p_1871, p_1871] (by decide)

def p_686204221 : CertifiedPrime :=
  certify 686204221 (6) [p_2, p_2, p_3, p_5, p_13, p_13, p_31, p_37, p_59] (by decide)

def p_686518603 : CertifiedPrime :=
  certify 686518603 (3) [p_2, p_3, p_7, p_11, p_19, p_197, p_397] (by decide)

def p_686833057 : CertifiedPrime :=
  certify 686833057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_13, p_73, p_359] (by decide)

def p_687698177 : CertifiedPrime :=
  certify 687698177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_149, p_149] (by decide)

def p_688144057 : CertifiedPrime :=
  certify 688144057 (5) [p_2, p_2, p_2, p_3, p_37, p_709, p_1093] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
