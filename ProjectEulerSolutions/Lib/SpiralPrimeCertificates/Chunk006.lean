import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk005

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_6719 : CertifiedPrime :=
  certify 6719 (11) [p_2, p_3359] (by decide)

def p_6733 : CertifiedPrime :=
  certify 6733 (2) [p_2, p_2, p_3, p_3, p_11, p_17] (by decide)

def p_6737 : CertifiedPrime :=
  certify 6737 (3) [p_2, p_2, p_2, p_2, p_421] (by decide)

def p_6761 : CertifiedPrime :=
  certify 6761 (3) [p_2, p_2, p_2, p_5, p_13, p_13] (by decide)

def p_6791 : CertifiedPrime :=
  certify 6791 (7) [p_2, p_5, p_7, p_97] (by decide)

def p_6793 : CertifiedPrime :=
  certify 6793 (10) [p_2, p_2, p_2, p_3, p_283] (by decide)

def p_6827 : CertifiedPrime :=
  certify 6827 (2) [p_2, p_3413] (by decide)

def p_6833 : CertifiedPrime :=
  certify 6833 (3) [p_2, p_2, p_2, p_2, p_7, p_61] (by decide)

def p_6841 : CertifiedPrime :=
  certify 6841 (22) [p_2, p_2, p_2, p_3, p_3, p_5, p_19] (by decide)

def p_6857 : CertifiedPrime :=
  certify 6857 (3) [p_2, p_2, p_2, p_857] (by decide)

def p_6871 : CertifiedPrime :=
  certify 6871 (3) [p_2, p_3, p_5, p_229] (by decide)

def p_6883 : CertifiedPrime :=
  certify 6883 (2) [p_2, p_3, p_31, p_37] (by decide)

def p_6907 : CertifiedPrime :=
  certify 6907 (2) [p_2, p_3, p_1151] (by decide)

def p_6911 : CertifiedPrime :=
  certify 6911 (7) [p_2, p_5, p_691] (by decide)

def p_6917 : CertifiedPrime :=
  certify 6917 (2) [p_2, p_2, p_7, p_13, p_19] (by decide)

def p_6947 : CertifiedPrime :=
  certify 6947 (2) [p_2, p_23, p_151] (by decide)

def p_6949 : CertifiedPrime :=
  certify 6949 (2) [p_2, p_2, p_3, p_3, p_193] (by decide)

def p_6997 : CertifiedPrime :=
  certify 6997 (5) [p_2, p_2, p_3, p_11, p_53] (by decide)

def p_7001 : CertifiedPrime :=
  certify 7001 (3) [p_2, p_2, p_2, p_5, p_5, p_5, p_7] (by decide)

def p_7013 : CertifiedPrime :=
  certify 7013 (2) [p_2, p_2, p_1753] (by decide)

def p_7027 : CertifiedPrime :=
  certify 7027 (2) [p_2, p_3, p_1171] (by decide)

def p_7039 : CertifiedPrime :=
  certify 7039 (3) [p_2, p_3, p_3, p_17, p_23] (by decide)

def p_7043 : CertifiedPrime :=
  certify 7043 (2) [p_2, p_7, p_503] (by decide)

def p_7057 : CertifiedPrime :=
  certify 7057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7] (by decide)

def p_7079 : CertifiedPrime :=
  certify 7079 (7) [p_2, p_3539] (by decide)

def p_7103 : CertifiedPrime :=
  certify 7103 (5) [p_2, p_53, p_67] (by decide)

def p_7109 : CertifiedPrime :=
  certify 7109 (2) [p_2, p_2, p_1777] (by decide)

def p_7127 : CertifiedPrime :=
  certify 7127 (5) [p_2, p_7, p_509] (by decide)

def p_7159 : CertifiedPrime :=
  certify 7159 (3) [p_2, p_3, p_1193] (by decide)

def p_7177 : CertifiedPrime :=
  certify 7177 (10) [p_2, p_2, p_2, p_3, p_13, p_23] (by decide)

def p_7187 : CertifiedPrime :=
  certify 7187 (2) [p_2, p_3593] (by decide)

def p_7193 : CertifiedPrime :=
  certify 7193 (3) [p_2, p_2, p_2, p_29, p_31] (by decide)

def p_7207 : CertifiedPrime :=
  certify 7207 (3) [p_2, p_3, p_1201] (by decide)

def p_7211 : CertifiedPrime :=
  certify 7211 (2) [p_2, p_5, p_7, p_103] (by decide)

def p_7213 : CertifiedPrime :=
  certify 7213 (5) [p_2, p_2, p_3, p_601] (by decide)

def p_7219 : CertifiedPrime :=
  certify 7219 (2) [p_2, p_3, p_3, p_401] (by decide)

def p_7229 : CertifiedPrime :=
  certify 7229 (2) [p_2, p_2, p_13, p_139] (by decide)

def p_7237 : CertifiedPrime :=
  certify 7237 (2) [p_2, p_2, p_3, p_3, p_3, p_67] (by decide)

def p_7243 : CertifiedPrime :=
  certify 7243 (2) [p_2, p_3, p_17, p_71] (by decide)

def p_7247 : CertifiedPrime :=
  certify 7247 (5) [p_2, p_3623] (by decide)

def p_7253 : CertifiedPrime :=
  certify 7253 (2) [p_2, p_2, p_7, p_7, p_37] (by decide)

def p_7283 : CertifiedPrime :=
  certify 7283 (2) [p_2, p_11, p_331] (by decide)

def p_7297 : CertifiedPrime :=
  certify 7297 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_19] (by decide)

def p_7321 : CertifiedPrime :=
  certify 7321 (7) [p_2, p_2, p_2, p_3, p_5, p_61] (by decide)

def p_7333 : CertifiedPrime :=
  certify 7333 (6) [p_2, p_2, p_3, p_13, p_47] (by decide)

def p_7351 : CertifiedPrime :=
  certify 7351 (6) [p_2, p_3, p_5, p_5, p_7, p_7] (by decide)

def p_7369 : CertifiedPrime :=
  certify 7369 (7) [p_2, p_2, p_2, p_3, p_307] (by decide)

def p_7393 : CertifiedPrime :=
  certify 7393 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11] (by decide)

def p_7411 : CertifiedPrime :=
  certify 7411 (2) [p_2, p_3, p_5, p_13, p_19] (by decide)

def p_7417 : CertifiedPrime :=
  certify 7417 (5) [p_2, p_2, p_2, p_3, p_3, p_103] (by decide)

def p_7433 : CertifiedPrime :=
  certify 7433 (3) [p_2, p_2, p_2, p_929] (by decide)

def p_7451 : CertifiedPrime :=
  certify 7451 (2) [p_2, p_5, p_5, p_149] (by decide)

def p_7457 : CertifiedPrime :=
  certify 7457 (3) [p_2, p_2, p_2, p_2, p_2, p_233] (by decide)

def p_7459 : CertifiedPrime :=
  certify 7459 (2) [p_2, p_3, p_11, p_113] (by decide)

def p_7499 : CertifiedPrime :=
  certify 7499 (2) [p_2, p_23, p_163] (by decide)

def p_7507 : CertifiedPrime :=
  certify 7507 (2) [p_2, p_3, p_3, p_3, p_139] (by decide)

def p_7537 : CertifiedPrime :=
  certify 7537 (7) [p_2, p_2, p_2, p_2, p_3, p_157] (by decide)

def p_7547 : CertifiedPrime :=
  certify 7547 (2) [p_2, p_7, p_7, p_7, p_11] (by decide)

def p_7549 : CertifiedPrime :=
  certify 7549 (2) [p_2, p_2, p_3, p_17, p_37] (by decide)

def p_7559 : CertifiedPrime :=
  certify 7559 (13) [p_2, p_3779] (by decide)

def p_7573 : CertifiedPrime :=
  certify 7573 (2) [p_2, p_2, p_3, p_631] (by decide)

def p_7583 : CertifiedPrime :=
  certify 7583 (5) [p_2, p_17, p_223] (by decide)

def p_7591 : CertifiedPrime :=
  certify 7591 (6) [p_2, p_3, p_5, p_11, p_23] (by decide)

def p_7607 : CertifiedPrime :=
  certify 7607 (5) [p_2, p_3803] (by decide)

def p_7621 : CertifiedPrime :=
  certify 7621 (2) [p_2, p_2, p_3, p_5, p_127] (by decide)

def p_7639 : CertifiedPrime :=
  certify 7639 (7) [p_2, p_3, p_19, p_67] (by decide)

def p_7649 : CertifiedPrime :=
  certify 7649 (3) [p_2, p_2, p_2, p_2, p_2, p_239] (by decide)

def p_7673 : CertifiedPrime :=
  certify 7673 (3) [p_2, p_2, p_2, p_7, p_137] (by decide)

def p_7681 : CertifiedPrime :=
  certify 7681 (17) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5] (by decide)

def p_7703 : CertifiedPrime :=
  certify 7703 (5) [p_2, p_3851] (by decide)

def p_7717 : CertifiedPrime :=
  certify 7717 (2) [p_2, p_2, p_3, p_643] (by decide)

def p_7727 : CertifiedPrime :=
  certify 7727 (5) [p_2, p_3863] (by decide)

def p_7741 : CertifiedPrime :=
  certify 7741 (7) [p_2, p_2, p_3, p_3, p_5, p_43] (by decide)

def p_7753 : CertifiedPrime :=
  certify 7753 (10) [p_2, p_2, p_2, p_3, p_17, p_19] (by decide)

def p_7757 : CertifiedPrime :=
  certify 7757 (2) [p_2, p_2, p_7, p_277] (by decide)

def p_7793 : CertifiedPrime :=
  certify 7793 (3) [p_2, p_2, p_2, p_2, p_487] (by decide)

def p_7817 : CertifiedPrime :=
  certify 7817 (3) [p_2, p_2, p_2, p_977] (by decide)

def p_7823 : CertifiedPrime :=
  certify 7823 (5) [p_2, p_3911] (by decide)

def p_7829 : CertifiedPrime :=
  certify 7829 (2) [p_2, p_2, p_19, p_103] (by decide)

def p_7873 : CertifiedPrime :=
  certify 7873 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_41] (by decide)

def p_7877 : CertifiedPrime :=
  certify 7877 (2) [p_2, p_2, p_11, p_179] (by decide)

def p_7883 : CertifiedPrime :=
  certify 7883 (2) [p_2, p_7, p_563] (by decide)

def p_7901 : CertifiedPrime :=
  certify 7901 (2) [p_2, p_2, p_5, p_5, p_79] (by decide)

def p_7927 : CertifiedPrime :=
  certify 7927 (3) [p_2, p_3, p_1321] (by decide)

def p_7933 : CertifiedPrime :=
  certify 7933 (2) [p_2, p_2, p_3, p_661] (by decide)

def p_7951 : CertifiedPrime :=
  certify 7951 (6) [p_2, p_3, p_5, p_5, p_53] (by decide)

def p_7963 : CertifiedPrime :=
  certify 7963 (5) [p_2, p_3, p_1327] (by decide)

def p_7993 : CertifiedPrime :=
  certify 7993 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_37] (by decide)

def p_8011 : CertifiedPrime :=
  certify 8011 (14) [p_2, p_3, p_3, p_5, p_89] (by decide)

def p_8017 : CertifiedPrime :=
  certify 8017 (5) [p_2, p_2, p_2, p_2, p_3, p_167] (by decide)

def p_8053 : CertifiedPrime :=
  certify 8053 (2) [p_2, p_2, p_3, p_11, p_61] (by decide)

def p_8069 : CertifiedPrime :=
  certify 8069 (2) [p_2, p_2, p_2017] (by decide)

def p_8081 : CertifiedPrime :=
  certify 8081 (3) [p_2, p_2, p_2, p_2, p_5, p_101] (by decide)

def p_8087 : CertifiedPrime :=
  certify 8087 (5) [p_2, p_13, p_311] (by decide)

def p_8089 : CertifiedPrime :=
  certify 8089 (17) [p_2, p_2, p_2, p_3, p_337] (by decide)

def p_8093 : CertifiedPrime :=
  certify 8093 (2) [p_2, p_2, p_7, p_17, p_17] (by decide)

def p_8101 : CertifiedPrime :=
  certify 8101 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_8117 : CertifiedPrime :=
  certify 8117 (2) [p_2, p_2, p_2029] (by decide)

def p_8161 : CertifiedPrime :=
  certify 8161 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_17] (by decide)

def p_8191 : CertifiedPrime :=
  certify 8191 (17) [p_2, p_3, p_3, p_5, p_7, p_13] (by decide)

def p_8209 : CertifiedPrime :=
  certify 8209 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_19] (by decide)

def p_8219 : CertifiedPrime :=
  certify 8219 (2) [p_2, p_7, p_587] (by decide)

def p_8231 : CertifiedPrime :=
  certify 8231 (11) [p_2, p_5, p_823] (by decide)

def p_8237 : CertifiedPrime :=
  certify 8237 (2) [p_2, p_2, p_29, p_71] (by decide)

def p_8269 : CertifiedPrime :=
  certify 8269 (2) [p_2, p_2, p_3, p_13, p_53] (by decide)

def p_8287 : CertifiedPrime :=
  certify 8287 (3) [p_2, p_3, p_1381] (by decide)

def p_8297 : CertifiedPrime :=
  certify 8297 (3) [p_2, p_2, p_2, p_17, p_61] (by decide)

def p_8311 : CertifiedPrime :=
  certify 8311 (3) [p_2, p_3, p_5, p_277] (by decide)

def p_8317 : CertifiedPrime :=
  certify 8317 (6) [p_2, p_2, p_3, p_3, p_3, p_7, p_11] (by decide)

def p_8329 : CertifiedPrime :=
  certify 8329 (7) [p_2, p_2, p_2, p_3, p_347] (by decide)

def p_8353 : CertifiedPrime :=
  certify 8353 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_29] (by decide)

def p_8377 : CertifiedPrime :=
  certify 8377 (5) [p_2, p_2, p_2, p_3, p_349] (by decide)

def p_8387 : CertifiedPrime :=
  certify 8387 (2) [p_2, p_7, p_599] (by decide)

def p_8419 : CertifiedPrime :=
  certify 8419 (3) [p_2, p_3, p_23, p_61] (by decide)

def p_8423 : CertifiedPrime :=
  certify 8423 (5) [p_2, p_4211] (by decide)

def p_8501 : CertifiedPrime :=
  certify 8501 (7) [p_2, p_2, p_5, p_5, p_5, p_17] (by decide)

def p_8513 : CertifiedPrime :=
  certify 8513 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_19] (by decide)

def p_8543 : CertifiedPrime :=
  certify 8543 (5) [p_2, p_4271] (by decide)

def p_8563 : CertifiedPrime :=
  certify 8563 (2) [p_2, p_3, p_1427] (by decide)

def p_8573 : CertifiedPrime :=
  certify 8573 (2) [p_2, p_2, p_2143] (by decide)

def p_8581 : CertifiedPrime :=
  certify 8581 (6) [p_2, p_2, p_3, p_5, p_11, p_13] (by decide)

def p_8609 : CertifiedPrime :=
  certify 8609 (3) [p_2, p_2, p_2, p_2, p_2, p_269] (by decide)

def p_8627 : CertifiedPrime :=
  certify 8627 (2) [p_2, p_19, p_227] (by decide)

def p_8629 : CertifiedPrime :=
  certify 8629 (6) [p_2, p_2, p_3, p_719] (by decide)

def p_8647 : CertifiedPrime :=
  certify 8647 (3) [p_2, p_3, p_11, p_131] (by decide)

def p_8663 : CertifiedPrime :=
  certify 8663 (5) [p_2, p_61, p_71] (by decide)

def p_8669 : CertifiedPrime :=
  certify 8669 (2) [p_2, p_2, p_11, p_197] (by decide)

def p_8681 : CertifiedPrime :=
  certify 8681 (15) [p_2, p_2, p_2, p_5, p_7, p_31] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
