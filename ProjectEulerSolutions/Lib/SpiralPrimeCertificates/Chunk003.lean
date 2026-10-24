import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk002

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_2659 : CertifiedPrime :=
  certify 2659 (2) [p_2, p_3, p_443] (by decide)

def p_2663 : CertifiedPrime :=
  certify 2663 (5) [p_2, p_11, p_11, p_11] (by decide)

def p_2671 : CertifiedPrime :=
  certify 2671 (7) [p_2, p_3, p_5, p_89] (by decide)

def p_2683 : CertifiedPrime :=
  certify 2683 (2) [p_2, p_3, p_3, p_149] (by decide)

def p_2687 : CertifiedPrime :=
  certify 2687 (5) [p_2, p_17, p_79] (by decide)

def p_2689 : CertifiedPrime :=
  certify 2689 (19) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7] (by decide)

def p_2693 : CertifiedPrime :=
  certify 2693 (2) [p_2, p_2, p_673] (by decide)

def p_2699 : CertifiedPrime :=
  certify 2699 (2) [p_2, p_19, p_71] (by decide)

def p_2707 : CertifiedPrime :=
  certify 2707 (2) [p_2, p_3, p_11, p_41] (by decide)

def p_2711 : CertifiedPrime :=
  certify 2711 (7) [p_2, p_5, p_271] (by decide)

def p_2713 : CertifiedPrime :=
  certify 2713 (5) [p_2, p_2, p_2, p_3, p_113] (by decide)

def p_2719 : CertifiedPrime :=
  certify 2719 (3) [p_2, p_3, p_3, p_151] (by decide)

def p_2729 : CertifiedPrime :=
  certify 2729 (3) [p_2, p_2, p_2, p_11, p_31] (by decide)

def p_2731 : CertifiedPrime :=
  certify 2731 (3) [p_2, p_3, p_5, p_7, p_13] (by decide)

def p_2741 : CertifiedPrime :=
  certify 2741 (2) [p_2, p_2, p_5, p_137] (by decide)

def p_2749 : CertifiedPrime :=
  certify 2749 (6) [p_2, p_2, p_3, p_229] (by decide)

def p_2753 : CertifiedPrime :=
  certify 2753 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_43] (by decide)

def p_2767 : CertifiedPrime :=
  certify 2767 (3) [p_2, p_3, p_461] (by decide)

def p_2777 : CertifiedPrime :=
  certify 2777 (3) [p_2, p_2, p_2, p_347] (by decide)

def p_2789 : CertifiedPrime :=
  certify 2789 (2) [p_2, p_2, p_17, p_41] (by decide)

def p_2791 : CertifiedPrime :=
  certify 2791 (6) [p_2, p_3, p_3, p_5, p_31] (by decide)

def p_2797 : CertifiedPrime :=
  certify 2797 (2) [p_2, p_2, p_3, p_233] (by decide)

def p_2801 : CertifiedPrime :=
  certify 2801 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7] (by decide)

def p_2803 : CertifiedPrime :=
  certify 2803 (2) [p_2, p_3, p_467] (by decide)

def p_2819 : CertifiedPrime :=
  certify 2819 (2) [p_2, p_1409] (by decide)

def p_2833 : CertifiedPrime :=
  certify 2833 (5) [p_2, p_2, p_2, p_2, p_3, p_59] (by decide)

def p_2837 : CertifiedPrime :=
  certify 2837 (2) [p_2, p_2, p_709] (by decide)

def p_2843 : CertifiedPrime :=
  certify 2843 (2) [p_2, p_7, p_7, p_29] (by decide)

def p_2851 : CertifiedPrime :=
  certify 2851 (2) [p_2, p_3, p_5, p_5, p_19] (by decide)

def p_2857 : CertifiedPrime :=
  certify 2857 (11) [p_2, p_2, p_2, p_3, p_7, p_17] (by decide)

def p_2861 : CertifiedPrime :=
  certify 2861 (2) [p_2, p_2, p_5, p_11, p_13] (by decide)

def p_2887 : CertifiedPrime :=
  certify 2887 (5) [p_2, p_3, p_13, p_37] (by decide)

def p_2897 : CertifiedPrime :=
  certify 2897 (3) [p_2, p_2, p_2, p_2, p_181] (by decide)

def p_2903 : CertifiedPrime :=
  certify 2903 (5) [p_2, p_1451] (by decide)

def p_2909 : CertifiedPrime :=
  certify 2909 (2) [p_2, p_2, p_727] (by decide)

def p_2917 : CertifiedPrime :=
  certify 2917 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3] (by decide)

def p_2927 : CertifiedPrime :=
  certify 2927 (5) [p_2, p_7, p_11, p_19] (by decide)

def p_2939 : CertifiedPrime :=
  certify 2939 (2) [p_2, p_13, p_113] (by decide)

def p_2953 : CertifiedPrime :=
  certify 2953 (13) [p_2, p_2, p_2, p_3, p_3, p_41] (by decide)

def p_2957 : CertifiedPrime :=
  certify 2957 (2) [p_2, p_2, p_739] (by decide)

def p_2963 : CertifiedPrime :=
  certify 2963 (2) [p_2, p_1481] (by decide)

def p_2969 : CertifiedPrime :=
  certify 2969 (3) [p_2, p_2, p_2, p_7, p_53] (by decide)

def p_2971 : CertifiedPrime :=
  certify 2971 (10) [p_2, p_3, p_3, p_3, p_5, p_11] (by decide)

def p_2999 : CertifiedPrime :=
  certify 2999 (17) [p_2, p_1499] (by decide)

def p_3001 : CertifiedPrime :=
  certify 3001 (14) [p_2, p_2, p_2, p_3, p_5, p_5, p_5] (by decide)

def p_3011 : CertifiedPrime :=
  certify 3011 (2) [p_2, p_5, p_7, p_43] (by decide)

def p_3019 : CertifiedPrime :=
  certify 3019 (2) [p_2, p_3, p_503] (by decide)

def p_3023 : CertifiedPrime :=
  certify 3023 (5) [p_2, p_1511] (by decide)

def p_3037 : CertifiedPrime :=
  certify 3037 (2) [p_2, p_2, p_3, p_11, p_23] (by decide)

def p_3041 : CertifiedPrime :=
  certify 3041 (3) [p_2, p_2, p_2, p_2, p_2, p_5, p_19] (by decide)

def p_3049 : CertifiedPrime :=
  certify 3049 (11) [p_2, p_2, p_2, p_3, p_127] (by decide)

def p_3061 : CertifiedPrime :=
  certify 3061 (6) [p_2, p_2, p_3, p_3, p_5, p_17] (by decide)

def p_3067 : CertifiedPrime :=
  certify 3067 (2) [p_2, p_3, p_7, p_73] (by decide)

def p_3079 : CertifiedPrime :=
  certify 3079 (6) [p_2, p_3, p_3, p_3, p_3, p_19] (by decide)

def p_3083 : CertifiedPrime :=
  certify 3083 (2) [p_2, p_23, p_67] (by decide)

def p_3089 : CertifiedPrime :=
  certify 3089 (3) [p_2, p_2, p_2, p_2, p_193] (by decide)

def p_3109 : CertifiedPrime :=
  certify 3109 (6) [p_2, p_2, p_3, p_7, p_37] (by decide)

def p_3119 : CertifiedPrime :=
  certify 3119 (7) [p_2, p_1559] (by decide)

def p_3121 : CertifiedPrime :=
  certify 3121 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_13] (by decide)

def p_3137 : CertifiedPrime :=
  certify 3137 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7] (by decide)

def p_3163 : CertifiedPrime :=
  certify 3163 (3) [p_2, p_3, p_17, p_31] (by decide)

def p_3169 : CertifiedPrime :=
  certify 3169 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11] (by decide)

def p_3181 : CertifiedPrime :=
  certify 3181 (7) [p_2, p_2, p_3, p_5, p_53] (by decide)

def p_3187 : CertifiedPrime :=
  certify 3187 (2) [p_2, p_3, p_3, p_3, p_59] (by decide)

def p_3191 : CertifiedPrime :=
  certify 3191 (11) [p_2, p_5, p_11, p_29] (by decide)

def p_3203 : CertifiedPrime :=
  certify 3203 (2) [p_2, p_1601] (by decide)

def p_3209 : CertifiedPrime :=
  certify 3209 (3) [p_2, p_2, p_2, p_401] (by decide)

def p_3217 : CertifiedPrime :=
  certify 3217 (5) [p_2, p_2, p_2, p_2, p_3, p_67] (by decide)

def p_3221 : CertifiedPrime :=
  certify 3221 (10) [p_2, p_2, p_5, p_7, p_23] (by decide)

def p_3229 : CertifiedPrime :=
  certify 3229 (6) [p_2, p_2, p_3, p_269] (by decide)

def p_3251 : CertifiedPrime :=
  certify 3251 (6) [p_2, p_5, p_5, p_5, p_13] (by decide)

def p_3253 : CertifiedPrime :=
  certify 3253 (2) [p_2, p_2, p_3, p_271] (by decide)

def p_3257 : CertifiedPrime :=
  certify 3257 (3) [p_2, p_2, p_2, p_11, p_37] (by decide)

def p_3259 : CertifiedPrime :=
  certify 3259 (3) [p_2, p_3, p_3, p_181] (by decide)

def p_3271 : CertifiedPrime :=
  certify 3271 (3) [p_2, p_3, p_5, p_109] (by decide)

def p_3299 : CertifiedPrime :=
  certify 3299 (2) [p_2, p_17, p_97] (by decide)

def p_3301 : CertifiedPrime :=
  certify 3301 (6) [p_2, p_2, p_3, p_5, p_5, p_11] (by decide)

def p_3307 : CertifiedPrime :=
  certify 3307 (2) [p_2, p_3, p_19, p_29] (by decide)

def p_3313 : CertifiedPrime :=
  certify 3313 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_23] (by decide)

def p_3319 : CertifiedPrime :=
  certify 3319 (6) [p_2, p_3, p_7, p_79] (by decide)

def p_3323 : CertifiedPrime :=
  certify 3323 (2) [p_2, p_11, p_151] (by decide)

def p_3329 : CertifiedPrime :=
  certify 3329 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_13] (by decide)

def p_3331 : CertifiedPrime :=
  certify 3331 (3) [p_2, p_3, p_3, p_5, p_37] (by decide)

def p_3343 : CertifiedPrime :=
  certify 3343 (5) [p_2, p_3, p_557] (by decide)

def p_3359 : CertifiedPrime :=
  certify 3359 (11) [p_2, p_23, p_73] (by decide)

def p_3407 : CertifiedPrime :=
  certify 3407 (5) [p_2, p_13, p_131] (by decide)

def p_3413 : CertifiedPrime :=
  certify 3413 (2) [p_2, p_2, p_853] (by decide)

def p_3433 : CertifiedPrime :=
  certify 3433 (5) [p_2, p_2, p_2, p_3, p_11, p_13] (by decide)

def p_3449 : CertifiedPrime :=
  certify 3449 (3) [p_2, p_2, p_2, p_431] (by decide)

def p_3457 : CertifiedPrime :=
  certify 3457 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3] (by decide)

def p_3461 : CertifiedPrime :=
  certify 3461 (2) [p_2, p_2, p_5, p_173] (by decide)

def p_3463 : CertifiedPrime :=
  certify 3463 (3) [p_2, p_3, p_577] (by decide)

def p_3467 : CertifiedPrime :=
  certify 3467 (2) [p_2, p_1733] (by decide)

def p_3469 : CertifiedPrime :=
  certify 3469 (2) [p_2, p_2, p_3, p_17, p_17] (by decide)

def p_3491 : CertifiedPrime :=
  certify 3491 (2) [p_2, p_5, p_349] (by decide)

def p_3499 : CertifiedPrime :=
  certify 3499 (2) [p_2, p_3, p_11, p_53] (by decide)

def p_3511 : CertifiedPrime :=
  certify 3511 (7) [p_2, p_3, p_3, p_3, p_5, p_13] (by decide)

def p_3517 : CertifiedPrime :=
  certify 3517 (2) [p_2, p_2, p_3, p_293] (by decide)

def p_3533 : CertifiedPrime :=
  certify 3533 (2) [p_2, p_2, p_883] (by decide)

def p_3539 : CertifiedPrime :=
  certify 3539 (2) [p_2, p_29, p_61] (by decide)

def p_3541 : CertifiedPrime :=
  certify 3541 (7) [p_2, p_2, p_3, p_5, p_59] (by decide)

def p_3547 : CertifiedPrime :=
  certify 3547 (2) [p_2, p_3, p_3, p_197] (by decide)

def p_3557 : CertifiedPrime :=
  certify 3557 (2) [p_2, p_2, p_7, p_127] (by decide)

def p_3559 : CertifiedPrime :=
  certify 3559 (3) [p_2, p_3, p_593] (by decide)

def p_3571 : CertifiedPrime :=
  certify 3571 (2) [p_2, p_3, p_5, p_7, p_17] (by decide)

def p_3581 : CertifiedPrime :=
  certify 3581 (2) [p_2, p_2, p_5, p_179] (by decide)

def p_3583 : CertifiedPrime :=
  certify 3583 (3) [p_2, p_3, p_3, p_199] (by decide)

def p_3593 : CertifiedPrime :=
  certify 3593 (3) [p_2, p_2, p_2, p_449] (by decide)

def p_3607 : CertifiedPrime :=
  certify 3607 (5) [p_2, p_3, p_601] (by decide)

def p_3613 : CertifiedPrime :=
  certify 3613 (2) [p_2, p_2, p_3, p_7, p_43] (by decide)

def p_3617 : CertifiedPrime :=
  certify 3617 (3) [p_2, p_2, p_2, p_2, p_2, p_113] (by decide)

def p_3623 : CertifiedPrime :=
  certify 3623 (5) [p_2, p_1811] (by decide)

def p_3631 : CertifiedPrime :=
  certify 3631 (15) [p_2, p_3, p_5, p_11, p_11] (by decide)

def p_3637 : CertifiedPrime :=
  certify 3637 (2) [p_2, p_2, p_3, p_3, p_101] (by decide)

def p_3643 : CertifiedPrime :=
  certify 3643 (2) [p_2, p_3, p_607] (by decide)

def p_3659 : CertifiedPrime :=
  certify 3659 (2) [p_2, p_31, p_59] (by decide)

def p_3671 : CertifiedPrime :=
  certify 3671 (13) [p_2, p_5, p_367] (by decide)

def p_3673 : CertifiedPrime :=
  certify 3673 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_17] (by decide)

def p_3677 : CertifiedPrime :=
  certify 3677 (2) [p_2, p_2, p_919] (by decide)

def p_3697 : CertifiedPrime :=
  certify 3697 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_11] (by decide)

def p_3701 : CertifiedPrime :=
  certify 3701 (2) [p_2, p_2, p_5, p_5, p_37] (by decide)

def p_3709 : CertifiedPrime :=
  certify 3709 (2) [p_2, p_2, p_3, p_3, p_103] (by decide)

def p_3719 : CertifiedPrime :=
  certify 3719 (7) [p_2, p_11, p_13, p_13] (by decide)

def p_3727 : CertifiedPrime :=
  certify 3727 (3) [p_2, p_3, p_3, p_3, p_3, p_23] (by decide)

def p_3733 : CertifiedPrime :=
  certify 3733 (2) [p_2, p_2, p_3, p_311] (by decide)

def p_3761 : CertifiedPrime :=
  certify 3761 (3) [p_2, p_2, p_2, p_2, p_5, p_47] (by decide)

def p_3767 : CertifiedPrime :=
  certify 3767 (5) [p_2, p_7, p_269] (by decide)

def p_3769 : CertifiedPrime :=
  certify 3769 (7) [p_2, p_2, p_2, p_3, p_157] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
