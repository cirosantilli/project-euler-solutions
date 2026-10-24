import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk003

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_3779 : CertifiedPrime :=
  certify 3779 (2) [p_2, p_1889] (by decide)

def p_3793 : CertifiedPrime :=
  certify 3793 (5) [p_2, p_2, p_2, p_2, p_3, p_79] (by decide)

def p_3803 : CertifiedPrime :=
  certify 3803 (2) [p_2, p_1901] (by decide)

def p_3821 : CertifiedPrime :=
  certify 3821 (3) [p_2, p_2, p_5, p_191] (by decide)

def p_3823 : CertifiedPrime :=
  certify 3823 (3) [p_2, p_3, p_7, p_7, p_13] (by decide)

def p_3833 : CertifiedPrime :=
  certify 3833 (3) [p_2, p_2, p_2, p_479] (by decide)

def p_3847 : CertifiedPrime :=
  certify 3847 (5) [p_2, p_3, p_641] (by decide)

def p_3851 : CertifiedPrime :=
  certify 3851 (2) [p_2, p_5, p_5, p_7, p_11] (by decide)

def p_3853 : CertifiedPrime :=
  certify 3853 (2) [p_2, p_2, p_3, p_3, p_107] (by decide)

def p_3863 : CertifiedPrime :=
  certify 3863 (5) [p_2, p_1931] (by decide)

def p_3877 : CertifiedPrime :=
  certify 3877 (2) [p_2, p_2, p_3, p_17, p_19] (by decide)

def p_3881 : CertifiedPrime :=
  certify 3881 (13) [p_2, p_2, p_2, p_5, p_97] (by decide)

def p_3889 : CertifiedPrime :=
  certify 3889 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3] (by decide)

def p_3907 : CertifiedPrime :=
  certify 3907 (2) [p_2, p_3, p_3, p_7, p_31] (by decide)

def p_3911 : CertifiedPrime :=
  certify 3911 (13) [p_2, p_5, p_17, p_23] (by decide)

def p_3917 : CertifiedPrime :=
  certify 3917 (2) [p_2, p_2, p_11, p_89] (by decide)

def p_3919 : CertifiedPrime :=
  certify 3919 (3) [p_2, p_3, p_653] (by decide)

def p_3923 : CertifiedPrime :=
  certify 3923 (2) [p_2, p_37, p_53] (by decide)

def p_3929 : CertifiedPrime :=
  certify 3929 (3) [p_2, p_2, p_2, p_491] (by decide)

def p_3931 : CertifiedPrime :=
  certify 3931 (2) [p_2, p_3, p_5, p_131] (by decide)

def p_3943 : CertifiedPrime :=
  certify 3943 (3) [p_2, p_3, p_3, p_3, p_73] (by decide)

def p_3967 : CertifiedPrime :=
  certify 3967 (6) [p_2, p_3, p_661] (by decide)

def p_3989 : CertifiedPrime :=
  certify 3989 (2) [p_2, p_2, p_997] (by decide)

def p_4001 : CertifiedPrime :=
  certify 4001 (3) [p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5] (by decide)

def p_4003 : CertifiedPrime :=
  certify 4003 (2) [p_2, p_3, p_23, p_29] (by decide)

def p_4007 : CertifiedPrime :=
  certify 4007 (5) [p_2, p_2003] (by decide)

def p_4013 : CertifiedPrime :=
  certify 4013 (2) [p_2, p_2, p_17, p_59] (by decide)

def p_4019 : CertifiedPrime :=
  certify 4019 (2) [p_2, p_7, p_7, p_41] (by decide)

def p_4021 : CertifiedPrime :=
  certify 4021 (2) [p_2, p_2, p_3, p_5, p_67] (by decide)

def p_4027 : CertifiedPrime :=
  certify 4027 (3) [p_2, p_3, p_11, p_61] (by decide)

def p_4049 : CertifiedPrime :=
  certify 4049 (3) [p_2, p_2, p_2, p_2, p_11, p_23] (by decide)

def p_4051 : CertifiedPrime :=
  certify 4051 (10) [p_2, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_4057 : CertifiedPrime :=
  certify 4057 (5) [p_2, p_2, p_2, p_3, p_13, p_13] (by decide)

def p_4073 : CertifiedPrime :=
  certify 4073 (3) [p_2, p_2, p_2, p_509] (by decide)

def p_4079 : CertifiedPrime :=
  certify 4079 (11) [p_2, p_2039] (by decide)

def p_4091 : CertifiedPrime :=
  certify 4091 (2) [p_2, p_5, p_409] (by decide)

def p_4093 : CertifiedPrime :=
  certify 4093 (2) [p_2, p_2, p_3, p_11, p_31] (by decide)

def p_4099 : CertifiedPrime :=
  certify 4099 (2) [p_2, p_3, p_683] (by decide)

def p_4111 : CertifiedPrime :=
  certify 4111 (12) [p_2, p_3, p_5, p_137] (by decide)

def p_4127 : CertifiedPrime :=
  certify 4127 (5) [p_2, p_2063] (by decide)

def p_4129 : CertifiedPrime :=
  certify 4129 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_43] (by decide)

def p_4133 : CertifiedPrime :=
  certify 4133 (2) [p_2, p_2, p_1033] (by decide)

def p_4153 : CertifiedPrime :=
  certify 4153 (5) [p_2, p_2, p_2, p_3, p_173] (by decide)

def p_4157 : CertifiedPrime :=
  certify 4157 (2) [p_2, p_2, p_1039] (by decide)

def p_4159 : CertifiedPrime :=
  certify 4159 (3) [p_2, p_3, p_3, p_3, p_7, p_11] (by decide)

def p_4177 : CertifiedPrime :=
  certify 4177 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_29] (by decide)

def p_4211 : CertifiedPrime :=
  certify 4211 (6) [p_2, p_5, p_421] (by decide)

def p_4217 : CertifiedPrime :=
  certify 4217 (3) [p_2, p_2, p_2, p_17, p_31] (by decide)

def p_4219 : CertifiedPrime :=
  certify 4219 (2) [p_2, p_3, p_19, p_37] (by decide)

def p_4229 : CertifiedPrime :=
  certify 4229 (2) [p_2, p_2, p_7, p_151] (by decide)

def p_4231 : CertifiedPrime :=
  certify 4231 (3) [p_2, p_3, p_3, p_5, p_47] (by decide)

def p_4241 : CertifiedPrime :=
  certify 4241 (3) [p_2, p_2, p_2, p_2, p_5, p_53] (by decide)

def p_4259 : CertifiedPrime :=
  certify 4259 (2) [p_2, p_2129] (by decide)

def p_4261 : CertifiedPrime :=
  certify 4261 (2) [p_2, p_2, p_3, p_5, p_71] (by decide)

def p_4271 : CertifiedPrime :=
  certify 4271 (7) [p_2, p_5, p_7, p_61] (by decide)

def p_4273 : CertifiedPrime :=
  certify 4273 (5) [p_2, p_2, p_2, p_2, p_3, p_89] (by decide)

def p_4283 : CertifiedPrime :=
  certify 4283 (2) [p_2, p_2141] (by decide)

def p_4289 : CertifiedPrime :=
  certify 4289 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_67] (by decide)

def p_4327 : CertifiedPrime :=
  certify 4327 (3) [p_2, p_3, p_7, p_103] (by decide)

def p_4337 : CertifiedPrime :=
  certify 4337 (3) [p_2, p_2, p_2, p_2, p_271] (by decide)

def p_4339 : CertifiedPrime :=
  certify 4339 (10) [p_2, p_3, p_3, p_241] (by decide)

def p_4349 : CertifiedPrime :=
  certify 4349 (2) [p_2, p_2, p_1087] (by decide)

def p_4357 : CertifiedPrime :=
  certify 4357 (2) [p_2, p_2, p_3, p_3, p_11, p_11] (by decide)

def p_4363 : CertifiedPrime :=
  certify 4363 (2) [p_2, p_3, p_727] (by decide)

def p_4391 : CertifiedPrime :=
  certify 4391 (14) [p_2, p_5, p_439] (by decide)

def p_4409 : CertifiedPrime :=
  certify 4409 (3) [p_2, p_2, p_2, p_19, p_29] (by decide)

def p_4423 : CertifiedPrime :=
  certify 4423 (3) [p_2, p_3, p_11, p_67] (by decide)

def p_4441 : CertifiedPrime :=
  certify 4441 (21) [p_2, p_2, p_2, p_3, p_5, p_37] (by decide)

def p_4447 : CertifiedPrime :=
  certify 4447 (3) [p_2, p_3, p_3, p_13, p_19] (by decide)

def p_4451 : CertifiedPrime :=
  certify 4451 (2) [p_2, p_5, p_5, p_89] (by decide)

def p_4457 : CertifiedPrime :=
  certify 4457 (3) [p_2, p_2, p_2, p_557] (by decide)

def p_4463 : CertifiedPrime :=
  certify 4463 (5) [p_2, p_23, p_97] (by decide)

def p_4481 : CertifiedPrime :=
  certify 4481 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_7] (by decide)

def p_4507 : CertifiedPrime :=
  certify 4507 (2) [p_2, p_3, p_751] (by decide)

def p_4517 : CertifiedPrime :=
  certify 4517 (2) [p_2, p_2, p_1129] (by decide)

def p_4519 : CertifiedPrime :=
  certify 4519 (3) [p_2, p_3, p_3, p_251] (by decide)

def p_4523 : CertifiedPrime :=
  certify 4523 (5) [p_2, p_7, p_17, p_19] (by decide)

def p_4547 : CertifiedPrime :=
  certify 4547 (2) [p_2, p_2273] (by decide)

def p_4549 : CertifiedPrime :=
  certify 4549 (6) [p_2, p_2, p_3, p_379] (by decide)

def p_4561 : CertifiedPrime :=
  certify 4561 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_19] (by decide)

def p_4567 : CertifiedPrime :=
  certify 4567 (3) [p_2, p_3, p_761] (by decide)

def p_4583 : CertifiedPrime :=
  certify 4583 (5) [p_2, p_29, p_79] (by decide)

def p_4637 : CertifiedPrime :=
  certify 4637 (2) [p_2, p_2, p_19, p_61] (by decide)

def p_4639 : CertifiedPrime :=
  certify 4639 (3) [p_2, p_3, p_773] (by decide)

def p_4643 : CertifiedPrime :=
  certify 4643 (5) [p_2, p_11, p_211] (by decide)

def p_4649 : CertifiedPrime :=
  certify 4649 (3) [p_2, p_2, p_2, p_7, p_83] (by decide)

def p_4657 : CertifiedPrime :=
  certify 4657 (15) [p_2, p_2, p_2, p_2, p_3, p_97] (by decide)

def p_4673 : CertifiedPrime :=
  certify 4673 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_73] (by decide)

def p_4679 : CertifiedPrime :=
  certify 4679 (11) [p_2, p_2339] (by decide)

def p_4703 : CertifiedPrime :=
  certify 4703 (5) [p_2, p_2351] (by decide)

def p_4721 : CertifiedPrime :=
  certify 4721 (6) [p_2, p_2, p_2, p_2, p_5, p_59] (by decide)

def p_4723 : CertifiedPrime :=
  certify 4723 (2) [p_2, p_3, p_787] (by decide)

def p_4729 : CertifiedPrime :=
  certify 4729 (17) [p_2, p_2, p_2, p_3, p_197] (by decide)

def p_4733 : CertifiedPrime :=
  certify 4733 (5) [p_2, p_2, p_7, p_13, p_13] (by decide)

def p_4751 : CertifiedPrime :=
  certify 4751 (19) [p_2, p_5, p_5, p_5, p_19] (by decide)

def p_4787 : CertifiedPrime :=
  certify 4787 (2) [p_2, p_2393] (by decide)

def p_4789 : CertifiedPrime :=
  certify 4789 (2) [p_2, p_2, p_3, p_3, p_7, p_19] (by decide)

def p_4793 : CertifiedPrime :=
  certify 4793 (3) [p_2, p_2, p_2, p_599] (by decide)

def p_4799 : CertifiedPrime :=
  certify 4799 (7) [p_2, p_2399] (by decide)

def p_4813 : CertifiedPrime :=
  certify 4813 (2) [p_2, p_2, p_3, p_401] (by decide)

def p_4831 : CertifiedPrime :=
  certify 4831 (3) [p_2, p_3, p_5, p_7, p_23] (by decide)

def p_4861 : CertifiedPrime :=
  certify 4861 (11) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_5] (by decide)

def p_4871 : CertifiedPrime :=
  certify 4871 (11) [p_2, p_5, p_487] (by decide)

def p_4877 : CertifiedPrime :=
  certify 4877 (2) [p_2, p_2, p_23, p_53] (by decide)

def p_4889 : CertifiedPrime :=
  certify 4889 (3) [p_2, p_2, p_2, p_13, p_47] (by decide)

def p_4903 : CertifiedPrime :=
  certify 4903 (3) [p_2, p_3, p_19, p_43] (by decide)

def p_4909 : CertifiedPrime :=
  certify 4909 (6) [p_2, p_2, p_3, p_409] (by decide)

def p_4919 : CertifiedPrime :=
  certify 4919 (13) [p_2, p_2459] (by decide)

def p_4931 : CertifiedPrime :=
  certify 4931 (6) [p_2, p_5, p_17, p_29] (by decide)

def p_4933 : CertifiedPrime :=
  certify 4933 (2) [p_2, p_2, p_3, p_3, p_137] (by decide)

def p_4937 : CertifiedPrime :=
  certify 4937 (3) [p_2, p_2, p_2, p_617] (by decide)

def p_4943 : CertifiedPrime :=
  certify 4943 (7) [p_2, p_7, p_353] (by decide)

def p_4951 : CertifiedPrime :=
  certify 4951 (6) [p_2, p_3, p_3, p_5, p_5, p_11] (by decide)

def p_4957 : CertifiedPrime :=
  certify 4957 (2) [p_2, p_2, p_3, p_7, p_59] (by decide)

def p_4973 : CertifiedPrime :=
  certify 4973 (2) [p_2, p_2, p_11, p_113] (by decide)

def p_4993 : CertifiedPrime :=
  certify 4993 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_13] (by decide)

def p_4999 : CertifiedPrime :=
  certify 4999 (3) [p_2, p_3, p_7, p_7, p_17] (by decide)

def p_5003 : CertifiedPrime :=
  certify 5003 (2) [p_2, p_41, p_61] (by decide)

def p_5009 : CertifiedPrime :=
  certify 5009 (3) [p_2, p_2, p_2, p_2, p_313] (by decide)

def p_5011 : CertifiedPrime :=
  certify 5011 (2) [p_2, p_3, p_5, p_167] (by decide)

def p_5021 : CertifiedPrime :=
  certify 5021 (3) [p_2, p_2, p_5, p_251] (by decide)

def p_5039 : CertifiedPrime :=
  certify 5039 (11) [p_2, p_11, p_229] (by decide)

def p_5051 : CertifiedPrime :=
  certify 5051 (2) [p_2, p_5, p_5, p_101] (by decide)

def p_5077 : CertifiedPrime :=
  certify 5077 (2) [p_2, p_2, p_3, p_3, p_3, p_47] (by decide)

def p_5081 : CertifiedPrime :=
  certify 5081 (3) [p_2, p_2, p_2, p_5, p_127] (by decide)

def p_5087 : CertifiedPrime :=
  certify 5087 (5) [p_2, p_2543] (by decide)

def p_5099 : CertifiedPrime :=
  certify 5099 (2) [p_2, p_2549] (by decide)

def p_5101 : CertifiedPrime :=
  certify 5101 (6) [p_2, p_2, p_3, p_5, p_5, p_17] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
