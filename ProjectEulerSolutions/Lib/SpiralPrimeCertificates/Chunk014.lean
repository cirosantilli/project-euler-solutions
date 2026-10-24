import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk013

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_4002001 : CertifiedPrime :=
  certify 4002001 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_5, p_23, p_29] (by decide)

def p_4024037 : CertifiedPrime :=
  certify 4024037 (2) [p_2, p_2, p_17, p_17, p_59, p_59] (by decide)

def p_4042111 : CertifiedPrime :=
  certify 4042111 (3) [p_2, p_3, p_5, p_67, p_2011] (by decide)

def p_4062241 : CertifiedPrime :=
  certify 4062241 (17) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_31] (by decide)

def p_4066273 : CertifiedPrime :=
  certify 4066273 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_2017] (by decide)

def p_4090507 : CertifiedPrime :=
  certify 4090507 (3) [p_2, p_3, p_7, p_17, p_17, p_337] (by decide)

def p_4104677 : CertifiedPrime :=
  certify 4104677 (2) [p_2, p_2, p_1013, p_1013] (by decide)

def p_4135123 : CertifiedPrime :=
  certify 4135123 (5) [p_2, p_3, p_3, p_19, p_107, p_113] (by decide)

def p_4137157 : CertifiedPrime :=
  certify 4137157 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_113, p_113] (by decide)

def p_4175893 : CertifiedPrime :=
  certify 4175893 (2) [p_2, p_2, p_3, p_3, p_7, p_73, p_227] (by decide)

def p_4200451 : CertifiedPrime :=
  certify 4200451 (2) [p_2, p_3, p_5, p_5, p_41, p_683] (by decide)

def p_4202501 : CertifiedPrime :=
  certify 4202501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_41, p_41] (by decide)

def p_4218917 : CertifiedPrime :=
  certify 4218917 (2) [p_2, p_2, p_13, p_13, p_79, p_79] (by decide)

def p_4220971 : CertifiedPrime :=
  certify 4220971 (2) [p_2, p_3, p_5, p_13, p_79, p_137] (by decide)

def p_4227137 : CertifiedPrime :=
  certify 4227137 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_257, p_257] (by decide)

def p_4237423 : CertifiedPrime :=
  certify 4237423 (6) [p_2, p_3, p_7, p_7, p_7, p_29, p_71] (by decide)

def p_4260097 : CertifiedPrime :=
  certify 4260097 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_43, p_43] (by decide)

def p_4262161 : CertifiedPrime :=
  certify 4262161 (29) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_43, p_59] (by decide)

def p_4299403 : CertifiedPrime :=
  certify 4299403 (2) [p_2, p_3, p_17, p_61, p_691] (by decide)

def p_4301477 : CertifiedPrime :=
  certify 4301477 (3) [p_2, p_2, p_17, p_17, p_61, p_61] (by decide)

def p_4307701 : CertifiedPrime :=
  certify 4307701 (6) [p_2, p_2, p_3, p_5, p_5, p_83, p_173] (by decide)

def p_4324321 : CertifiedPrime :=
  certify 4324321 (17) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_11, p_13] (by decide)

def p_4326401 : CertifiedPrime :=
  certify 4326401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13] (by decide)

def p_4343057 : CertifiedPrime :=
  certify 4343057 (3) [p_2, p_2, p_2, p_2, p_521, p_521] (by decide)

def p_4349311 : CertifiedPrime :=
  certify 4349311 (7) [p_2, p_3, p_5, p_7, p_139, p_149] (by decide)

def p_4351397 : CertifiedPrime :=
  certify 4351397 (2) [p_2, p_2, p_7, p_7, p_149, p_149] (by decide)

def p_4374373 : CertifiedPrime :=
  certify 4374373 (2) [p_2, p_2, p_3, p_17, p_41, p_523] (by decide)

def p_4384837 : CertifiedPrime :=
  certify 4384837 (5) [p_2, p_2, p_3, p_3, p_349, p_349] (by decide)

def p_4393217 : CertifiedPrime :=
  certify 4393217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_131, p_131] (by decide)

def p_4433131 : CertifiedPrime :=
  certify 4433131 (10) [p_2, p_3, p_3, p_3, p_3, p_5, p_13, p_421] (by decide)

def p_4435237 : CertifiedPrime :=
  certify 4435237 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_13, p_13] (by decide)

def p_4462657 : CertifiedPrime :=
  certify 4462657 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_11, p_2113] (by decide)

def p_4477457 : CertifiedPrime :=
  certify 4477457 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_23, p_23] (by decide)

def p_4483807 : CertifiedPrime :=
  certify 4483807 (3) [p_2, p_3, p_29, p_73, p_353] (by decide)

def p_4494401 : CertifiedPrime :=
  certify 4494401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_53, p_53] (by decide)

def p_4519877 : CertifiedPrime :=
  certify 4519877 (2) [p_2, p_2, p_1063, p_1063] (by decide)

def p_4534771 : CertifiedPrime :=
  certify 4534771 (13) [p_2, p_3, p_5, p_71, p_2129] (by decide)

def p_4562497 : CertifiedPrime :=
  certify 4562497 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_89, p_89] (by decide)

def p_4564633 : CertifiedPrime :=
  certify 4564633 (5) [p_2, p_2, p_2, p_3, p_89, p_2137] (by decide)

def p_4573183 : CertifiedPrime :=
  certify 4573183 (3) [p_2, p_3, p_23, p_31, p_1069] (by decide)

def p_4590307 : CertifiedPrime :=
  certify 4590307 (5) [p_2, p_3, p_3, p_7, p_17, p_2143] (by decide)

def p_4603171 : CertifiedPrime :=
  certify 4603171 (2) [p_2, p_3, p_5, p_11, p_13, p_29, p_37] (by decide)

def p_4639717 : CertifiedPrime :=
  certify 4639717 (19) [p_2, p_2, p_3, p_3, p_359, p_359] (by decide)

def p_4654807 : CertifiedPrime :=
  certify 4654807 (3) [p_2, p_3, p_13, p_83, p_719] (by decide)

def p_4663441 : CertifiedPrime :=
  certify 4663441 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_17, p_127] (by decide)

def p_4676407 : CertifiedPrime :=
  certify 4676407 (3) [p_2, p_3, p_7, p_23, p_47, p_103] (by decide)

def p_4680733 : CertifiedPrime :=
  certify 4680733 (2) [p_2, p_2, p_3, p_7, p_103, p_541] (by decide)

def p_4702393 : CertifiedPrime :=
  certify 4702393 (5) [p_2, p_2, p_2, p_3, p_3, p_241, p_271] (by decide)

def p_4715413 : CertifiedPrime :=
  certify 4715413 (6) [p_2, p_2, p_3, p_13, p_167, p_181] (by decide)

def p_4726277 : CertifiedPrime :=
  certify 4726277 (2) [p_2, p_2, p_1087, p_1087] (by decide)

def p_4767673 : CertifiedPrime :=
  certify 4767673 (5) [p_2, p_2, p_2, p_3, p_7, p_13, p_37, p_59] (by decide)

def p_4846603 : CertifiedPrime :=
  certify 4846603 (3) [p_2, p_3, p_31, p_71, p_367] (by decide)

def p_4859821 : CertifiedPrime :=
  certify 4859821 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_7, p_19, p_29] (by decide)

def p_4864231 : CertifiedPrime :=
  certify 4864231 (6) [p_2, p_3, p_3, p_5, p_7, p_7, p_1103] (by decide)

def p_4884101 : CertifiedPrime :=
  certify 4884101 (3) [p_2, p_2, p_5, p_5, p_13, p_13, p_17, p_17] (by decide)

def p_4946177 : CertifiedPrime :=
  certify 4946177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_139, p_139] (by decide)

def p_5010883 : CertifiedPrime :=
  certify 5010883 (3) [p_2, p_3, p_373, p_2239] (by decide)

def p_5024323 : CertifiedPrime :=
  certify 5024323 (2) [p_2, p_3, p_3, p_3, p_19, p_59, p_83] (by decide)

def p_5051257 : CertifiedPrime :=
  certify 5051257 (5) [p_2, p_2, p_2, p_3, p_7, p_107, p_281] (by decide)

def p_5087281 : CertifiedPrime :=
  certify 5087281 (17) [p_2, p_2, p_2, p_2, p_3, p_5, p_11, p_41, p_47] (by decide)

def p_5105341 : CertifiedPrime :=
  certify 5105341 (2) [p_2, p_2, p_3, p_3, p_5, p_113, p_251] (by decide)

def p_5107601 : CertifiedPrime :=
  certify 5107601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_113, p_113] (by decide)

def p_5118907 : CertifiedPrime :=
  certify 5118907 (5) [p_2, p_3, p_13, p_29, p_31, p_73] (by decide)

def p_5127961 : CertifiedPrime :=
  certify 5127961 (7) [p_2, p_2, p_2, p_3, p_5, p_151, p_283] (by decide)

def p_5134757 : CertifiedPrime :=
  certify 5134757 (2) [p_2, p_2, p_11, p_11, p_103, p_103] (by decide)

def p_5168803 : CertifiedPrime :=
  certify 5168803 (3) [p_2, p_3, p_379, p_2273] (by decide)

def p_5182453 : CertifiedPrime :=
  certify 5182453 (2) [p_2, p_2, p_3, p_3, p_11, p_23, p_569] (by decide)

def p_5200681 : CertifiedPrime :=
  certify 5200681 (17) [p_2, p_2, p_2, p_3, p_5, p_19, p_2281] (by decide)

def p_5225797 : CertifiedPrime :=
  certify 5225797 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_127, p_127] (by decide)

def p_5262437 : CertifiedPrime :=
  certify 5262437 (2) [p_2, p_2, p_31, p_31, p_37, p_37] (by decide)

def p_5296903 : CertifiedPrime :=
  certify 5296903 (3) [p_2, p_3, p_13, p_59, p_1151] (by decide)

def p_5306113 : CertifiedPrime :=
  certify 5306113 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_47] (by decide)

def p_5308417 : CertifiedPrime :=
  certify 5308417 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3] (by decide)

def p_5333791 : CertifiedPrime :=
  certify 5333791 (12) [p_2, p_3, p_5, p_7, p_11, p_2309] (by decide)

def p_5336101 : CertifiedPrime :=
  certify 5336101 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_11, p_11] (by decide)

def p_5352283 : CertifiedPrime :=
  certify 5352283 (2) [p_2, p_3, p_3, p_13, p_89, p_257] (by decide)

def p_5354597 : CertifiedPrime :=
  certify 5354597 (2) [p_2, p_2, p_13, p_13, p_89, p_89] (by decide)

def p_5375443 : CertifiedPrime :=
  certify 5375443 (2) [p_2, p_3, p_19, p_61, p_773] (by decide)

def p_5382401 : CertifiedPrime :=
  certify 5382401 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_29, p_29] (by decide)

def p_5403301 : CertifiedPrime :=
  certify 5403301 (14) [p_2, p_2, p_3, p_5, p_5, p_7, p_31, p_83] (by decide)

def p_5410277 : CertifiedPrime :=
  certify 5410277 (2) [p_2, p_2, p_1163, p_1163] (by decide)

def p_5428901 : CertifiedPrime :=
  certify 5428901 (2) [p_2, p_2, p_5, p_5, p_233, p_233] (by decide)

def p_5456897 : CertifiedPrime :=
  certify 5456897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_73, p_73] (by decide)

def p_5473261 : CertifiedPrime :=
  certify 5473261 (11) [p_2, p_2, p_3, p_3, p_5, p_13, p_2339] (by decide)

def p_5501371 : CertifiedPrime :=
  certify 5501371 (3) [p_2, p_3, p_5, p_7, p_17, p_23, p_67] (by decide)

def p_5541317 : CertifiedPrime :=
  certify 5541317 (2) [p_2, p_2, p_11, p_11, p_107, p_107] (by decide)

def p_5569601 : CertifiedPrime :=
  certify 5569601 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_59, p_59] (by decide)

def p_5571961 : CertifiedPrime :=
  certify 5571961 (11) [p_2, p_2, p_2, p_3, p_5, p_59, p_787] (by decide)

def p_5590861 : CertifiedPrime :=
  certify 5590861 (10) [p_2, p_2, p_3, p_5, p_11, p_43, p_197] (by decide)

def p_5628757 : CertifiedPrime :=
  certify 5628757 (2) [p_2, p_2, p_3, p_7, p_113, p_593] (by decide)

def p_5633503 : CertifiedPrime :=
  certify 5633503 (3) [p_2, p_3, p_7, p_113, p_1187] (by decide)

def p_5664401 : CertifiedPrime :=
  certify 5664401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_17, p_17] (by decide)

def p_5690611 : CertifiedPrime :=
  certify 5690611 (2) [p_2, p_3, p_3, p_5, p_53, p_1193] (by decide)

def p_5700157 : CertifiedPrime :=
  certify 5700157 (6) [p_2, p_2, p_3, p_7, p_11, p_31, p_199] (by decide)

def p_5714491 : CertifiedPrime :=
  certify 5714491 (2) [p_2, p_3, p_5, p_239, p_797] (by decide)

def p_5748007 : CertifiedPrime :=
  certify 5748007 (3) [p_2, p_3, p_11, p_17, p_47, p_109] (by decide)

def p_5772007 : CertifiedPrime :=
  certify 5772007 (5) [p_2, p_3, p_3, p_3, p_89, p_1201] (by decide)

def p_5779217 : CertifiedPrime :=
  certify 5779217 (3) [p_2, p_2, p_2, p_2, p_601, p_601] (by decide)

def p_5788837 : CertifiedPrime :=
  certify 5788837 (2) [p_2, p_2, p_3, p_3, p_401, p_401] (by decide)

def p_5805691 : CertifiedPrime :=
  certify 5805691 (2) [p_2, p_3, p_5, p_11, p_73, p_241] (by decide)

def p_5815333 : CertifiedPrime :=
  certify 5815333 (7) [p_2, p_2, p_3, p_3, p_67, p_2411] (by decide)

def p_5856401 : CertifiedPrime :=
  certify 5856401 (7) [p_2, p_2, p_2, p_2, p_5, p_5, p_11, p_11, p_11, p_11] (by decide)

def p_5902471 : CertifiedPrime :=
  certify 5902471 (7) [p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_7, p_347] (by decide)

def p_5904901 : CertifiedPrime :=
  certify 5904901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_5907331 : CertifiedPrime :=
  certify 5907331 (11) [p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_11, p_13, p_17] (by decide)

def p_5931661 : CertifiedPrime :=
  certify 5931661 (13) [p_2, p_2, p_3, p_5, p_7, p_29, p_487] (by decide)

def p_5965807 : CertifiedPrime :=
  certify 5965807 (6) [p_2, p_3, p_7, p_11, p_37, p_349] (by decide)

def p_6004951 : CertifiedPrime :=
  certify 6004951 (7) [p_2, p_3, p_5, p_5, p_7, p_7, p_19, p_43] (by decide)

def p_6009853 : CertifiedPrime :=
  certify 6009853 (2) [p_2, p_2, p_3, p_19, p_43, p_613] (by decide)

def p_6022117 : CertifiedPrime :=
  certify 6022117 (2) [p_2, p_2, p_3, p_3, p_409, p_409] (by decide)

def p_6031937 : CertifiedPrime :=
  certify 6031937 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_307, p_307] (by decide)

def p_6034393 : CertifiedPrime :=
  certify 6034393 (10) [p_2, p_2, p_2, p_3, p_3, p_3, p_7, p_13, p_307] (by decide)

def p_6051601 : CertifiedPrime :=
  certify 6051601 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_41, p_41] (by decide)

def p_6068833 : CertifiedPrime :=
  certify 6068833 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_821] (by decide)

def p_6071297 : CertifiedPrime :=
  certify 6071297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_11, p_11] (by decide)

def p_6078691 : CertifiedPrime :=
  certify 6078691 (13) [p_2, p_3, p_3, p_5, p_17, p_29, p_137] (by decide)

def p_6100901 : CertifiedPrime :=
  certify 6100901 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_19, p_19] (by decide)

def p_6108313 : CertifiedPrime :=
  certify 6108313 (11) [p_2, p_2, p_2, p_3, p_7, p_103, p_353] (by decide)

def p_6142963 : CertifiedPrime :=
  certify 6142963 (3) [p_2, p_3, p_7, p_37, p_59, p_67] (by decide)

def p_6167773 : CertifiedPrime :=
  certify 6167773 (5) [p_2, p_2, p_3, p_3, p_3, p_13, p_23, p_191] (by decide)

def p_6172741 : CertifiedPrime :=
  certify 6172741 (10) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_23, p_71] (by decide)

def p_6217543 : CertifiedPrime :=
  certify 6217543 (6) [p_2, p_3, p_3, p_29, p_43, p_277] (by decide)

def p_6227521 : CertifiedPrime :=
  certify 6227521 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_13, p_499] (by decide)

def p_6230017 : CertifiedPrime :=
  certify 6230017 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13] (by decide)

def p_6242503 : CertifiedPrime :=
  certify 6242503 (6) [p_2, p_3, p_7, p_7, p_17, p_1249] (by decide)

def p_6307633 : CertifiedPrime :=
  certify 6307633 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_31, p_157] (by decide)

def p_6317683 : CertifiedPrime :=
  certify 6317683 (3) [p_2, p_3, p_7, p_359, p_419] (by decide)

def p_6330257 : CertifiedPrime :=
  certify 6330257 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_37, p_37] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
