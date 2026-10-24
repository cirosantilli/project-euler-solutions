import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk044

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_487371853 : CertifiedPrime :=
  certify 487371853 (2) [p_2, p_2, p_3, p_3, p_11, p_223, p_5519] (by decide)

def p_488078557 : CertifiedPrime :=
  certify 488078557 (5) [p_2, p_2, p_3, p_7, p_263, p_22093] (by decide)

def p_488741557 : CertifiedPrime :=
  certify 488741557 (2) [p_2, p_2, p_3, p_5527, p_7369] (by decide)

def p_488962657 : CertifiedPrime :=
  certify 488962657 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_7, p_13, p_691] (by decide)

def p_489117457 : CertifiedPrime :=
  certify 489117457 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_19, p_19, p_97, p_97] (by decide)

def p_489537751 : CertifiedPrime :=
  certify 489537751 (6) [p_2, p_3, p_5, p_5, p_5, p_13, p_23, p_37, p_59] (by decide)

def p_489936091 : CertifiedPrime :=
  certify 489936091 (12) [p_2, p_3, p_5, p_7, p_17, p_19, p_31, p_233] (by decide)

def p_490002497 : CertifiedPrime :=
  certify 490002497 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2767, p_2767] (by decide)

def p_490068907 : CertifiedPrime :=
  certify 490068907 (2) [p_2, p_3, p_47, p_157, p_11069] (by decide)

def p_490622501 : CertifiedPrime :=
  certify 490622501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_443, p_443] (by decide)

def p_490821871 : CertifiedPrime :=
  certify 490821871 (3) [p_2, p_3, p_5, p_7, p_11, p_19, p_53, p_211] (by decide)

def p_490888337 : CertifiedPrime :=
  certify 490888337 (3) [p_2, p_2, p_2, p_2, p_29, p_29, p_191, p_191] (by decide)

def p_491331557 : CertifiedPrime :=
  certify 491331557 (2) [p_2, p_2, p_11083, p_11083] (by decide)

def p_491398057 : CertifiedPrime :=
  certify 491398057 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_17, p_163, p_821] (by decide)

def p_491686277 : CertifiedPrime :=
  certify 491686277 (2) [p_2, p_2, p_11087, p_11087] (by decide)

def p_491930221 : CertifiedPrime :=
  certify 491930221 (6) [p_2, p_2, p_3, p_5, p_1109, p_7393] (by decide)

def p_491952401 : CertifiedPrime :=
  certify 491952401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1109, p_1109] (by decide)

def p_492063307 : CertifiedPrime :=
  certify 492063307 (3) [p_2, p_3, p_7, p_3169, p_3697] (by decide)

def p_492684613 : CertifiedPrime :=
  certify 492684613 (2) [p_2, p_2, p_3, p_7, p_7, p_31, p_151, p_179] (by decide)

def p_493017617 : CertifiedPrime :=
  certify 493017617 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_13, p_13, p_61, p_61] (by decide)

def p_493261891 : CertifiedPrime :=
  certify 493261891 (2) [p_2, p_3, p_5, p_11, p_673, p_2221] (by decide)

def p_493461797 : CertifiedPrime :=
  certify 493461797 (2) [p_2, p_2, p_29, p_29, p_383, p_383] (by decide)

def p_493883953 : CertifiedPrime :=
  certify 493883953 (5) [p_2, p_2, p_2, p_2, p_3, p_71, p_313, p_463] (by decide)

def p_494195131 : CertifiedPrime :=
  certify 494195131 (11) [p_2, p_3, p_3, p_5, p_11, p_13, p_19, p_43, p_47] (by decide)

def p_494350757 : CertifiedPrime :=
  certify 494350757 (2) [p_2, p_2, p_11117, p_11117] (by decide)

def p_494728807 : CertifiedPrime :=
  certify 494728807 (3) [p_2, p_3, p_11, p_13, p_29, p_59, p_337] (by decide)

def p_495062501 : CertifiedPrime :=
  certify 495062501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_89, p_89] (by decide)

def p_495218263 : CertifiedPrime :=
  certify 495218263 (3) [p_2, p_3, p_7, p_11, p_17, p_17, p_3709] (by decide)

def p_495329537 : CertifiedPrime :=
  certify 495329537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_107, p_107] (by decide)

def p_495886093 : CertifiedPrime :=
  certify 495886093 (2) [p_2, p_2, p_3, p_13, p_19, p_293, p_571] (by decide)

def p_495952901 : CertifiedPrime :=
  certify 495952901 (3) [p_2, p_2, p_5, p_5, p_17, p_17, p_131, p_131] (by decide)

def p_496287007 : CertifiedPrime :=
  certify 496287007 (3) [p_2, p_3, p_47, p_79, p_22277] (by decide)

def p_496732657 : CertifiedPrime :=
  certify 496732657 (10) [p_2, p_2, p_2, p_2, p_3, p_7, p_17, p_19, p_23, p_199] (by decide)

def p_497089321 : CertifiedPrime :=
  certify 497089321 (29) [p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_7, p_13, p_929] (by decide)

def p_497111617 : CertifiedPrime :=
  certify 497111617 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_929, p_929] (by decide)

def p_497535331 : CertifiedPrime :=
  certify 497535331 (2) [p_2, p_3, p_5, p_19, p_587, p_1487] (by decide)

def p_497669173 : CertifiedPrime :=
  certify 497669173 (6) [p_2, p_2, p_3, p_7, p_11, p_13, p_13, p_3187] (by decide)

def p_497736101 : CertifiedPrime :=
  certify 497736101 (2) [p_2, p_2, p_5, p_5, p_23, p_23, p_97, p_97] (by decide)

def p_497914597 : CertifiedPrime :=
  certify 497914597 (2) [p_2, p_2, p_3, p_3, p_3719, p_3719] (by decide)

def p_498294007 : CertifiedPrime :=
  certify 498294007 (3) [p_2, p_3, p_7, p_1063, p_11161] (by decide)

def p_498561913 : CertifiedPrime :=
  certify 498561913 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_827, p_2791] (by decide)

def p_498695893 : CertifiedPrime :=
  certify 498695893 (6) [p_2, p_2, p_3, p_137, p_163, p_1861] (by decide)

def p_498874561 : CertifiedPrime :=
  certify 498874561 (17) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_349, p_1489] (by decide)

def p_499075601 : CertifiedPrime :=
  certify 499075601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1117, p_1117] (by decide)

def p_499254337 : CertifiedPrime :=
  certify 499254337 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_7, p_7, p_19, p_19] (by decide)

def p_499343717 : CertifiedPrime :=
  certify 499343717 (2) [p_2, p_2, p_11173, p_11173] (by decide)

def p_499500151 : CertifiedPrime :=
  certify 499500151 (3) [p_2, p_3, p_5, p_5, p_149, p_22349] (by decide)

def p_499522501 : CertifiedPrime :=
  certify 499522501 (10) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_149, p_149] (by decide)

def p_499634257 : CertifiedPrime :=
  certify 499634257 (5) [p_2, p_2, p_2, p_2, p_3, p_11, p_127, p_7451] (by decide)

def p_499790737 : CertifiedPrime :=
  certify 499790737 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_23, p_23] (by decide)

def p_499947241 : CertifiedPrime :=
  certify 499947241 (7) [p_2, p_2, p_2, p_3, p_5, p_13, p_29, p_43, p_257] (by decide)

def p_500170861 : CertifiedPrime :=
  certify 500170861 (6) [p_2, p_2, p_3, p_3, p_5, p_7, p_71, p_5591] (by decide)

def p_500237957 : CertifiedPrime :=
  certify 500237957 (2) [p_2, p_2, p_53, p_53, p_211, p_211] (by decide)

def p_500349793 : CertifiedPrime :=
  certify 500349793 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_233, p_22369] (by decide)

def p_500842021 : CertifiedPrime :=
  certify 500842021 (6) [p_2, p_2, p_3, p_5, p_7, p_23, p_139, p_373] (by decide)

def p_501424057 : CertifiedPrime :=
  certify 501424057 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_311, p_457] (by decide)

def p_501916813 : CertifiedPrime :=
  certify 501916813 (5) [p_2, p_2, p_3, p_43, p_521, p_1867] (by decide)

def p_502051243 : CertifiedPrime :=
  certify 502051243 (2) [p_2, p_3, p_7, p_11, p_17, p_97, p_659] (by decide)

def p_502230511 : CertifiedPrime :=
  certify 502230511 (31) [p_2, p_3, p_3, p_3, p_5, p_73, p_83, p_307] (by decide)

def p_502387397 : CertifiedPrime :=
  certify 502387397 (2) [p_2, p_2, p_7, p_7, p_1601, p_1601] (by decide)

def p_502768507 : CertifiedPrime :=
  certify 502768507 (3) [p_2, p_3, p_17, p_37, p_101, p_1319] (by decide)

def p_502835777 : CertifiedPrime :=
  certify 502835777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2803, p_2803] (by decide)

def p_503396533 : CertifiedPrime :=
  certify 503396533 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_71, p_79, p_277] (by decide)

def p_503800471 : CertifiedPrime :=
  certify 503800471 (11) [p_2, p_3, p_3, p_5, p_29, p_43, p_67, p_67] (by decide)

def p_504002501 : CertifiedPrime :=
  certify 504002501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_449, p_449] (by decide)

def p_504249481 : CertifiedPrime :=
  certify 504249481 (11) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_401, p_499] (by decide)

def p_504451601 : CertifiedPrime :=
  certify 504451601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1123, p_1123] (by decide)

def p_504721157 : CertifiedPrime :=
  certify 504721157 (2) [p_2, p_2, p_47, p_47, p_239, p_239] (by decide)

def p_505080677 : CertifiedPrime :=
  certify 505080677 (2) [p_2, p_2, p_17, p_17, p_661, p_661] (by decide)

def p_505282963 : CertifiedPrime :=
  certify 505282963 (2) [p_2, p_3, p_59, p_127, p_11239] (by decide)

def p_505350401 : CertifiedPrime :=
  certify 505350401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_281, p_281] (by decide)

def p_505620197 : CertifiedPrime :=
  certify 505620197 (2) [p_2, p_2, p_11243, p_11243] (by decide)

def p_505687657 : CertifiedPrime :=
  certify 505687657 (5) [p_2, p_2, p_2, p_3, p_113, p_199, p_937] (by decide)

def p_505980037 : CertifiedPrime :=
  certify 505980037 (2) [p_2, p_2, p_3, p_3, p_23, p_23, p_163, p_163] (by decide)

def p_506002531 : CertifiedPrime :=
  certify 506002531 (3) [p_2, p_3, p_5, p_11, p_23, p_163, p_409] (by decide)

def p_506700101 : CertifiedPrime :=
  certify 506700101 (7) [p_2, p_2, p_5, p_5, p_2251, p_2251] (by decide)

def p_506947741 : CertifiedPrime :=
  certify 506947741 (2) [p_2, p_2, p_3, p_5, p_13, p_19, p_79, p_433] (by decide)

def p_507082843 : CertifiedPrime :=
  certify 507082843 (7) [p_2, p_3, p_3, p_3, p_3, p_7, p_139, p_3217] (by decide)

def p_507172921 : CertifiedPrime :=
  certify 507172921 (19) [p_2, p_2, p_2, p_3, p_5, p_563, p_7507] (by decide)

def p_507353101 : CertifiedPrime :=
  certify 507353101 (6) [p_2, p_2, p_3, p_5, p_5, p_17, p_53, p_1877] (by decide)

def p_507578371 : CertifiedPrime :=
  certify 507578371 (12) [p_2, p_3, p_5, p_13, p_751, p_1733] (by decide)

def p_507871297 : CertifiedPrime :=
  certify 507871297 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_313, p_313] (by decide)

def p_508051601 : CertifiedPrime :=
  certify 508051601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_7, p_7, p_23, p_23] (by decide)

def p_508231937 : CertifiedPrime :=
  certify 508231937 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1409, p_1409] (by decide)

def p_508389757 : CertifiedPrime :=
  certify 508389757 (5) [p_2, p_2, p_3, p_7, p_1879, p_3221] (by decide)

def p_508773137 : CertifiedPrime :=
  certify 508773137 (3) [p_2, p_2, p_2, p_2, p_5639, p_5639] (by decide)

def p_509788663 : CertifiedPrime :=
  certify 509788663 (3) [p_2, p_3, p_53, p_67, p_71, p_337] (by decide)

def p_510285511 : CertifiedPrime :=
  certify 510285511 (7) [p_2, p_3, p_3, p_5, p_7, p_7, p_251, p_461] (by decide)

def p_510827803 : CertifiedPrime :=
  certify 510827803 (3) [p_2, p_3, p_97, p_233, p_3767] (by decide)

def p_511483457 : CertifiedPrime :=
  certify 511483457 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_257, p_257] (by decide)

def p_511845377 : CertifiedPrime :=
  certify 511845377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_101, p_101] (by decide)

def p_511935877 : CertifiedPrime :=
  certify 511935877 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_419, p_419] (by decide)

def p_512116901 : CertifiedPrime :=
  certify 512116901 (2) [p_2, p_2, p_5, p_5, p_31, p_31, p_73, p_73] (by decide)

def p_512184793 : CertifiedPrime :=
  certify 512184793 (15) [p_2, p_2, p_2, p_3, p_7, p_23, p_41, p_53, p_61] (by decide)

def p_512230057 : CertifiedPrime :=
  certify 512230057 (5) [p_2, p_2, p_2, p_3, p_13, p_23, p_41, p_1741] (by decide)

def p_512456407 : CertifiedPrime :=
  certify 512456407 (3) [p_2, p_3, p_7, p_7, p_7, p_11, p_22637] (by decide)

def p_512637523 : CertifiedPrime :=
  certify 512637523 (2) [p_2, p_3, p_7547, p_11321] (by decide)

def p_512728093 : CertifiedPrime :=
  certify 512728093 (5) [p_2, p_2, p_3, p_3, p_17, p_37, p_22643] (by decide)

def p_512863963 : CertifiedPrime :=
  certify 512863963 (2) [p_2, p_3, p_13, p_13, p_67, p_7549] (by decide)

def p_513181063 : CertifiedPrime :=
  certify 513181063 (6) [p_2, p_3, p_3, p_3, p_47, p_241, p_839] (by decide)

def p_513294337 : CertifiedPrime :=
  certify 513294337 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_59, p_59] (by decide)

def p_513407623 : CertifiedPrime :=
  certify 513407623 (3) [p_2, p_3, p_7, p_13, p_83, p_11329] (by decide)

def p_514223653 : CertifiedPrime :=
  certify 514223653 (2) [p_2, p_2, p_3, p_5669, p_7559] (by decide)

def p_514405081 : CertifiedPrime :=
  certify 514405081 (13) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_37, p_613] (by decide)

def p_514654597 : CertifiedPrime :=
  certify 514654597 (2) [p_2, p_2, p_3, p_3, p_19, p_19, p_199, p_199] (by decide)

def p_515017637 : CertifiedPrime :=
  certify 515017637 (2) [p_2, p_2, p_7, p_7, p_1621, p_1621] (by decide)

def p_515085721 : CertifiedPrime :=
  certify 515085721 (7) [p_2, p_2, p_2, p_3, p_5, p_17, p_89, p_2837] (by decide)

def p_515290001 : CertifiedPrime :=
  certify 515290001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_227, p_227] (by decide)

def p_515471617 : CertifiedPrime :=
  certify 515471617 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11, p_43, p_43] (by decide)

def p_515494321 : CertifiedPrime :=
  certify 515494321 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_11, p_19, p_43, p_239] (by decide)

def p_516266563 : CertifiedPrime :=
  certify 516266563 (3) [p_2, p_3, p_7, p_541, p_22721] (by decide)

def p_516312007 : CertifiedPrime :=
  certify 516312007 (3) [p_2, p_3, p_7, p_31, p_541, p_733] (by decide)

def p_516448351 : CertifiedPrime :=
  certify 516448351 (6) [p_2, p_3, p_3, p_5, p_5, p_11, p_101, p_1033] (by decide)

def p_516471077 : CertifiedPrime :=
  certify 516471077 (2) [p_2, p_2, p_11, p_11, p_1033, p_1033] (by decide)

def p_516857491 : CertifiedPrime :=
  certify 516857491 (10) [p_2, p_3, p_3, p_3, p_5, p_421, p_4547] (by decide)

def p_516948433 : CertifiedPrime :=
  certify 516948433 (10) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_11, p_13, p_29, p_53] (by decide)

def p_517266793 : CertifiedPrime :=
  certify 517266793 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_19, p_19, p_2843] (by decide)

def p_517539751 : CertifiedPrime :=
  certify 517539751 (3) [p_2, p_3, p_5, p_5, p_5, p_7, p_13, p_7583] (by decide)

def p_517630753 : CertifiedPrime :=
  certify 517630753 (17) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_79, p_22751] (by decide)

def p_517812781 : CertifiedPrime :=
  certify 517812781 (6) [p_2, p_2, p_3, p_5, p_37, p_41, p_5689] (by decide)

def p_517949323 : CertifiedPrime :=
  certify 517949323 (3) [p_2, p_3, p_11, p_2069, p_3793] (by decide)

def p_518040361 : CertifiedPrime :=
  certify 518040361 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_281, p_569] (by decide)

def p_518176933 : CertifiedPrime :=
  certify 518176933 (5) [p_2, p_2, p_3, p_7, p_13, p_17, p_103, p_271] (by decide)

def p_518290757 : CertifiedPrime :=
  certify 518290757 (2) [p_2, p_2, p_11383, p_11383] (by decide)

def p_518928401 : CertifiedPrime :=
  certify 518928401 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_67, p_67] (by decide)

def p_519384101 : CertifiedPrime :=
  certify 519384101 (2) [p_2, p_2, p_5, p_5, p_43, p_43, p_53, p_53] (by decide)

def p_519543643 : CertifiedPrime :=
  certify 519543643 (3) [p_2, p_3, p_23, p_29, p_131, p_991] (by decide)

def p_519657617 : CertifiedPrime :=
  certify 519657617 (3) [p_2, p_2, p_2, p_2, p_41, p_41, p_139, p_139] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
