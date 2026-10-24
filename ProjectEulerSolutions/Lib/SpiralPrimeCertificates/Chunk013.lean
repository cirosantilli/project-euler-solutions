import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk012

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_2131601 : CertifiedPrime :=
  certify 2131601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_73, p_73] (by decide)

def p_2171203 : CertifiedPrime :=
  certify 2171203 (2) [p_2, p_3, p_11, p_67, p_491] (by decide)

def p_2177101 : CertifiedPrime :=
  certify 2177101 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_41, p_59] (by decide)

def p_2203741 : CertifiedPrime :=
  certify 2203741 (6) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_11, p_53] (by decide)

def p_2212657 : CertifiedPrime :=
  certify 2212657 (7) [p_2, p_2, p_2, p_2, p_3, p_31, p_1487] (by decide)

def p_2232037 : CertifiedPrime :=
  certify 2232037 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_83, p_83] (by decide)

def p_2233531 : CertifiedPrime :=
  certify 2233531 (10) [p_2, p_3, p_3, p_5, p_13, p_23, p_83] (by decide)

def p_2262017 : CertifiedPrime :=
  certify 2262017 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_47, p_47] (by decide)

def p_2266531 : CertifiedPrime :=
  certify 2266531 (3) [p_2, p_3, p_5, p_7, p_43, p_251] (by decide)

def p_2296741 : CertifiedPrime :=
  certify 2296741 (2) [p_2, p_2, p_3, p_5, p_101, p_379] (by decide)

def p_2311921 : CertifiedPrime :=
  certify 2311921 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_13, p_13, p_19] (by decide)

def p_2322577 : CertifiedPrime :=
  certify 2322577 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_127, p_127] (by decide)

def p_2330203 : CertifiedPrime :=
  certify 2330203 (2) [p_2, p_3, p_7, p_109, p_509] (by decide)

def p_2339371 : CertifiedPrime :=
  certify 2339371 (11) [p_2, p_3, p_3, p_5, p_11, p_17, p_139] (by decide)

def p_2360833 : CertifiedPrime :=
  certify 2360833 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_29, p_53] (by decide)

def p_2370061 : CertifiedPrime :=
  certify 2370061 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_11, p_19] (by decide)

def p_2390117 : CertifiedPrime :=
  certify 2390117 (2) [p_2, p_2, p_773, p_773] (by decide)

def p_2394757 : CertifiedPrime :=
  certify 2394757 (5) [p_2, p_2, p_3, p_3, p_7, p_13, p_17, p_43] (by decide)

def p_2397853 : CertifiedPrime :=
  certify 2397853 (7) [p_2, p_2, p_3, p_3, p_43, p_1549] (by decide)

def p_2402501 : CertifiedPrime :=
  certify 2402501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_31, p_31] (by decide)

def p_2421137 : CertifiedPrime :=
  certify 2421137 (3) [p_2, p_2, p_2, p_2, p_389, p_389] (by decide)

def p_2432041 : CertifiedPrime :=
  certify 2432041 (7) [p_2, p_2, p_2, p_3, p_5, p_13, p_1559] (by decide)

def p_2435161 : CertifiedPrime :=
  certify 2435161 (43) [p_2, p_2, p_2, p_3, p_5, p_7, p_13, p_223] (by decide)

def p_2441407 : CertifiedPrime :=
  certify 2441407 (3) [p_2, p_3, p_11, p_71, p_521] (by decide)

def p_2446097 : CertifiedPrime :=
  certify 2446097 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_23, p_23] (by decide)

def p_2452357 : CertifiedPrime :=
  certify 2452357 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_29, p_29] (by decide)

def p_2453923 : CertifiedPrime :=
  certify 2453923 (5) [p_2, p_3, p_3, p_3, p_29, p_1567] (by decide)

def p_2460193 : CertifiedPrime :=
  certify 2460193 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_523] (by decide)

def p_2464901 : CertifiedPrime :=
  certify 2464901 (2) [p_2, p_2, p_5, p_5, p_157, p_157] (by decide)

def p_2483777 : CertifiedPrime :=
  certify 2483777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_197, p_197] (by decide)

def p_2496401 : CertifiedPrime :=
  certify 2496401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_79, p_79] (by decide)

def p_2501143 : CertifiedPrime :=
  certify 2501143 (3) [p_2, p_3, p_7, p_17, p_31, p_113] (by decide)

def p_2507473 : CertifiedPrime :=
  certify 2507473 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_1583] (by decide)

def p_2515397 : CertifiedPrime :=
  certify 2515397 (2) [p_2, p_2, p_13, p_13, p_61, p_61] (by decide)

def p_2529691 : CertifiedPrime :=
  certify 2529691 (3) [p_2, p_3, p_5, p_37, p_43, p_53] (by decide)

def p_2536057 : CertifiedPrime :=
  certify 2536057 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_59, p_199] (by decide)

def p_2548813 : CertifiedPrime :=
  certify 2548813 (5) [p_2, p_2, p_3, p_7, p_19, p_1597] (by decide)

def p_2596933 : CertifiedPrime :=
  certify 2596933 (2) [p_2, p_2, p_3, p_3, p_13, p_31, p_179] (by decide)

def p_2604997 : CertifiedPrime :=
  certify 2604997 (5) [p_2, p_2, p_3, p_3, p_269, p_269] (by decide)

def p_2611457 : CertifiedPrime :=
  certify 2611457 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_101, p_101] (by decide)

def p_2613073 : CertifiedPrime :=
  certify 2613073 (13) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_11, p_101] (by decide)

def p_2626021 : CertifiedPrime :=
  certify 2626021 (7) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_1621] (by decide)

def p_2665057 : CertifiedPrime :=
  certify 2665057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_17, p_23, p_71] (by decide)

def p_2689601 : CertifiedPrime :=
  certify 2689601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_41, p_41] (by decide)

def p_2702737 : CertifiedPrime :=
  certify 2702737 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_137, p_137] (by decide)

def p_2704381 : CertifiedPrime :=
  certify 2704381 (6) [p_2, p_2, p_3, p_5, p_7, p_47, p_137] (by decide)

def p_2710963 : CertifiedPrime :=
  certify 2710963 (2) [p_2, p_3, p_3, p_3, p_61, p_823] (by decide)

def p_2724151 : CertifiedPrime :=
  certify 2724151 (6) [p_2, p_3, p_5, p_5, p_11, p_13, p_127] (by decide)

def p_2735717 : CertifiedPrime :=
  certify 2735717 (2) [p_2, p_2, p_827, p_827] (by decide)

def p_2750623 : CertifiedPrime :=
  certify 2750623 (5) [p_2, p_3, p_7, p_79, p_829] (by decide)

def p_2753941 : CertifiedPrime :=
  certify 2753941 (2) [p_2, p_2, p_3, p_5, p_7, p_79, p_83] (by decide)

def p_2755601 : CertifiedPrime :=
  certify 2755601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_83, p_83] (by decide)

def p_2768897 : CertifiedPrime :=
  certify 2768897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13] (by decide)

def p_2773891 : CertifiedPrime :=
  certify 2773891 (2) [p_2, p_3, p_3, p_5, p_7, p_7, p_17, p_37] (by decide)

def p_2802277 : CertifiedPrime :=
  certify 2802277 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_31, p_31] (by decide)

def p_2808977 : CertifiedPrime :=
  certify 2808977 (3) [p_2, p_2, p_2, p_2, p_419, p_419] (by decide)

def p_2810653 : CertifiedPrime :=
  certify 2810653 (2) [p_2, p_2, p_3, p_13, p_43, p_419] (by decide)

def p_2835857 : CertifiedPrime :=
  certify 2835857 (3) [p_2, p_2, p_2, p_2, p_421, p_421] (by decide)

def p_2840911 : CertifiedPrime :=
  certify 2840911 (3) [p_2, p_3, p_5, p_281, p_337] (by decide)

def p_2842597 : CertifiedPrime :=
  certify 2842597 (5) [p_2, p_2, p_3, p_3, p_281, p_281] (by decide)

def p_2864557 : CertifiedPrime :=
  certify 2864557 (5) [p_2, p_2, p_3, p_3, p_47, p_1693] (by decide)

def p_2874721 : CertifiedPrime :=
  certify 2874721 (14) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_53, p_113] (by decide)

def p_2890001 : CertifiedPrime :=
  certify 2890001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_17, p_17] (by decide)

def p_2891701 : CertifiedPrime :=
  certify 2891701 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_17] (by decide)

def p_2895103 : CertifiedPrime :=
  certify 2895103 (5) [p_2, p_3, p_3, p_3, p_3, p_3, p_7, p_23, p_37] (by decide)

def p_2915557 : CertifiedPrime :=
  certify 2915557 (11) [p_2, p_2, p_3, p_7, p_61, p_569] (by decide)

def p_2922391 : CertifiedPrime :=
  certify 2922391 (17) [p_2, p_3, p_3, p_5, p_19, p_1709] (by decide)

def p_2944657 : CertifiedPrime :=
  certify 2944657 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11, p_13, p_13] (by decide)

def p_2946373 : CertifiedPrime :=
  certify 2946373 (5) [p_2, p_2, p_3, p_11, p_13, p_17, p_101] (by decide)

def p_2953243 : CertifiedPrime :=
  certify 2953243 (2) [p_2, p_3, p_3, p_191, p_859] (by decide)

def p_3013697 : CertifiedPrime :=
  certify 3013697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_31, p_31] (by decide)

def p_3050263 : CertifiedPrime :=
  certify 3050263 (5) [p_2, p_3, p_3, p_97, p_1747] (by decide)

def p_3057253 : CertifiedPrime :=
  certify 3057253 (2) [p_2, p_2, p_3, p_11, p_19, p_23, p_53] (by decide)

def p_3083537 : CertifiedPrime :=
  certify 3083537 (3) [p_2, p_2, p_2, p_2, p_439, p_439] (by decide)

def p_3088807 : CertifiedPrime :=
  certify 3088807 (3) [p_2, p_3, p_7, p_251, p_293] (by decide)

def p_3118757 : CertifiedPrime :=
  certify 3118757 (2) [p_2, p_2, p_883, p_883] (by decide)

def p_3124057 : CertifiedPrime :=
  certify 3124057 (5) [p_2, p_2, p_2, p_3, p_13, p_17, p_19, p_31] (by decide)

def p_3134671 : CertifiedPrime :=
  certify 3134671 (3) [p_2, p_3, p_5, p_7, p_11, p_23, p_59] (by decide)

def p_3147077 : CertifiedPrime :=
  certify 3147077 (2) [p_2, p_2, p_887, p_887] (by decide)

def p_3182657 : CertifiedPrime :=
  certify 3182657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_223, p_223] (by decide)

def p_3198733 : CertifiedPrime :=
  certify 3198733 (6) [p_2, p_2, p_3, p_149, p_1789] (by decide)

def p_3204101 : CertifiedPrime :=
  certify 3204101 (2) [p_2, p_2, p_5, p_5, p_179, p_179] (by decide)

def p_3218437 : CertifiedPrime :=
  certify 3218437 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_23, p_23] (by decide)

def p_3241801 : CertifiedPrime :=
  certify 3241801 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_1801] (by decide)

def p_3263443 : CertifiedPrime :=
  certify 3263443 (3) [p_2, p_3, p_7, p_13, p_43, p_139] (by decide)

def p_3274291 : CertifiedPrime :=
  certify 3274291 (2) [p_2, p_3, p_3, p_3, p_5, p_67, p_181] (by decide)

def p_3281533 : CertifiedPrime :=
  certify 3281533 (6) [p_2, p_2, p_3, p_151, p_1811] (by decide)

def p_3292411 : CertifiedPrime :=
  certify 3292411 (3) [p_2, p_3, p_5, p_11, p_11, p_907] (by decide)

def p_3297857 : CertifiedPrime :=
  certify 3297857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_227, p_227] (by decide)

def p_3306943 : CertifiedPrime :=
  certify 3306943 (23) [p_2, p_3, p_3, p_17, p_101, p_107] (by decide)

def p_3326977 : CertifiedPrime :=
  certify 3326977 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_19, p_19] (by decide)

def p_3336103 : CertifiedPrime :=
  certify 3336103 (6) [p_2, p_3, p_3, p_7, p_11, p_29, p_83] (by decide)

def p_3358057 : CertifiedPrime :=
  certify 3358057 (7) [p_2, p_2, p_2, p_3, p_13, p_47, p_229] (by decide)

def p_3391123 : CertifiedPrime :=
  certify 3391123 (3) [p_2, p_3, p_7, p_263, p_307] (by decide)

def p_3413257 : CertifiedPrime :=
  certify 3413257 (5) [p_2, p_2, p_2, p_3, p_7, p_11, p_1847] (by decide)

def p_3416953 : CertifiedPrime :=
  certify 3416953 (13) [p_2, p_2, p_2, p_3, p_7, p_11, p_43, p_43] (by decide)

def p_3422501 : CertifiedPrime :=
  certify 3422501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_37, p_37] (by decide)

def p_3435463 : CertifiedPrime :=
  certify 3435463 (10) [p_2, p_3, p_3, p_17, p_103, p_109] (by decide)

def p_3459601 : CertifiedPrime :=
  certify 3459601 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_31, p_31] (by decide)

def p_3472633 : CertifiedPrime :=
  certify 3472633 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_23, p_233] (by decide)

def p_3496901 : CertifiedPrime :=
  certify 3496901 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_17, p_17] (by decide)

def p_3519377 : CertifiedPrime :=
  certify 3519377 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_67, p_67] (by decide)

def p_3549457 : CertifiedPrime :=
  certify 3549457 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_157, p_157] (by decide)

def p_3558883 : CertifiedPrime :=
  certify 3558883 (2) [p_2, p_3, p_17, p_23, p_37, p_41] (by decide)

def p_3587237 : CertifiedPrime :=
  certify 3587237 (2) [p_2, p_2, p_947, p_947] (by decide)

def p_3592921 : CertifiedPrime :=
  certify 3592921 (7) [p_2, p_2, p_2, p_3, p_5, p_79, p_379] (by decide)

def p_3619507 : CertifiedPrime :=
  certify 3619507 (3) [p_2, p_3, p_11, p_173, p_317] (by decide)

def p_3648101 : CertifiedPrime :=
  certify 3648101 (2) [p_2, p_2, p_5, p_5, p_191, p_191] (by decide)

def p_3686401 : CertifiedPrime :=
  certify 3686401 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5] (by decide)

def p_3699853 : CertifiedPrime :=
  certify 3699853 (2) [p_2, p_2, p_3, p_13, p_37, p_641] (by decide)

def p_3730693 : CertifiedPrime :=
  certify 3730693 (11) [p_2, p_2, p_3, p_7, p_23, p_1931] (by decide)

def p_3734557 : CertifiedPrime :=
  certify 3734557 (5) [p_2, p_2, p_3, p_7, p_23, p_1933] (by decide)

def p_3763601 : CertifiedPrime :=
  certify 3763601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_97, p_97] (by decide)

def p_3812257 : CertifiedPrime :=
  certify 3812257 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_31, p_61] (by decide)

def p_3827893 : CertifiedPrime :=
  certify 3827893 (5) [p_2, p_2, p_3, p_19, p_103, p_163] (by decide)

def p_3857297 : CertifiedPrime :=
  certify 3857297 (3) [p_2, p_2, p_2, p_2, p_491, p_491] (by decide)

def p_3865157 : CertifiedPrime :=
  certify 3865157 (2) [p_2, p_2, p_983, p_983] (by decide)

def p_3880901 : CertifiedPrime :=
  certify 3880901 (7) [p_2, p_2, p_5, p_5, p_197, p_197] (by decide)

def p_3882871 : CertifiedPrime :=
  certify 3882871 (6) [p_2, p_3, p_3, p_3, p_5, p_73, p_197] (by decide)

def p_3894703 : CertifiedPrime :=
  certify 3894703 (3) [p_2, p_3, p_7, p_47, p_1973] (by decide)

def p_3896677 : CertifiedPrime :=
  certify 3896677 (5) [p_2, p_2, p_3, p_3, p_7, p_7, p_47, p_47] (by decide)

def p_3898651 : CertifiedPrime :=
  certify 3898651 (10) [p_2, p_3, p_5, p_5, p_7, p_47, p_79] (by decide)

def p_3910507 : CertifiedPrime :=
  certify 3910507 (2) [p_2, p_3, p_23, p_43, p_659] (by decide)

def p_3920401 : CertifiedPrime :=
  certify 3920401 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_11, p_11] (by decide)

def p_3930307 : CertifiedPrime :=
  certify 3930307 (2) [p_2, p_3, p_661, p_991] (by decide)

def p_3946183 : CertifiedPrime :=
  certify 3946183 (3) [p_2, p_3, p_331, p_1987] (by decide)

def p_3960101 : CertifiedPrime :=
  certify 3960101 (2) [p_2, p_2, p_5, p_5, p_199, p_199] (by decide)

def p_3978031 : CertifiedPrime :=
  certify 3978031 (6) [p_2, p_3, p_5, p_7, p_19, p_997] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
