import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk001

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_1621 : CertifiedPrime :=
  certify 1621 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5] (by decide)

def p_1627 : CertifiedPrime :=
  certify 1627 (3) [p_2, p_3, p_271] (by decide)

def p_1637 : CertifiedPrime :=
  certify 1637 (2) [p_2, p_2, p_409] (by decide)

def p_1657 : CertifiedPrime :=
  certify 1657 (11) [p_2, p_2, p_2, p_3, p_3, p_23] (by decide)

def p_1663 : CertifiedPrime :=
  certify 1663 (3) [p_2, p_3, p_277] (by decide)

def p_1667 : CertifiedPrime :=
  certify 1667 (2) [p_2, p_7, p_7, p_17] (by decide)

def p_1669 : CertifiedPrime :=
  certify 1669 (2) [p_2, p_2, p_3, p_139] (by decide)

def p_1693 : CertifiedPrime :=
  certify 1693 (2) [p_2, p_2, p_3, p_3, p_47] (by decide)

def p_1697 : CertifiedPrime :=
  certify 1697 (3) [p_2, p_2, p_2, p_2, p_2, p_53] (by decide)

def p_1699 : CertifiedPrime :=
  certify 1699 (3) [p_2, p_3, p_283] (by decide)

def p_1709 : CertifiedPrime :=
  certify 1709 (3) [p_2, p_2, p_7, p_61] (by decide)

def p_1721 : CertifiedPrime :=
  certify 1721 (3) [p_2, p_2, p_2, p_5, p_43] (by decide)

def p_1723 : CertifiedPrime :=
  certify 1723 (3) [p_2, p_3, p_7, p_41] (by decide)

def p_1733 : CertifiedPrime :=
  certify 1733 (2) [p_2, p_2, p_433] (by decide)

def p_1741 : CertifiedPrime :=
  certify 1741 (2) [p_2, p_2, p_3, p_5, p_29] (by decide)

def p_1747 : CertifiedPrime :=
  certify 1747 (2) [p_2, p_3, p_3, p_97] (by decide)

def p_1753 : CertifiedPrime :=
  certify 1753 (7) [p_2, p_2, p_2, p_3, p_73] (by decide)

def p_1759 : CertifiedPrime :=
  certify 1759 (6) [p_2, p_3, p_293] (by decide)

def p_1777 : CertifiedPrime :=
  certify 1777 (5) [p_2, p_2, p_2, p_2, p_3, p_37] (by decide)

def p_1783 : CertifiedPrime :=
  certify 1783 (10) [p_2, p_3, p_3, p_3, p_3, p_11] (by decide)

def p_1787 : CertifiedPrime :=
  certify 1787 (2) [p_2, p_19, p_47] (by decide)

def p_1789 : CertifiedPrime :=
  certify 1789 (6) [p_2, p_2, p_3, p_149] (by decide)

def p_1801 : CertifiedPrime :=
  certify 1801 (11) [p_2, p_2, p_2, p_3, p_3, p_5, p_5] (by decide)

def p_1811 : CertifiedPrime :=
  certify 1811 (6) [p_2, p_5, p_181] (by decide)

def p_1823 : CertifiedPrime :=
  certify 1823 (5) [p_2, p_911] (by decide)

def p_1831 : CertifiedPrime :=
  certify 1831 (3) [p_2, p_3, p_5, p_61] (by decide)

def p_1847 : CertifiedPrime :=
  certify 1847 (5) [p_2, p_13, p_71] (by decide)

def p_1861 : CertifiedPrime :=
  certify 1861 (2) [p_2, p_2, p_3, p_5, p_31] (by decide)

def p_1867 : CertifiedPrime :=
  certify 1867 (2) [p_2, p_3, p_311] (by decide)

def p_1871 : CertifiedPrime :=
  certify 1871 (14) [p_2, p_5, p_11, p_17] (by decide)

def p_1873 : CertifiedPrime :=
  certify 1873 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_13] (by decide)

def p_1877 : CertifiedPrime :=
  certify 1877 (2) [p_2, p_2, p_7, p_67] (by decide)

def p_1879 : CertifiedPrime :=
  certify 1879 (6) [p_2, p_3, p_313] (by decide)

def p_1889 : CertifiedPrime :=
  certify 1889 (3) [p_2, p_2, p_2, p_2, p_2, p_59] (by decide)

def p_1901 : CertifiedPrime :=
  certify 1901 (2) [p_2, p_2, p_5, p_5, p_19] (by decide)

def p_1907 : CertifiedPrime :=
  certify 1907 (2) [p_2, p_953] (by decide)

def p_1913 : CertifiedPrime :=
  certify 1913 (3) [p_2, p_2, p_2, p_239] (by decide)

def p_1931 : CertifiedPrime :=
  certify 1931 (2) [p_2, p_5, p_193] (by decide)

def p_1933 : CertifiedPrime :=
  certify 1933 (5) [p_2, p_2, p_3, p_7, p_23] (by decide)

def p_1949 : CertifiedPrime :=
  certify 1949 (2) [p_2, p_2, p_487] (by decide)

def p_1951 : CertifiedPrime :=
  certify 1951 (3) [p_2, p_3, p_5, p_5, p_13] (by decide)

def p_1973 : CertifiedPrime :=
  certify 1973 (2) [p_2, p_2, p_17, p_29] (by decide)

def p_1979 : CertifiedPrime :=
  certify 1979 (2) [p_2, p_23, p_43] (by decide)

def p_1987 : CertifiedPrime :=
  certify 1987 (2) [p_2, p_3, p_331] (by decide)

def p_1993 : CertifiedPrime :=
  certify 1993 (5) [p_2, p_2, p_2, p_3, p_83] (by decide)

def p_1997 : CertifiedPrime :=
  certify 1997 (2) [p_2, p_2, p_499] (by decide)

def p_1999 : CertifiedPrime :=
  certify 1999 (3) [p_2, p_3, p_3, p_3, p_37] (by decide)

def p_2003 : CertifiedPrime :=
  certify 2003 (5) [p_2, p_7, p_11, p_13] (by decide)

def p_2011 : CertifiedPrime :=
  certify 2011 (3) [p_2, p_3, p_5, p_67] (by decide)

def p_2017 : CertifiedPrime :=
  certify 2017 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7] (by decide)

def p_2027 : CertifiedPrime :=
  certify 2027 (2) [p_2, p_1013] (by decide)

def p_2029 : CertifiedPrime :=
  certify 2029 (2) [p_2, p_2, p_3, p_13, p_13] (by decide)

def p_2039 : CertifiedPrime :=
  certify 2039 (7) [p_2, p_1019] (by decide)

def p_2053 : CertifiedPrime :=
  certify 2053 (2) [p_2, p_2, p_3, p_3, p_3, p_19] (by decide)

def p_2063 : CertifiedPrime :=
  certify 2063 (5) [p_2, p_1031] (by decide)

def p_2069 : CertifiedPrime :=
  certify 2069 (2) [p_2, p_2, p_11, p_47] (by decide)

def p_2081 : CertifiedPrime :=
  certify 2081 (3) [p_2, p_2, p_2, p_2, p_2, p_5, p_13] (by decide)

def p_2083 : CertifiedPrime :=
  certify 2083 (2) [p_2, p_3, p_347] (by decide)

def p_2087 : CertifiedPrime :=
  certify 2087 (5) [p_2, p_7, p_149] (by decide)

def p_2089 : CertifiedPrime :=
  certify 2089 (7) [p_2, p_2, p_2, p_3, p_3, p_29] (by decide)

def p_2099 : CertifiedPrime :=
  certify 2099 (2) [p_2, p_1049] (by decide)

def p_2111 : CertifiedPrime :=
  certify 2111 (7) [p_2, p_5, p_211] (by decide)

def p_2113 : CertifiedPrime :=
  certify 2113 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_11] (by decide)

def p_2129 : CertifiedPrime :=
  certify 2129 (3) [p_2, p_2, p_2, p_2, p_7, p_19] (by decide)

def p_2131 : CertifiedPrime :=
  certify 2131 (2) [p_2, p_3, p_5, p_71] (by decide)

def p_2137 : CertifiedPrime :=
  certify 2137 (10) [p_2, p_2, p_2, p_3, p_89] (by decide)

def p_2141 : CertifiedPrime :=
  certify 2141 (2) [p_2, p_2, p_5, p_107] (by decide)

def p_2143 : CertifiedPrime :=
  certify 2143 (3) [p_2, p_3, p_3, p_7, p_17] (by decide)

def p_2153 : CertifiedPrime :=
  certify 2153 (3) [p_2, p_2, p_2, p_269] (by decide)

def p_2161 : CertifiedPrime :=
  certify 2161 (23) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5] (by decide)

def p_2179 : CertifiedPrime :=
  certify 2179 (7) [p_2, p_3, p_3, p_11, p_11] (by decide)

def p_2203 : CertifiedPrime :=
  certify 2203 (5) [p_2, p_3, p_367] (by decide)

def p_2207 : CertifiedPrime :=
  certify 2207 (5) [p_2, p_1103] (by decide)

def p_2213 : CertifiedPrime :=
  certify 2213 (2) [p_2, p_2, p_7, p_79] (by decide)

def p_2221 : CertifiedPrime :=
  certify 2221 (2) [p_2, p_2, p_3, p_5, p_37] (by decide)

def p_2237 : CertifiedPrime :=
  certify 2237 (2) [p_2, p_2, p_13, p_43] (by decide)

def p_2239 : CertifiedPrime :=
  certify 2239 (3) [p_2, p_3, p_373] (by decide)

def p_2243 : CertifiedPrime :=
  certify 2243 (2) [p_2, p_19, p_59] (by decide)

def p_2251 : CertifiedPrime :=
  certify 2251 (7) [p_2, p_3, p_3, p_5, p_5, p_5] (by decide)

def p_2267 : CertifiedPrime :=
  certify 2267 (2) [p_2, p_11, p_103] (by decide)

def p_2269 : CertifiedPrime :=
  certify 2269 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_7] (by decide)

def p_2273 : CertifiedPrime :=
  certify 2273 (3) [p_2, p_2, p_2, p_2, p_2, p_71] (by decide)

def p_2281 : CertifiedPrime :=
  certify 2281 (7) [p_2, p_2, p_2, p_3, p_5, p_19] (by decide)

def p_2287 : CertifiedPrime :=
  certify 2287 (19) [p_2, p_3, p_3, p_127] (by decide)

def p_2293 : CertifiedPrime :=
  certify 2293 (2) [p_2, p_2, p_3, p_191] (by decide)

def p_2297 : CertifiedPrime :=
  certify 2297 (5) [p_2, p_2, p_2, p_7, p_41] (by decide)

def p_2309 : CertifiedPrime :=
  certify 2309 (2) [p_2, p_2, p_577] (by decide)

def p_2311 : CertifiedPrime :=
  certify 2311 (3) [p_2, p_3, p_5, p_7, p_11] (by decide)

def p_2333 : CertifiedPrime :=
  certify 2333 (2) [p_2, p_2, p_11, p_53] (by decide)

def p_2339 : CertifiedPrime :=
  certify 2339 (2) [p_2, p_7, p_167] (by decide)

def p_2341 : CertifiedPrime :=
  certify 2341 (7) [p_2, p_2, p_3, p_3, p_5, p_13] (by decide)

def p_2347 : CertifiedPrime :=
  certify 2347 (3) [p_2, p_3, p_17, p_23] (by decide)

def p_2351 : CertifiedPrime :=
  certify 2351 (13) [p_2, p_5, p_5, p_47] (by decide)

def p_2357 : CertifiedPrime :=
  certify 2357 (2) [p_2, p_2, p_19, p_31] (by decide)

def p_2371 : CertifiedPrime :=
  certify 2371 (2) [p_2, p_3, p_5, p_79] (by decide)

def p_2377 : CertifiedPrime :=
  certify 2377 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_11] (by decide)

def p_2381 : CertifiedPrime :=
  certify 2381 (3) [p_2, p_2, p_5, p_7, p_17] (by decide)

def p_2383 : CertifiedPrime :=
  certify 2383 (5) [p_2, p_3, p_397] (by decide)

def p_2389 : CertifiedPrime :=
  certify 2389 (2) [p_2, p_2, p_3, p_199] (by decide)

def p_2393 : CertifiedPrime :=
  certify 2393 (3) [p_2, p_2, p_2, p_13, p_23] (by decide)

def p_2399 : CertifiedPrime :=
  certify 2399 (11) [p_2, p_11, p_109] (by decide)

def p_2411 : CertifiedPrime :=
  certify 2411 (6) [p_2, p_5, p_241] (by decide)

def p_2417 : CertifiedPrime :=
  certify 2417 (3) [p_2, p_2, p_2, p_2, p_151] (by decide)

def p_2423 : CertifiedPrime :=
  certify 2423 (5) [p_2, p_7, p_173] (by decide)

def p_2437 : CertifiedPrime :=
  certify 2437 (2) [p_2, p_2, p_3, p_7, p_29] (by decide)

def p_2441 : CertifiedPrime :=
  certify 2441 (6) [p_2, p_2, p_2, p_5, p_61] (by decide)

def p_2447 : CertifiedPrime :=
  certify 2447 (5) [p_2, p_1223] (by decide)

def p_2459 : CertifiedPrime :=
  certify 2459 (2) [p_2, p_1229] (by decide)

def p_2467 : CertifiedPrime :=
  certify 2467 (2) [p_2, p_3, p_3, p_137] (by decide)

def p_2473 : CertifiedPrime :=
  certify 2473 (5) [p_2, p_2, p_2, p_3, p_103] (by decide)

def p_2477 : CertifiedPrime :=
  certify 2477 (2) [p_2, p_2, p_619] (by decide)

def p_2503 : CertifiedPrime :=
  certify 2503 (3) [p_2, p_3, p_3, p_139] (by decide)

def p_2521 : CertifiedPrime :=
  certify 2521 (17) [p_2, p_2, p_2, p_3, p_3, p_5, p_7] (by decide)

def p_2531 : CertifiedPrime :=
  certify 2531 (2) [p_2, p_5, p_11, p_23] (by decide)

def p_2539 : CertifiedPrime :=
  certify 2539 (2) [p_2, p_3, p_3, p_3, p_47] (by decide)

def p_2543 : CertifiedPrime :=
  certify 2543 (5) [p_2, p_31, p_41] (by decide)

def p_2549 : CertifiedPrime :=
  certify 2549 (2) [p_2, p_2, p_7, p_7, p_13] (by decide)

def p_2551 : CertifiedPrime :=
  certify 2551 (6) [p_2, p_3, p_5, p_5, p_17] (by decide)

def p_2557 : CertifiedPrime :=
  certify 2557 (2) [p_2, p_2, p_3, p_3, p_71] (by decide)

def p_2579 : CertifiedPrime :=
  certify 2579 (2) [p_2, p_1289] (by decide)

def p_2591 : CertifiedPrime :=
  certify 2591 (7) [p_2, p_5, p_7, p_37] (by decide)

def p_2593 : CertifiedPrime :=
  certify 2593 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3] (by decide)

def p_2609 : CertifiedPrime :=
  certify 2609 (3) [p_2, p_2, p_2, p_2, p_163] (by decide)

def p_2617 : CertifiedPrime :=
  certify 2617 (5) [p_2, p_2, p_2, p_3, p_109] (by decide)

def p_2621 : CertifiedPrime :=
  certify 2621 (2) [p_2, p_2, p_5, p_131] (by decide)

def p_2633 : CertifiedPrime :=
  certify 2633 (3) [p_2, p_2, p_2, p_7, p_47] (by decide)

def p_2647 : CertifiedPrime :=
  certify 2647 (3) [p_2, p_3, p_3, p_3, p_7, p_7] (by decide)

def p_2657 : CertifiedPrime :=
  certify 2657 (3) [p_2, p_2, p_2, p_2, p_2, p_83] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
