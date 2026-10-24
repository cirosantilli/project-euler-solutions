import ProjectEulerSolutions.Lib.PrattCertificates

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_2 : CertifiedPrime :=
  certify 2 (1) [] (by decide)

def p_3 : CertifiedPrime :=
  certify 3 (2) [p_2] (by decide)

def p_5 : CertifiedPrime :=
  certify 5 (2) [p_2, p_2] (by decide)

def p_7 : CertifiedPrime :=
  certify 7 (3) [p_2, p_3] (by decide)

def p_11 : CertifiedPrime :=
  certify 11 (2) [p_2, p_5] (by decide)

def p_13 : CertifiedPrime :=
  certify 13 (2) [p_2, p_2, p_3] (by decide)

def p_17 : CertifiedPrime :=
  certify 17 (3) [p_2, p_2, p_2, p_2] (by decide)

def p_19 : CertifiedPrime :=
  certify 19 (2) [p_2, p_3, p_3] (by decide)

def p_23 : CertifiedPrime :=
  certify 23 (5) [p_2, p_11] (by decide)

def p_29 : CertifiedPrime :=
  certify 29 (2) [p_2, p_2, p_7] (by decide)

def p_31 : CertifiedPrime :=
  certify 31 (3) [p_2, p_3, p_5] (by decide)

def p_37 : CertifiedPrime :=
  certify 37 (2) [p_2, p_2, p_3, p_3] (by decide)

def p_41 : CertifiedPrime :=
  certify 41 (6) [p_2, p_2, p_2, p_5] (by decide)

def p_43 : CertifiedPrime :=
  certify 43 (3) [p_2, p_3, p_7] (by decide)

def p_47 : CertifiedPrime :=
  certify 47 (5) [p_2, p_23] (by decide)

def p_53 : CertifiedPrime :=
  certify 53 (2) [p_2, p_2, p_13] (by decide)

def p_59 : CertifiedPrime :=
  certify 59 (2) [p_2, p_29] (by decide)

def p_61 : CertifiedPrime :=
  certify 61 (2) [p_2, p_2, p_3, p_5] (by decide)

def p_67 : CertifiedPrime :=
  certify 67 (2) [p_2, p_3, p_11] (by decide)

def p_71 : CertifiedPrime :=
  certify 71 (7) [p_2, p_5, p_7] (by decide)

def p_73 : CertifiedPrime :=
  certify 73 (5) [p_2, p_2, p_2, p_3, p_3] (by decide)

def p_79 : CertifiedPrime :=
  certify 79 (3) [p_2, p_3, p_13] (by decide)

def p_83 : CertifiedPrime :=
  certify 83 (2) [p_2, p_41] (by decide)

def p_89 : CertifiedPrime :=
  certify 89 (3) [p_2, p_2, p_2, p_11] (by decide)

def p_97 : CertifiedPrime :=
  certify 97 (5) [p_2, p_2, p_2, p_2, p_2, p_3] (by decide)

def p_101 : CertifiedPrime :=
  certify 101 (2) [p_2, p_2, p_5, p_5] (by decide)

def p_103 : CertifiedPrime :=
  certify 103 (5) [p_2, p_3, p_17] (by decide)

def p_107 : CertifiedPrime :=
  certify 107 (2) [p_2, p_53] (by decide)

def p_109 : CertifiedPrime :=
  certify 109 (6) [p_2, p_2, p_3, p_3, p_3] (by decide)

def p_113 : CertifiedPrime :=
  certify 113 (3) [p_2, p_2, p_2, p_2, p_7] (by decide)

def p_127 : CertifiedPrime :=
  certify 127 (3) [p_2, p_3, p_3, p_7] (by decide)

def p_131 : CertifiedPrime :=
  certify 131 (2) [p_2, p_5, p_13] (by decide)

def p_137 : CertifiedPrime :=
  certify 137 (3) [p_2, p_2, p_2, p_17] (by decide)

def p_139 : CertifiedPrime :=
  certify 139 (2) [p_2, p_3, p_23] (by decide)

def p_149 : CertifiedPrime :=
  certify 149 (2) [p_2, p_2, p_37] (by decide)

def p_151 : CertifiedPrime :=
  certify 151 (6) [p_2, p_3, p_5, p_5] (by decide)

def p_157 : CertifiedPrime :=
  certify 157 (5) [p_2, p_2, p_3, p_13] (by decide)

def p_163 : CertifiedPrime :=
  certify 163 (2) [p_2, p_3, p_3, p_3, p_3] (by decide)

def p_167 : CertifiedPrime :=
  certify 167 (5) [p_2, p_83] (by decide)

def p_173 : CertifiedPrime :=
  certify 173 (2) [p_2, p_2, p_43] (by decide)

def p_179 : CertifiedPrime :=
  certify 179 (2) [p_2, p_89] (by decide)

def p_181 : CertifiedPrime :=
  certify 181 (2) [p_2, p_2, p_3, p_3, p_5] (by decide)

def p_191 : CertifiedPrime :=
  certify 191 (19) [p_2, p_5, p_19] (by decide)

def p_193 : CertifiedPrime :=
  certify 193 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3] (by decide)

def p_197 : CertifiedPrime :=
  certify 197 (2) [p_2, p_2, p_7, p_7] (by decide)

def p_199 : CertifiedPrime :=
  certify 199 (3) [p_2, p_3, p_3, p_11] (by decide)

def p_211 : CertifiedPrime :=
  certify 211 (2) [p_2, p_3, p_5, p_7] (by decide)

def p_223 : CertifiedPrime :=
  certify 223 (3) [p_2, p_3, p_37] (by decide)

def p_227 : CertifiedPrime :=
  certify 227 (2) [p_2, p_113] (by decide)

def p_229 : CertifiedPrime :=
  certify 229 (6) [p_2, p_2, p_3, p_19] (by decide)

def p_233 : CertifiedPrime :=
  certify 233 (3) [p_2, p_2, p_2, p_29] (by decide)

def p_239 : CertifiedPrime :=
  certify 239 (7) [p_2, p_7, p_17] (by decide)

def p_241 : CertifiedPrime :=
  certify 241 (7) [p_2, p_2, p_2, p_2, p_3, p_5] (by decide)

def p_251 : CertifiedPrime :=
  certify 251 (6) [p_2, p_5, p_5, p_5] (by decide)

def p_257 : CertifiedPrime :=
  certify 257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2] (by decide)

def p_263 : CertifiedPrime :=
  certify 263 (5) [p_2, p_131] (by decide)

def p_269 : CertifiedPrime :=
  certify 269 (2) [p_2, p_2, p_67] (by decide)

def p_271 : CertifiedPrime :=
  certify 271 (6) [p_2, p_3, p_3, p_3, p_5] (by decide)

def p_277 : CertifiedPrime :=
  certify 277 (5) [p_2, p_2, p_3, p_23] (by decide)

def p_281 : CertifiedPrime :=
  certify 281 (3) [p_2, p_2, p_2, p_5, p_7] (by decide)

def p_283 : CertifiedPrime :=
  certify 283 (3) [p_2, p_3, p_47] (by decide)

def p_293 : CertifiedPrime :=
  certify 293 (2) [p_2, p_2, p_73] (by decide)

def p_307 : CertifiedPrime :=
  certify 307 (5) [p_2, p_3, p_3, p_17] (by decide)

def p_311 : CertifiedPrime :=
  certify 311 (17) [p_2, p_5, p_31] (by decide)

def p_313 : CertifiedPrime :=
  certify 313 (10) [p_2, p_2, p_2, p_3, p_13] (by decide)

def p_317 : CertifiedPrime :=
  certify 317 (2) [p_2, p_2, p_79] (by decide)

def p_331 : CertifiedPrime :=
  certify 331 (3) [p_2, p_3, p_5, p_11] (by decide)

def p_337 : CertifiedPrime :=
  certify 337 (10) [p_2, p_2, p_2, p_2, p_3, p_7] (by decide)

def p_347 : CertifiedPrime :=
  certify 347 (2) [p_2, p_173] (by decide)

def p_349 : CertifiedPrime :=
  certify 349 (2) [p_2, p_2, p_3, p_29] (by decide)

def p_353 : CertifiedPrime :=
  certify 353 (3) [p_2, p_2, p_2, p_2, p_2, p_11] (by decide)

def p_359 : CertifiedPrime :=
  certify 359 (7) [p_2, p_179] (by decide)

def p_367 : CertifiedPrime :=
  certify 367 (6) [p_2, p_3, p_61] (by decide)

def p_373 : CertifiedPrime :=
  certify 373 (2) [p_2, p_2, p_3, p_31] (by decide)

def p_379 : CertifiedPrime :=
  certify 379 (2) [p_2, p_3, p_3, p_3, p_7] (by decide)

def p_383 : CertifiedPrime :=
  certify 383 (5) [p_2, p_191] (by decide)

def p_389 : CertifiedPrime :=
  certify 389 (2) [p_2, p_2, p_97] (by decide)

def p_397 : CertifiedPrime :=
  certify 397 (5) [p_2, p_2, p_3, p_3, p_11] (by decide)

def p_401 : CertifiedPrime :=
  certify 401 (3) [p_2, p_2, p_2, p_2, p_5, p_5] (by decide)

def p_409 : CertifiedPrime :=
  certify 409 (21) [p_2, p_2, p_2, p_3, p_17] (by decide)

def p_419 : CertifiedPrime :=
  certify 419 (2) [p_2, p_11, p_19] (by decide)

def p_421 : CertifiedPrime :=
  certify 421 (2) [p_2, p_2, p_3, p_5, p_7] (by decide)

def p_431 : CertifiedPrime :=
  certify 431 (7) [p_2, p_5, p_43] (by decide)

def p_433 : CertifiedPrime :=
  certify 433 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3] (by decide)

def p_439 : CertifiedPrime :=
  certify 439 (15) [p_2, p_3, p_73] (by decide)

def p_443 : CertifiedPrime :=
  certify 443 (2) [p_2, p_13, p_17] (by decide)

def p_449 : CertifiedPrime :=
  certify 449 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7] (by decide)

def p_457 : CertifiedPrime :=
  certify 457 (13) [p_2, p_2, p_2, p_3, p_19] (by decide)

def p_461 : CertifiedPrime :=
  certify 461 (2) [p_2, p_2, p_5, p_23] (by decide)

def p_463 : CertifiedPrime :=
  certify 463 (3) [p_2, p_3, p_7, p_11] (by decide)

def p_467 : CertifiedPrime :=
  certify 467 (2) [p_2, p_233] (by decide)

def p_479 : CertifiedPrime :=
  certify 479 (13) [p_2, p_239] (by decide)

def p_487 : CertifiedPrime :=
  certify 487 (3) [p_2, p_3, p_3, p_3, p_3, p_3] (by decide)

def p_491 : CertifiedPrime :=
  certify 491 (2) [p_2, p_5, p_7, p_7] (by decide)

def p_499 : CertifiedPrime :=
  certify 499 (7) [p_2, p_3, p_83] (by decide)

def p_503 : CertifiedPrime :=
  certify 503 (5) [p_2, p_251] (by decide)

def p_509 : CertifiedPrime :=
  certify 509 (2) [p_2, p_2, p_127] (by decide)

def p_521 : CertifiedPrime :=
  certify 521 (3) [p_2, p_2, p_2, p_5, p_13] (by decide)

def p_523 : CertifiedPrime :=
  certify 523 (2) [p_2, p_3, p_3, p_29] (by decide)

def p_541 : CertifiedPrime :=
  certify 541 (2) [p_2, p_2, p_3, p_3, p_3, p_5] (by decide)

def p_547 : CertifiedPrime :=
  certify 547 (2) [p_2, p_3, p_7, p_13] (by decide)

def p_557 : CertifiedPrime :=
  certify 557 (2) [p_2, p_2, p_139] (by decide)

def p_563 : CertifiedPrime :=
  certify 563 (2) [p_2, p_281] (by decide)

def p_569 : CertifiedPrime :=
  certify 569 (3) [p_2, p_2, p_2, p_71] (by decide)

def p_571 : CertifiedPrime :=
  certify 571 (3) [p_2, p_3, p_5, p_19] (by decide)

def p_577 : CertifiedPrime :=
  certify 577 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3] (by decide)

def p_587 : CertifiedPrime :=
  certify 587 (2) [p_2, p_293] (by decide)

def p_593 : CertifiedPrime :=
  certify 593 (3) [p_2, p_2, p_2, p_2, p_37] (by decide)

def p_599 : CertifiedPrime :=
  certify 599 (7) [p_2, p_13, p_23] (by decide)

def p_601 : CertifiedPrime :=
  certify 601 (7) [p_2, p_2, p_2, p_3, p_5, p_5] (by decide)

def p_607 : CertifiedPrime :=
  certify 607 (3) [p_2, p_3, p_101] (by decide)

def p_613 : CertifiedPrime :=
  certify 613 (2) [p_2, p_2, p_3, p_3, p_17] (by decide)

def p_617 : CertifiedPrime :=
  certify 617 (3) [p_2, p_2, p_2, p_7, p_11] (by decide)

def p_619 : CertifiedPrime :=
  certify 619 (2) [p_2, p_3, p_103] (by decide)

def p_631 : CertifiedPrime :=
  certify 631 (3) [p_2, p_3, p_3, p_5, p_7] (by decide)

def p_641 : CertifiedPrime :=
  certify 641 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5] (by decide)

def p_643 : CertifiedPrime :=
  certify 643 (11) [p_2, p_3, p_107] (by decide)

def p_647 : CertifiedPrime :=
  certify 647 (5) [p_2, p_17, p_19] (by decide)

def p_653 : CertifiedPrime :=
  certify 653 (2) [p_2, p_2, p_163] (by decide)

def p_659 : CertifiedPrime :=
  certify 659 (2) [p_2, p_7, p_47] (by decide)

def p_661 : CertifiedPrime :=
  certify 661 (2) [p_2, p_2, p_3, p_5, p_11] (by decide)

def p_673 : CertifiedPrime :=
  certify 673 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7] (by decide)

def p_677 : CertifiedPrime :=
  certify 677 (2) [p_2, p_2, p_13, p_13] (by decide)

def p_683 : CertifiedPrime :=
  certify 683 (5) [p_2, p_11, p_31] (by decide)

def p_691 : CertifiedPrime :=
  certify 691 (3) [p_2, p_3, p_5, p_23] (by decide)

def p_701 : CertifiedPrime :=
  certify 701 (2) [p_2, p_2, p_5, p_5, p_7] (by decide)

def p_709 : CertifiedPrime :=
  certify 709 (2) [p_2, p_2, p_3, p_59] (by decide)

def p_719 : CertifiedPrime :=
  certify 719 (11) [p_2, p_359] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
