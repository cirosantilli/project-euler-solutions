import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk037

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_297752281 : CertifiedPrime :=
  certify 297752281 (11) [p_2, p_2, p_2, p_3, p_5, p_7, p_17, p_29, p_719] (by decide)

def p_297855823 : CertifiedPrime :=
  certify 297855823 (3) [p_2, p_3, p_11, p_523, p_8629] (by decide)

def p_297993907 : CertifiedPrime :=
  certify 297993907 (5) [p_2, p_3, p_3, p_7, p_61, p_137, p_283] (by decide)

def p_298045697 : CertifiedPrime :=
  certify 298045697 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_83, p_83] (by decide)

def p_298097491 : CertifiedPrime :=
  certify 298097491 (2) [p_2, p_3, p_5, p_89, p_97, p_1151] (by decide)

def p_298892233 : CertifiedPrime :=
  certify 298892233 (5) [p_2, p_2, p_2, p_3, p_3, p_17, p_113, p_2161] (by decide)

def p_299082437 : CertifiedPrime :=
  certify 299082437 (2) [p_2, p_2, p_8647, p_8647] (by decide)

def p_299134321 : CertifiedPrime :=
  certify 299134321 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_23, p_47, p_1153] (by decide)

def p_299238103 : CertifiedPrime :=
  certify 299238103 (13) [p_2, p_3, p_3, p_31, p_31, p_17299] (by decide)

def p_299445721 : CertifiedPrime :=
  certify 299445721 (38) [p_2, p_2, p_2, p_3, p_5, p_7, p_103, p_3461] (by decide)

def p_299497637 : CertifiedPrime :=
  certify 299497637 (2) [p_2, p_2, p_17, p_17, p_509, p_509] (by decide)

def p_299843857 : CertifiedPrime :=
  certify 299843857 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13, p_37, p_37] (by decide)

def p_299930443 : CertifiedPrime :=
  certify 299930443 (2) [p_2, p_3, p_7, p_23, p_251, p_1237] (by decide)

def p_299982401 : CertifiedPrime :=
  certify 299982401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_433, p_433] (by decide)

def p_300190277 : CertifiedPrime :=
  certify 300190277 (2) [p_2, p_2, p_8663, p_8663] (by decide)

def p_300276913 : CertifiedPrime :=
  certify 300276913 (11) [p_2, p_2, p_2, p_2, p_3, p_13, p_19, p_19, p_31, p_43] (by decide)

def p_300380893 : CertifiedPrime :=
  certify 300380893 (2) [p_2, p_2, p_3, p_7, p_53, p_109, p_619] (by decide)

def p_300450223 : CertifiedPrime :=
  certify 300450223 (5) [p_2, p_3, p_3, p_3, p_3, p_107, p_17333] (by decide)

def p_300883717 : CertifiedPrime :=
  certify 300883717 (6) [p_2, p_2, p_3, p_3, p_7, p_7, p_7, p_7, p_59, p_59] (by decide)

def p_300901063 : CertifiedPrime :=
  certify 300901063 (3) [p_2, p_3, p_7, p_7, p_11, p_19, p_59, p_83] (by decide)

def p_301005151 : CertifiedPrime :=
  certify 301005151 (3) [p_2, p_3, p_5, p_5, p_347, p_5783] (by decide)

def p_301230737 : CertifiedPrime :=
  certify 301230737 (3) [p_2, p_2, p_2, p_2, p_4339, p_4339] (by decide)

def p_301421683 : CertifiedPrime :=
  certify 301421683 (2) [p_2, p_3, p_3, p_3, p_643, p_8681] (by decide)

def p_301577957 : CertifiedPrime :=
  certify 301577957 (2) [p_2, p_2, p_19, p_19, p_457, p_457] (by decide)

def p_301734271 : CertifiedPrime :=
  certify 301734271 (11) [p_2, p_3, p_3, p_5, p_29, p_193, p_599] (by decide)

def p_301908001 : CertifiedPrime :=
  certify 301908001 (11) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_5, p_139, p_181] (by decide)

def p_302116543 : CertifiedPrime :=
  certify 302116543 (3) [p_2, p_3, p_7, p_13, p_191, p_2897] (by decide)

def p_302533843 : CertifiedPrime :=
  certify 302533843 (3) [p_2, p_3, p_13, p_223, p_17393] (by decide)

def p_302551237 : CertifiedPrime :=
  certify 302551237 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_223, p_223] (by decide)

def p_302968837 : CertifiedPrime :=
  certify 302968837 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_967, p_967] (by decide)

def p_303090691 : CertifiedPrime :=
  certify 303090691 (2) [p_2, p_3, p_5, p_7, p_829, p_1741] (by decide)

def p_303595777 : CertifiedPrime :=
  certify 303595777 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_11, p_11, p_11, p_11] (by decide)

def p_303665477 : CertifiedPrime :=
  certify 303665477 (2) [p_2, p_2, p_8713, p_8713] (by decide)

def p_304502501 : CertifiedPrime :=
  certify 304502501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_349, p_349] (by decide)

def p_304519951 : CertifiedPrime :=
  certify 304519951 (6) [p_2, p_3, p_3, p_5, p_5, p_7, p_277, p_349] (by decide)

def p_304729393 : CertifiedPrime :=
  certify 304729393 (10) [p_2, p_2, p_2, p_2, p_3, p_11, p_23, p_23, p_1091] (by decide)

def p_305043691 : CertifiedPrime :=
  certify 305043691 (3) [p_2, p_3, p_5, p_7, p_41, p_71, p_499] (by decide)

def p_305061157 : CertifiedPrime :=
  certify 305061157 (5) [p_2, p_2, p_3, p_3, p_41, p_41, p_71, p_71] (by decide)

def p_305463007 : CertifiedPrime :=
  certify 305463007 (5) [p_2, p_3, p_3, p_971, p_17477] (by decide)

def p_305707741 : CertifiedPrime :=
  certify 305707741 (6) [p_2, p_2, p_3, p_5, p_13, p_31, p_47, p_269] (by decide)

def p_305777683 : CertifiedPrime :=
  certify 305777683 (2) [p_2, p_3, p_3, p_7, p_29, p_67, p_1249] (by decide)

def p_306127513 : CertifiedPrime :=
  certify 306127513 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_17497] (by decide)

def p_306442531 : CertifiedPrime :=
  certify 306442531 (2) [p_2, p_3, p_3, p_5, p_389, p_8753] (by decide)

def p_306600101 : CertifiedPrime :=
  certify 306600101 (2) [p_2, p_2, p_5, p_5, p_17, p_17, p_103, p_103] (by decide)

def p_307090577 : CertifiedPrime :=
  certify 307090577 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_337, p_337] (by decide)

def p_307178203 : CertifiedPrime :=
  certify 307178203 (3) [p_2, p_3, p_17, p_23, p_127, p_1031] (by decide)

def p_307353493 : CertifiedPrime :=
  certify 307353493 (7) [p_2, p_2, p_3, p_3, p_47, p_373, p_487] (by decide)

def p_307704223 : CertifiedPrime :=
  certify 307704223 (6) [p_2, p_3, p_3, p_7, p_7, p_179, p_1949] (by decide)

def p_307774393 : CertifiedPrime :=
  certify 307774393 (5) [p_2, p_2, p_2, p_3, p_17, p_43, p_53, p_331] (by decide)

def p_307914757 : CertifiedPrime :=
  certify 307914757 (2) [p_2, p_2, p_3, p_41, p_107, p_5849] (by decide)

def p_308002501 : CertifiedPrime :=
  certify 308002501 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_13, p_13] (by decide)

def p_308090257 : CertifiedPrime :=
  certify 308090257 (5) [p_2, p_2, p_2, p_2, p_3, p_1097, p_5851] (by decide)

def p_308441407 : CertifiedPrime :=
  certify 308441407 (3) [p_2, p_3, p_7, p_13, p_193, p_2927] (by decide)

def p_308704901 : CertifiedPrime :=
  certify 308704901 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_251, p_251] (by decide)

def p_308722471 : CertifiedPrime :=
  certify 308722471 (3) [p_2, p_3, p_5, p_7, p_251, p_5857] (by decide)

def p_308827903 : CertifiedPrime :=
  certify 308827903 (3) [p_2, p_3, p_29, p_101, p_17573] (by decide)

def p_308933353 : CertifiedPrime :=
  certify 308933353 (10) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_13, p_13, p_13, p_31] (by decide)

def p_309390511 : CertifiedPrime :=
  certify 309390511 (3) [p_2, p_3, p_5, p_11, p_13, p_41, p_1759] (by decide)

def p_309408101 : CertifiedPrime :=
  certify 309408101 (2) [p_2, p_2, p_5, p_5, p_1759, p_1759] (by decide)

def p_309672007 : CertifiedPrime :=
  certify 309672007 (3) [p_2, p_3, p_7, p_419, p_17597] (by decide)

def p_309760001 : CertifiedPrime :=
  certify 309760001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_11, p_11] (by decide)

def p_309883213 : CertifiedPrime :=
  certify 309883213 (5) [p_2, p_2, p_3, p_3, p_3, p_29, p_163, p_607] (by decide)

def p_309900817 : CertifiedPrime :=
  certify 309900817 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_163, p_163] (by decide)

def p_309988843 : CertifiedPrime :=
  certify 309988843 (2) [p_2, p_3, p_5869, p_8803] (by decide)

def p_310252997 : CertifiedPrime :=
  certify 310252997 (2) [p_2, p_2, p_8807, p_8807] (by decide)

def p_310728757 : CertifiedPrime :=
  certify 310728757 (5) [p_2, p_2, p_3, p_13, p_113, p_17627] (by decide)

def p_310957957 : CertifiedPrime :=
  certify 310957957 (5) [p_2, p_2, p_3, p_3, p_2939, p_2939] (by decide)

def p_311663717 : CertifiedPrime :=
  certify 311663717 (2) [p_2, p_2, p_7, p_7, p_13, p_13, p_97, p_97] (by decide)

def p_311716681 : CertifiedPrime :=
  certify 311716681 (17) [p_2, p_2, p_2, p_3, p_5, p_11, p_107, p_2207] (by decide)

def p_312228901 : CertifiedPrime :=
  certify 312228901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_19, p_31, p_31] (by decide)

def p_312370277 : CertifiedPrime :=
  certify 312370277 (2) [p_2, p_2, p_8837, p_8837] (by decide)

def p_312529363 : CertifiedPrime :=
  certify 312529363 (2) [p_2, p_3, p_71, p_83, p_8839] (by decide)

def p_312794597 : CertifiedPrime :=
  certify 312794597 (2) [p_2, p_2, p_37, p_37, p_239, p_239] (by decide)

def p_312847657 : CertifiedPrime :=
  certify 312847657 (5) [p_2, p_2, p_2, p_3, p_11, p_23, p_67, p_769] (by decide)

def p_312936101 : CertifiedPrime :=
  certify 312936101 (2) [p_2, p_2, p_5, p_5, p_29, p_29, p_61, p_61] (by decide)

def p_312953791 : CertifiedPrime :=
  certify 312953791 (3) [p_2, p_3, p_5, p_29, p_61, p_5897] (by decide)

def p_313077637 : CertifiedPrime :=
  certify 313077637 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_983, p_983] (by decide)

def p_313095331 : CertifiedPrime :=
  certify 313095331 (15) [p_2, p_3, p_3, p_5, p_983, p_3539] (by decide)

def p_313148417 : CertifiedPrime :=
  certify 313148417 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_79, p_79] (by decide)

def p_313431617 : CertifiedPrime :=
  certify 313431617 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2213, p_2213] (by decide)

def p_313945243 : CertifiedPrime :=
  certify 313945243 (3) [p_2, p_3, p_13, p_29, p_47, p_2953] (by decide)

def p_313998401 : CertifiedPrime :=
  certify 313998401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_443, p_443] (by decide)

def p_314689861 : CertifiedPrime :=
  certify 314689861 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_73, p_887] (by decide)

def p_314867281 : CertifiedPrime :=
  certify 314867281 (37) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_13, p_13, p_1109] (by decide)

def p_315080251 : CertifiedPrime :=
  certify 315080251 (2) [p_2, p_3, p_5, p_5, p_5, p_61, p_71, p_97] (by decide)

def p_315399841 : CertifiedPrime :=
  certify 315399841 (17) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_37, p_43, p_59] (by decide)

def p_315559697 : CertifiedPrime :=
  certify 315559697 (3) [p_2, p_2, p_2, p_2, p_4441, p_4441] (by decide)

def p_315755131 : CertifiedPrime :=
  certify 315755131 (2) [p_2, p_3, p_5, p_1777, p_5923] (by decide)

def p_316181743 : CertifiedPrime :=
  certify 316181743 (3) [p_2, p_3, p_17, p_523, p_5927] (by decide)

def p_316501891 : CertifiedPrime :=
  certify 316501891 (3) [p_2, p_3, p_5, p_593, p_17791] (by decide)

def p_316697617 : CertifiedPrime :=
  certify 316697617 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1483, p_1483] (by decide)

def p_316893403 : CertifiedPrime :=
  certify 316893403 (7) [p_2, p_3, p_3, p_7, p_23, p_43, p_2543] (by decide)

def p_316929007 : CertifiedPrime :=
  certify 316929007 (5) [p_2, p_3, p_3, p_19, p_23, p_43, p_937] (by decide)

def p_316982417 : CertifiedPrime :=
  certify 316982417 (3) [p_2, p_2, p_2, p_2, p_4451, p_4451] (by decide)

def p_317196101 : CertifiedPrime :=
  certify 317196101 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_137, p_137] (by decide)

def p_317427673 : CertifiedPrime :=
  certify 317427673 (10) [p_2, p_2, p_2, p_3, p_17, p_131, p_5939] (by decide)

def p_317694977 : CertifiedPrime :=
  certify 317694977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_557, p_557] (by decide)

def p_317855413 : CertifiedPrime :=
  certify 317855413 (2) [p_2, p_2, p_3, p_3, p_7, p_283, p_4457] (by decide)

def p_317908901 : CertifiedPrime :=
  certify 317908901 (2) [p_2, p_2, p_5, p_5, p_1783, p_1783] (by decide)

def p_318176407 : CertifiedPrime :=
  certify 318176407 (5) [p_2, p_3, p_3, p_991, p_17837] (by decide)

def p_318497563 : CertifiedPrime :=
  certify 318497563 (2) [p_2, p_3, p_3, p_3, p_661, p_8923] (by decide)

def p_318622501 : CertifiedPrime :=
  certify 318622501 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_7, p_7, p_17, p_17] (by decide)

def p_318747463 : CertifiedPrime :=
  certify 318747463 (3) [p_2, p_3, p_11, p_79, p_113, p_541] (by decide)

def p_318836737 : CertifiedPrime :=
  certify 318836737 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_31, p_31] (by decide)

def p_318854593 : CertifiedPrime :=
  certify 318854593 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_31, p_2551] (by decide)

def p_318979601 : CertifiedPrime :=
  certify 318979601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_19, p_19, p_47, p_47] (by decide)

def p_319068907 : CertifiedPrime :=
  certify 319068907 (3) [p_2, p_3, p_13, p_229, p_17863] (by decide)

def p_319908997 : CertifiedPrime :=
  certify 319908997 (2) [p_2, p_2, p_3, p_3, p_11, p_11, p_271, p_271] (by decide)

def p_320052101 : CertifiedPrime :=
  certify 320052101 (2) [p_2, p_2, p_5, p_5, p_1789, p_1789] (by decide)

def p_320248921 : CertifiedPrime :=
  certify 320248921 (7) [p_2, p_2, p_2, p_3, p_5, p_1193, p_2237] (by decide)

def p_320356303 : CertifiedPrime :=
  certify 320356303 (3) [p_2, p_3, p_7, p_19, p_157, p_2557] (by decide)

def p_320427901 : CertifiedPrime :=
  certify 320427901 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_13, p_17, p_179] (by decide)

def p_320553217 : CertifiedPrime :=
  certify 320553217 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_373, p_373] (by decide)

def p_320642743 : CertifiedPrime :=
  certify 320642743 (6) [p_2, p_3, p_7, p_47, p_127, p_1279] (by decide)

def p_320893483 : CertifiedPrime :=
  certify 320893483 (2) [p_2, p_3, p_7, p_13, p_13, p_53, p_853] (by decide)

def p_321000973 : CertifiedPrime :=
  certify 321000973 (5) [p_2, p_2, p_3, p_19, p_23, p_41, p_1493] (by decide)

def p_321323551 : CertifiedPrime :=
  certify 321323551 (12) [p_2, p_3, p_5, p_5, p_239, p_8963] (by decide)

def p_321484901 : CertifiedPrime :=
  certify 321484901 (3) [p_2, p_2, p_5, p_5, p_11, p_11, p_163, p_163] (by decide)

def p_321610423 : CertifiedPrime :=
  certify 321610423 (3) [p_2, p_3, p_7, p_7, p_61, p_79, p_227] (by decide)

def p_321700097 : CertifiedPrime :=
  certify 321700097 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_59, p_59] (by decide)

def p_321843601 : CertifiedPrime :=
  certify 321843601 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13, p_23, p_23] (by decide)

def p_322058917 : CertifiedPrime :=
  certify 322058917 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_997, p_997] (by decide)

def p_322076863 : CertifiedPrime :=
  certify 322076863 (5) [p_2, p_3, p_3, p_131, p_137, p_997] (by decide)

def p_322148653 : CertifiedPrime :=
  certify 322148653 (2) [p_2, p_2, p_3, p_7, p_31, p_193, p_641] (by decide)

def p_322202501 : CertifiedPrime :=
  certify 322202501 (7) [p_2, p_2, p_5, p_5, p_5, p_5, p_359, p_359] (by decide)

def p_322346117 : CertifiedPrime :=
  certify 322346117 (2) [p_2, p_2, p_47, p_47, p_191, p_191] (by decide)

def p_322615483 : CertifiedPrime :=
  certify 322615483 (2) [p_2, p_3, p_7, p_1283, p_5987] (by decide)

def p_322687333 : CertifiedPrime :=
  certify 322687333 (5) [p_2, p_2, p_3, p_3, p_11, p_23, p_71, p_499] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
