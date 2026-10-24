import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk034

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_232745537 : CertifiedPrime :=
  certify 232745537 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1907, p_1907] (by decide)

def p_232867601 : CertifiedPrime :=
  certify 232867601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_109, p_109] (by decide)

def p_232882861 : CertifiedPrime :=
  certify 232882861 (2) [p_2, p_2, p_3, p_5, p_7, p_109, p_5087] (by decide)

def p_232989697 : CertifiedPrime :=
  certify 232989697 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_53, p_53] (by decide)

def p_233066023 : CertifiedPrime :=
  certify 233066023 (3) [p_2, p_3, p_7, p_17, p_449, p_727] (by decide)

def p_233249257 : CertifiedPrime :=
  certify 233249257 (5) [p_2, p_2, p_2, p_3, p_3, p_23, p_83, p_1697] (by decide)

def p_233279803 : CertifiedPrime :=
  certify 233279803 (2) [p_2, p_3, p_3, p_7, p_1091, p_1697] (by decide)

def p_233463121 : CertifiedPrime :=
  certify 233463121 (14) [p_2, p_2, p_2, p_2, p_3, p_5, p_11, p_191, p_463] (by decide)

def p_233524243 : CertifiedPrime :=
  certify 233524243 (7) [p_2, p_3, p_3, p_3, p_7, p_37, p_59, p_283] (by decide)

def p_233554807 : CertifiedPrime :=
  certify 233554807 (10) [p_2, p_3, p_3, p_3, p_17, p_29, p_31, p_283] (by decide)

def p_233982913 : CertifiedPrime :=
  certify 233982913 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_239, p_5099] (by decide)

def p_234380791 : CertifiedPrime :=
  certify 234380791 (6) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_7, p_1531] (by decide)

def p_234396101 : CertifiedPrime :=
  certify 234396101 (2) [p_2, p_2, p_5, p_5, p_1531, p_1531] (by decide)

def p_234518597 : CertifiedPrime :=
  certify 234518597 (2) [p_2, p_2, p_13, p_13, p_19, p_19, p_31, p_31] (by decide)

def p_234564541 : CertifiedPrime :=
  certify 234564541 (2) [p_2, p_2, p_3, p_5, p_7, p_547, p_1021] (by decide)

def p_234702401 : CertifiedPrime :=
  certify 234702401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_383, p_383] (by decide)

def p_234886277 : CertifiedPrime :=
  certify 234886277 (2) [p_2, p_2, p_79, p_79, p_97, p_97] (by decide)

def p_235116223 : CertifiedPrime :=
  certify 235116223 (10) [p_2, p_3, p_11, p_17, p_19, p_41, p_269] (by decide)

def p_235499717 : CertifiedPrime :=
  certify 235499717 (2) [p_2, p_2, p_7673, p_7673] (by decide)

def p_235760671 : CertifiedPrime :=
  certify 235760671 (15) [p_2, p_3, p_3, p_5, p_37, p_83, p_853] (by decide)

def p_236190793 : CertifiedPrime :=
  certify 236190793 (5) [p_2, p_2, p_2, p_3, p_17, p_47, p_109, p_113] (by decide)

def p_236236901 : CertifiedPrime :=
  certify 236236901 (2) [p_2, p_2, p_5, p_5, p_29, p_29, p_53, p_53] (by decide)

def p_236283013 : CertifiedPrime :=
  certify 236283013 (13) [p_2, p_2, p_3, p_3, p_7, p_19, p_61, p_809] (by decide)

def p_236682841 : CertifiedPrime :=
  certify 236682841 (21) [p_2, p_2, p_2, p_3, p_5, p_17, p_181, p_641] (by decide)

def p_236929057 : CertifiedPrime :=
  certify 236929057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_37, p_733] (by decide)

def p_236975237 : CertifiedPrime :=
  certify 236975237 (2) [p_2, p_2, p_43, p_43, p_179, p_179] (by decide)

def p_237144601 : CertifiedPrime :=
  certify 237144601 (13) [p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_11, p_29, p_59] (by decide)

def p_237329431 : CertifiedPrime :=
  certify 237329431 (12) [p_2, p_3, p_5, p_13, p_79, p_7703] (by decide)

def p_237391057 : CertifiedPrime :=
  certify 237391057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_31, p_71, p_107] (by decide)

def p_237575983 : CertifiedPrime :=
  certify 237575983 (3) [p_2, p_3, p_7, p_367, p_15413] (by decide)

def p_237853507 : CertifiedPrime :=
  certify 237853507 (2) [p_2, p_3, p_11, p_53, p_97, p_701] (by decide)

def p_238131193 : CertifiedPrime :=
  certify 238131193 (7) [p_2, p_2, p_2, p_3, p_13, p_643, p_1187] (by decide)

def p_238347283 : CertifiedPrime :=
  certify 238347283 (3) [p_2, p_3, p_31, p_83, p_15439] (by decide)

def p_238717951 : CertifiedPrime :=
  certify 238717951 (3) [p_2, p_3, p_5, p_5, p_103, p_15451] (by decide)

def p_238779757 : CertifiedPrime :=
  certify 238779757 (2) [p_2, p_2, p_3, p_3, p_17, p_101, p_3863] (by decide)

def p_238903393 : CertifiedPrime :=
  certify 238903393 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_23, p_29, p_41] (by decide)

def p_238996141 : CertifiedPrime :=
  certify 238996141 (2) [p_2, p_2, p_3, p_5, p_773, p_5153] (by decide)

def p_239135297 : CertifiedPrime :=
  certify 239135297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1933, p_1933] (by decide)

def p_239274493 : CertifiedPrime :=
  certify 239274493 (6) [p_2, p_2, p_3, p_31, p_499, p_1289] (by decide)

def p_239320901 : CertifiedPrime :=
  certify 239320901 (10) [p_2, p_2, p_5, p_5, p_7, p_7, p_13, p_13, p_17, p_17] (by decide)

def p_239506577 : CertifiedPrime :=
  certify 239506577 (3) [p_2, p_2, p_2, p_2, p_53, p_53, p_73, p_73] (by decide)

def p_239630401 : CertifiedPrime :=
  certify 239630401 (21) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_43, p_43] (by decide)

def p_239707807 : CertifiedPrime :=
  certify 239707807 (3) [p_2, p_3, p_13, p_397, p_7741] (by decide)

def p_239738773 : CertifiedPrime :=
  certify 239738773 (2) [p_2, p_2, p_3, p_7, p_7, p_13, p_79, p_397] (by decide)

def p_239800711 : CertifiedPrime :=
  certify 239800711 (6) [p_2, p_3, p_5, p_19, p_29, p_89, p_163] (by decide)

def p_239924611 : CertifiedPrime :=
  certify 239924611 (2) [p_2, p_3, p_3, p_5, p_1549, p_1721] (by decide)

def p_240203503 : CertifiedPrime :=
  certify 240203503 (11) [p_2, p_3, p_3, p_3, p_7, p_11, p_41, p_1409] (by decide)

def p_240296503 : CertifiedPrime :=
  certify 240296503 (3) [p_2, p_3, p_23, p_337, p_5167] (by decide)

def p_240389521 : CertifiedPrime :=
  certify 240389521 (23) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_17, p_19, p_443] (by decide)

def p_240560101 : CertifiedPrime :=
  certify 240560101 (10) [p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_11, p_47, p_47] (by decide)

def p_240575611 : CertifiedPrime :=
  certify 240575611 (3) [p_2, p_3, p_5, p_11, p_47, p_15511] (by decide)

def p_240746257 : CertifiedPrime :=
  certify 240746257 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_431, p_431] (by decide)

def p_240854881 : CertifiedPrime :=
  certify 240854881 (19) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_97, p_739] (by decide)

def p_240948007 : CertifiedPrime :=
  certify 240948007 (3) [p_2, p_3, p_13, p_19, p_19, p_43, p_199] (by decide)

def p_240994577 : CertifiedPrime :=
  certify 240994577 (3) [p_2, p_2, p_2, p_2, p_3881, p_3881] (by decide)

def p_241103257 : CertifiedPrime :=
  certify 241103257 (5) [p_2, p_2, p_2, p_3, p_647, p_15527] (by decide)

def p_241289623 : CertifiedPrime :=
  certify 241289623 (5) [p_2, p_3, p_3, p_7, p_7, p_317, p_863] (by decide)

def p_241367297 : CertifiedPrime :=
  certify 241367297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_971, p_971] (by decide)

def p_241693663 : CertifiedPrime :=
  certify 241693663 (3) [p_2, p_3, p_7, p_2221, p_2591] (by decide)

def p_241802501 : CertifiedPrime :=
  certify 241802501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_311, p_311] (by decide)

def p_241880257 : CertifiedPrime :=
  certify 241880257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_103, p_151] (by decide)

def p_242113601 : CertifiedPrime :=
  certify 242113601 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_389, p_389] (by decide)

def p_242129161 : CertifiedPrime :=
  certify 242129161 (17) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_19, p_389] (by decide)

def p_242409331 : CertifiedPrime :=
  certify 242409331 (13) [p_2, p_3, p_3, p_5, p_173, p_15569] (by decide)

def p_242424901 : CertifiedPrime :=
  certify 242424901 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_173, p_173] (by decide)

def p_242596201 : CertifiedPrime :=
  certify 242596201 (79) [p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_11, p_59, p_89] (by decide)

def p_242736401 : CertifiedPrime :=
  certify 242736401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_19, p_19, p_41, p_41] (by decide)

def p_242861057 : CertifiedPrime :=
  certify 242861057 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_487, p_487] (by decide)

def p_242923397 : CertifiedPrime :=
  certify 242923397 (2) [p_2, p_2, p_7793, p_7793] (by decide)

def p_243906307 : CertifiedPrime :=
  certify 243906307 (3) [p_2, p_3, p_7, p_19, p_23, p_97, p_137] (by decide)

def p_244093753 : CertifiedPrime :=
  certify 244093753 (5) [p_2, p_2, p_2, p_3, p_3, p_7, p_17, p_31, p_919] (by decide)

def p_244109377 : CertifiedPrime :=
  certify 244109377 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_31, p_31] (by decide)

def p_244218757 : CertifiedPrime :=
  certify 244218757 (2) [p_2, p_2, p_3, p_3907, p_5209] (by decide)

def p_244500133 : CertifiedPrime :=
  certify 244500133 (5) [p_2, p_2, p_3, p_19, p_823, p_1303] (by decide)

def p_244562683 : CertifiedPrime :=
  certify 244562683 (2) [p_2, p_3, p_7, p_13, p_401, p_1117] (by decide)

def p_244609601 : CertifiedPrime :=
  certify 244609601 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_23, p_23] (by decide)

def p_244656523 : CertifiedPrime :=
  certify 244656523 (14) [p_2, p_3, p_3, p_11, p_79, p_15641] (by decide)

def p_244797317 : CertifiedPrime :=
  certify 244797317 (2) [p_2, p_2, p_7823, p_7823] (by decide)

def p_244969453 : CertifiedPrime :=
  certify 244969453 (5) [p_2, p_2, p_3, p_3, p_7, p_13, p_37, p_43, p_47] (by decide)

def p_245032063 : CertifiedPrime :=
  certify 245032063 (3) [p_2, p_3, p_11, p_1423, p_2609] (by decide)

def p_245047717 : CertifiedPrime :=
  certify 245047717 (6) [p_2, p_2, p_3, p_3, p_2609, p_2609] (by decide)

def p_245110337 : CertifiedPrime :=
  certify 245110337 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_103, p_103] (by decide)

def p_245157307 : CertifiedPrime :=
  certify 245157307 (2) [p_2, p_3, p_17, p_307, p_7829] (by decide)

def p_245219941 : CertifiedPrime :=
  certify 245219941 (14) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_29, p_2237] (by decide)

def p_245360897 : CertifiedPrime :=
  certify 245360897 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_89, p_89] (by decide)

def p_245533231 : CertifiedPrime :=
  certify 245533231 (6) [p_2, p_3, p_3, p_5, p_1567, p_1741] (by decide)

def p_245674277 : CertifiedPrime :=
  certify 245674277 (2) [p_2, p_2, p_17, p_17, p_461, p_461] (by decide)

def p_245736977 : CertifiedPrime :=
  certify 245736977 (3) [p_2, p_2, p_2, p_2, p_3919, p_3919] (by decide)

def p_245862401 : CertifiedPrime :=
  certify 245862401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_7, p_7] (by decide)

def p_246254557 : CertifiedPrime :=
  certify 246254557 (2) [p_2, p_2, p_3, p_3923, p_5231] (by decide)

def p_246285943 : CertifiedPrime :=
  certify 246285943 (3) [p_2, p_3, p_7, p_19, p_59, p_5231] (by decide)

def p_246348721 : CertifiedPrime :=
  certify 246348721 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_43, p_73, p_109] (by decide)

def p_246474301 : CertifiedPrime :=
  certify 246474301 (2) [p_2, p_2, p_3, p_5, p_5, p_157, p_5233] (by decide)

def p_246756973 : CertifiedPrime :=
  certify 246756973 (6) [p_2, p_2, p_3, p_7, p_11, p_17, p_23, p_683] (by decide)

def p_246804101 : CertifiedPrime :=
  certify 246804101 (2) [p_2, p_2, p_5, p_5, p_1571, p_1571] (by decide)

def p_246992657 : CertifiedPrime :=
  certify 246992657 (3) [p_2, p_2, p_2, p_2, p_3929, p_3929] (by decide)

def p_247008373 : CertifiedPrime :=
  certify 247008373 (2) [p_2, p_2, p_3, p_13, p_13, p_31, p_3929] (by decide)

def p_247228453 : CertifiedPrime :=
  certify 247228453 (2) [p_2, p_2, p_3, p_3, p_1747, p_3931] (by decide)

def p_247307077 : CertifiedPrime :=
  certify 247307077 (2) [p_2, p_2, p_3, p_3, p_2621, p_2621] (by decide)

def p_247417171 : CertifiedPrime :=
  certify 247417171 (2) [p_2, p_3, p_5, p_7, p_7, p_11, p_11, p_13, p_107] (by decide)

def p_247668907 : CertifiedPrime :=
  certify 247668907 (5) [p_2, p_3, p_43, p_61, p_15737] (by decide)

def p_247747601 : CertifiedPrime :=
  certify 247747601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_787, p_787] (by decide)

def p_247873537 : CertifiedPrime :=
  certify 247873537 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_41, p_41] (by decide)

def p_248188517 : CertifiedPrime :=
  certify 248188517 (2) [p_2, p_2, p_7877, p_7877] (by decide)

def p_248267293 : CertifiedPrime :=
  certify 248267293 (6) [p_2, p_2, p_3, p_7, p_13, p_101, p_2251] (by decide)

def p_248740213 : CertifiedPrime :=
  certify 248740213 (2) [p_2, p_2, p_3, p_7, p_751, p_3943] (by decide)

def p_249118873 : CertifiedPrime :=
  certify 249118873 (10) [p_2, p_2, p_2, p_3, p_1973, p_5261] (by decide)

def p_249134657 : CertifiedPrime :=
  certify 249134657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1973, p_1973] (by decide)

def p_249513617 : CertifiedPrime :=
  certify 249513617 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_359, p_359] (by decide)

def p_249561007 : CertifiedPrime :=
  certify 249561007 (3) [p_2, p_3, p_2633, p_15797] (by decide)

def p_249766417 : CertifiedPrime :=
  certify 249766417 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_439, p_439] (by decide)

def p_250272401 : CertifiedPrime :=
  certify 250272401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_113, p_113] (by decide)

def p_250319863 : CertifiedPrime :=
  certify 250319863 (17) [p_2, p_3, p_3, p_3, p_13, p_293, p_1217] (by decide)

def p_250699723 : CertifiedPrime :=
  certify 250699723 (3) [p_2, p_3, p_7, p_13, p_29, p_71, p_223] (by decide)

def p_250984807 : CertifiedPrime :=
  certify 250984807 (3) [p_2, p_3, p_89, p_89, p_5281] (by decide)

def p_251032337 : CertifiedPrime :=
  certify 251032337 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_233, p_233] (by decide)

def p_251174953 : CertifiedPrime :=
  certify 251174953 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_7, p_283, p_587] (by decide)

def p_251270053 : CertifiedPrime :=
  certify 251270053 (6) [p_2, p_2, p_3, p_11, p_11, p_131, p_1321] (by decide)

def p_251365171 : CertifiedPrime :=
  certify 251365171 (2) [p_2, p_3, p_5, p_7, p_151, p_7927] (by decide)

def p_251412737 : CertifiedPrime :=
  certify 251412737 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_991, p_991] (by decide)

def p_251777557 : CertifiedPrime :=
  certify 251777557 (2) [p_2, p_2, p_3, p_3, p_41, p_43, p_3967] (by decide)

def p_251872771 : CertifiedPrime :=
  certify 251872771 (3) [p_2, p_3, p_5, p_23, p_23, p_59, p_269] (by decide)

def p_252031501 : CertifiedPrime :=
  certify 252031501 (10) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_7, p_7, p_127] (by decide)

def p_252222043 : CertifiedPrime :=
  certify 252222043 (3) [p_2, p_3, p_2647, p_15881] (by decide)

def p_252317341 : CertifiedPrime :=
  certify 252317341 (2) [p_2, p_2, p_3, p_3, p_5, p_11, p_19, p_19, p_353] (by decide)

def p_252492101 : CertifiedPrime :=
  certify 252492101 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_227, p_227] (by decide)

def p_252603343 : CertifiedPrime :=
  certify 252603343 (5) [p_2, p_3, p_3, p_23, p_691, p_883] (by decide)

def p_252682817 : CertifiedPrime :=
  certify 252682817 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1987, p_1987] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
