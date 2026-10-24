import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk033

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_212416051 : CertifiedPrime :=
  certify 212416051 (17) [p_2, p_3, p_5, p_5, p_7, p_11, p_53, p_347] (by decide)

def p_212459777 : CertifiedPrime :=
  certify 212459777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_911, p_911] (by decide)

def p_212576401 : CertifiedPrime :=
  certify 212576401 (19) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_212765983 : CertifiedPrime :=
  certify 212765983 (5) [p_2, p_3, p_11, p_13, p_17, p_29, p_503] (by decide)

def p_212999431 : CertifiedPrime :=
  certify 212999431 (3) [p_2, p_3, p_5, p_7, p_139, p_7297] (by decide)

def p_213116203 : CertifiedPrime :=
  certify 213116203 (11) [p_2, p_3, p_3, p_13, p_811, p_1123] (by decide)

def p_213174601 : CertifiedPrime :=
  certify 213174601 (7) [p_2, p_2, p_2, p_3, p_5, p_5, p_31, p_73, p_157] (by decide)

def p_213203803 : CertifiedPrime :=
  certify 213203803 (2) [p_2, p_3, p_7, p_7, p_31, p_149, p_157] (by decide)

def p_213335237 : CertifiedPrime :=
  certify 213335237 (2) [p_2, p_2, p_67, p_67, p_109, p_109] (by decide)

def p_213379057 : CertifiedPrime :=
  certify 213379057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_11, p_83, p_541] (by decide)

def p_213642073 : CertifiedPrime :=
  certify 213642073 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_29, p_47, p_311] (by decide)

def p_214080793 : CertifiedPrime :=
  certify 214080793 (5) [p_2, p_2, p_2, p_3, p_31, p_59, p_4877] (by decide)

def p_214153957 : CertifiedPrime :=
  certify 214153957 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_271, p_271] (by decide)

def p_214212497 : CertifiedPrime :=
  certify 214212497 (3) [p_2, p_2, p_2, p_2, p_3659, p_3659] (by decide)

def p_214402807 : CertifiedPrime :=
  certify 214402807 (5) [p_2, p_3, p_3, p_1627, p_7321] (by decide)

def p_214607851 : CertifiedPrime :=
  certify 214607851 (2) [p_2, p_3, p_5, p_5, p_19, p_257, p_293] (by decide)

def p_214695757 : CertifiedPrime :=
  certify 214695757 (5) [p_2, p_2, p_3, p_3, p_11, p_37, p_14653] (by decide)

def p_214915601 : CertifiedPrime :=
  certify 214915601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_733, p_733] (by decide)

def p_215106223 : CertifiedPrime :=
  certify 215106223 (3) [p_2, p_3, p_4889, p_7333] (by decide)

def p_215326277 : CertifiedPrime :=
  certify 215326277 (2) [p_2, p_2, p_11, p_11, p_23, p_23, p_29, p_29] (by decide)

def p_215634541 : CertifiedPrime :=
  certify 215634541 (6) [p_2, p_2, p_3, p_5, p_11, p_89, p_3671] (by decide)

def p_215913637 : CertifiedPrime :=
  certify 215913637 (2) [p_2, p_2, p_3, p_3, p_31, p_31, p_79, p_79] (by decide)

def p_215987113 : CertifiedPrime :=
  certify 215987113 (7) [p_2, p_2, p_2, p_3, p_3, p_11, p_23, p_71, p_167] (by decide)

def p_216192913 : CertifiedPrime :=
  certify 216192913 (15) [p_2, p_2, p_2, p_2, p_3, p_13, p_13, p_29, p_919] (by decide)

def p_216251731 : CertifiedPrime :=
  certify 216251731 (10) [p_2, p_3, p_3, p_5, p_17, p_19, p_43, p_173] (by decide)

def p_216501797 : CertifiedPrime :=
  certify 216501797 (2) [p_2, p_2, p_7, p_7, p_1051, p_1051] (by decide)

def p_216560657 : CertifiedPrime :=
  certify 216560657 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_283, p_283] (by decide)

def p_216810901 : CertifiedPrime :=
  certify 216810901 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_31, p_409] (by decide)

def p_216869803 : CertifiedPrime :=
  certify 216869803 (2) [p_2, p_3, p_37, p_199, p_4909] (by decide)

def p_216899257 : CertifiedPrime :=
  certify 216899257 (5) [p_2, p_2, p_2, p_3, p_7, p_263, p_4909] (by decide)

def p_217076023 : CertifiedPrime :=
  certify 217076023 (6) [p_2, p_3, p_3, p_53, p_139, p_1637] (by decide)

def p_217134961 : CertifiedPrime :=
  certify 217134961 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_307, p_421] (by decide)

def p_217444517 : CertifiedPrime :=
  certify 217444517 (2) [p_2, p_2, p_73, p_73, p_101, p_101] (by decide)

def p_217931407 : CertifiedPrime :=
  certify 217931407 (3) [p_2, p_3, p_7, p_11, p_11, p_19, p_37, p_61] (by decide)

def p_218049523 : CertifiedPrime :=
  certify 218049523 (3) [p_2, p_3, p_23, p_107, p_14767] (by decide)

def p_218108593 : CertifiedPrime :=
  certify 218108593 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_13, p_71, p_547] (by decide)

def p_218271077 : CertifiedPrime :=
  certify 218271077 (2) [p_2, p_2, p_83, p_83, p_89, p_89] (by decide)

def p_218744101 : CertifiedPrime :=
  certify 218744101 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_17, p_17, p_29, p_29] (by decide)

def p_219025201 : CertifiedPrime :=
  certify 219025201 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_37, p_4933] (by decide)

def p_219173221 : CertifiedPrime :=
  certify 219173221 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_47, p_3701] (by decide)

def p_219454597 : CertifiedPrime :=
  certify 219454597 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_823, p_823] (by decide)

def p_219558307 : CertifiedPrime :=
  certify 219558307 (2) [p_2, p_3, p_11, p_31, p_239, p_449] (by decide)

def p_219647221 : CertifiedPrime :=
  certify 219647221 (7) [p_2, p_2, p_3, p_5, p_13, p_19, p_14821] (by decide)

def p_219706507 : CertifiedPrime :=
  certify 219706507 (2) [p_2, p_3, p_3, p_3, p_3, p_3, p_61, p_7411] (by decide)

def p_219750977 : CertifiedPrime :=
  certify 219750977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_17, p_17, p_109, p_109] (by decide)

def p_219810277 : CertifiedPrime :=
  certify 219810277 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_353, p_353] (by decide)

def p_220062391 : CertifiedPrime :=
  certify 220062391 (3) [p_2, p_3, p_5, p_23, p_43, p_7417] (by decide)

def p_220240441 : CertifiedPrime :=
  certify 220240441 (26) [p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_17, p_53, p_97] (by decide)

def p_220522501 : CertifiedPrime :=
  certify 220522501 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_11, p_11] (by decide)

def p_220596757 : CertifiedPrime :=
  certify 220596757 (5) [p_2, p_2, p_3, p_47, p_79, p_4951] (by decide)

def p_220626463 : CertifiedPrime :=
  certify 220626463 (3) [p_2, p_3, p_7, p_1061, p_4951] (by decide)

def p_220864183 : CertifiedPrime :=
  certify 220864183 (3) [p_2, p_3, p_7, p_11, p_193, p_2477] (by decide)

def p_221042557 : CertifiedPrime :=
  certify 221042557 (5) [p_2, p_2, p_3, p_3, p_7, p_59, p_14867] (by decide)

def p_221429281 : CertifiedPrime :=
  certify 221429281 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_23, p_31, p_647] (by decide)

def p_221488807 : CertifiedPrime :=
  certify 221488807 (5) [p_2, p_3, p_7, p_11, p_11, p_41, p_1063] (by decide)

def p_221592997 : CertifiedPrime :=
  certify 221592997 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_827, p_827] (by decide)

def p_221697211 : CertifiedPrime :=
  certify 221697211 (2) [p_2, p_3, p_5, p_7, p_709, p_1489] (by decide)

def p_221786557 : CertifiedPrime :=
  certify 221786557 (5) [p_2, p_2, p_3, p_17, p_53, p_73, p_281] (by decide)

def p_221875921 : CertifiedPrime :=
  certify 221875921 (26) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_7, p_19, p_331] (by decide)

def p_222144121 : CertifiedPrime :=
  certify 222144121 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_11, p_23, p_271] (by decide)

def p_222323011 : CertifiedPrime :=
  certify 222323011 (3) [p_2, p_3, p_5, p_7, p_13, p_31, p_37, p_71] (by decide)

def p_222427397 : CertifiedPrime :=
  certify 222427397 (2) [p_2, p_2, p_7457, p_7457] (by decide)

def p_222561643 : CertifiedPrime :=
  certify 222561643 (2) [p_2, p_3, p_4973, p_7459] (by decide)

def p_222785477 : CertifiedPrime :=
  certify 222785477 (2) [p_2, p_2, p_17, p_17, p_439, p_439] (by decide)

def p_223039291 : CertifiedPrime :=
  certify 223039291 (12) [p_2, p_3, p_5, p_19, p_29, p_103, p_131] (by decide)

def p_223084097 : CertifiedPrime :=
  certify 223084097 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1867, p_1867] (by decide)

def p_223188661 : CertifiedPrime :=
  certify 223188661 (19) [p_2, p_2, p_3, p_3, p_5, p_83, p_14939] (by decide)

def p_223203601 : CertifiedPrime :=
  certify 223203601 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_83, p_83] (by decide)

def p_223308193 : CertifiedPrime :=
  certify 223308193 (10) [p_2, p_2, p_2, p_2, p_2, p_3, p_17, p_293, p_467] (by decide)

def p_223502501 : CertifiedPrime :=
  certify 223502501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_13, p_13, p_23, p_23] (by decide)

def p_223547353 : CertifiedPrime :=
  certify 223547353 (11) [p_2, p_2, p_2, p_3, p_7, p_89, p_14951] (by decide)

def p_223577257 : CertifiedPrime :=
  certify 223577257 (5) [p_2, p_2, p_2, p_3, p_7, p_19, p_89, p_787] (by decide)

def p_224295553 : CertifiedPrime :=
  certify 224295553 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_17, p_881] (by decide)

def p_224625157 : CertifiedPrime :=
  certify 224625157 (5) [p_2, p_2, p_3, p_7, p_1249, p_2141] (by decide)

def p_224715091 : CertifiedPrime :=
  certify 224715091 (2) [p_2, p_3, p_5, p_19, p_263, p_1499] (by decide)

def p_224805043 : CertifiedPrime :=
  certify 224805043 (5) [p_2, p_3, p_3, p_7, p_7, p_11, p_17, p_29, p_47] (by decide)

def p_224880017 : CertifiedPrime :=
  certify 224880017 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_163, p_163] (by decide)

def p_224925007 : CertifiedPrime :=
  certify 224925007 (3) [p_2, p_3, p_4999, p_7499] (by decide)

def p_225195043 : CertifiedPrime :=
  certify 225195043 (3) [p_2, p_3, p_41, p_43, p_61, p_349] (by decide)

def p_225465241 : CertifiedPrime :=
  certify 225465241 (19) [p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_13, p_1877] (by decide)

def p_225615421 : CertifiedPrime :=
  certify 225615421 (2) [p_2, p_2, p_3, p_3, p_5, p_751, p_1669] (by decide)

def p_225825757 : CertifiedPrime :=
  certify 225825757 (2) [p_2, p_2, p_3, p_13, p_17, p_17, p_5009] (by decide)

def p_225900901 : CertifiedPrime :=
  certify 225900901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_167, p_167] (by decide)

def p_226066261 : CertifiedPrime :=
  certify 226066261 (6) [p_2, p_2, p_3, p_5, p_7, p_31, p_97, p_179] (by decide)

def p_226186561 : CertifiedPrime :=
  certify 226186561 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_47, p_557] (by decide)

def p_226502501 : CertifiedPrime :=
  certify 226502501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_43, p_43] (by decide)

def p_226878907 : CertifiedPrime :=
  certify 226878907 (2) [p_2, p_3, p_17, p_443, p_5021] (by decide)

def p_227104901 : CertifiedPrime :=
  certify 227104901 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_137, p_137] (by decide)

def p_227180257 : CertifiedPrime :=
  certify 227180257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_157, p_15073] (by decide)

def p_227225477 : CertifiedPrime :=
  certify 227225477 (2) [p_2, p_2, p_7537, p_7537] (by decide)

def p_227331007 : CertifiedPrime :=
  certify 227331007 (5) [p_2, p_3, p_7, p_359, p_15077] (by decide)

def p_227406401 : CertifiedPrime :=
  certify 227406401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_13, p_13, p_29, p_29] (by decide)

def p_227587397 : CertifiedPrime :=
  certify 227587397 (2) [p_2, p_2, p_19, p_19, p_397, p_397] (by decide)

def p_227723191 : CertifiedPrime :=
  certify 227723191 (3) [p_2, p_3, p_5, p_503, p_15091] (by decide)

def p_227874121 : CertifiedPrime :=
  certify 227874121 (7) [p_2, p_2, p_2, p_3, p_5, p_17, p_37, p_3019] (by decide)

def p_228130817 : CertifiedPrime :=
  certify 228130817 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_59, p_59] (by decide)

def p_228191237 : CertifiedPrime :=
  certify 228191237 (3) [p_2, p_2, p_7, p_7, p_13, p_13, p_83, p_83] (by decide)

def p_228236557 : CertifiedPrime :=
  certify 228236557 (5) [p_2, p_2, p_3, p_1259, p_15107] (by decide)

def p_228357433 : CertifiedPrime :=
  certify 228357433 (5) [p_2, p_2, p_2, p_3, p_3, p_23, p_73, p_1889] (by decide)

def p_228417883 : CertifiedPrime :=
  certify 228417883 (5) [p_2, p_3, p_7, p_11, p_17, p_127, p_229] (by decide)

def p_228780751 : CertifiedPrime :=
  certify 228780751 (3) [p_2, p_3, p_5, p_5, p_5, p_11, p_11, p_2521] (by decide)

def p_228811003 : CertifiedPrime :=
  certify 228811003 (3) [p_2, p_3, p_7, p_2161, p_2521] (by decide)

def p_228916901 : CertifiedPrime :=
  certify 228916901 (2) [p_2, p_2, p_5, p_5, p_17, p_17, p_89, p_89] (by decide)

def p_229143907 : CertifiedPrime :=
  certify 229143907 (5) [p_2, p_3, p_3, p_29, p_29, p_15137] (by decide)

def p_229234741 : CertifiedPrime :=
  certify 229234741 (13) [p_2, p_2, p_3, p_5, p_7, p_7, p_103, p_757] (by decide)

def p_229507351 : CertifiedPrime :=
  certify 229507351 (6) [p_2, p_3, p_5, p_5, p_101, p_15149] (by decide)

def p_229704337 : CertifiedPrime :=
  certify 229704337 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_421, p_421] (by decide)

def p_229780123 : CertifiedPrime :=
  certify 229780123 (2) [p_2, p_3, p_11, p_13, p_31, p_53, p_163] (by decide)

def p_229992391 : CertifiedPrime :=
  certify 229992391 (6) [p_2, p_3, p_3, p_5, p_337, p_7583] (by decide)

def p_230083393 : CertifiedPrime :=
  certify 230083393 (15) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_79, p_197] (by decide)

def p_230144071 : CertifiedPrime :=
  certify 230144071 (6) [p_2, p_3, p_5, p_13, p_37, p_41, p_389] (by decide)

def p_230250277 : CertifiedPrime :=
  certify 230250277 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_281, p_281] (by decide)

def p_230432401 : CertifiedPrime :=
  certify 230432401 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_11, p_23, p_23] (by decide)

def p_230508307 : CertifiedPrime :=
  certify 230508307 (2) [p_2, p_3, p_3, p_7, p_241, p_7591] (by decide)

def p_230614597 : CertifiedPrime :=
  certify 230614597 (2) [p_2, p_2, p_3, p_3, p_2531, p_2531] (by decide)

def p_230918417 : CertifiedPrime :=
  certify 230918417 (3) [p_2, p_2, p_2, p_2, p_29, p_29, p_131, p_131] (by decide)

def p_231161617 : CertifiedPrime :=
  certify 231161617 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_181, p_181] (by decide)

def p_231328891 : CertifiedPrime :=
  certify 231328891 (17) [p_2, p_3, p_3, p_5, p_13, p_13, p_67, p_227] (by decide)

def p_231344101 : CertifiedPrime :=
  certify 231344101 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_13, p_13, p_13, p_13] (by decide)

def p_231465797 : CertifiedPrime :=
  certify 231465797 (2) [p_2, p_2, p_7607, p_7607] (by decide)

def p_231694063 : CertifiedPrime :=
  certify 231694063 (3) [p_2, p_3, p_31, p_43, p_59, p_491] (by decide)

def p_231770177 : CertifiedPrime :=
  certify 231770177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_173, p_173] (by decide)

def p_231785401 : CertifiedPrime :=
  certify 231785401 (17) [p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_11, p_29, p_173] (by decide)

def p_231831077 : CertifiedPrime :=
  certify 231831077 (3) [p_2, p_2, p_23, p_23, p_331, p_331] (by decide)

def p_232272841 : CertifiedPrime :=
  certify 232272841 (7) [p_2, p_2, p_2, p_3, p_5, p_127, p_15241] (by decide)

def p_232333807 : CertifiedPrime :=
  certify 232333807 (3) [p_2, p_3, p_5081, p_7621] (by decide)

def p_232440517 : CertifiedPrime :=
  certify 232440517 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_11, p_11, p_11, p_11] (by decide)

def p_232608253 : CertifiedPrime :=
  certify 232608253 (6) [p_2, p_2, p_3, p_31, p_41, p_101, p_151] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
