import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk038

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_322705297 : CertifiedPrime :=
  certify 322705297 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_499, p_499] (by decide)

def p_322938871 : CertifiedPrime :=
  certify 322938871 (3) [p_2, p_3, p_5, p_599, p_17971] (by decide)

def p_323154553 : CertifiedPrime :=
  certify 323154553 (5) [p_2, p_2, p_2, p_3, p_7, p_107, p_17977] (by decide)

def p_323262421 : CertifiedPrime :=
  certify 323262421 (2) [p_2, p_2, p_3, p_5, p_13, p_29, p_31, p_461] (by decide)

def p_323496197 : CertifiedPrime :=
  certify 323496197 (2) [p_2, p_2, p_17, p_17, p_23, p_23, p_23, p_23] (by decide)

def p_323658091 : CertifiedPrime :=
  certify 323658091 (7) [p_2, p_3, p_3, p_5, p_7, p_257, p_1999] (by decide)

def p_323856017 : CertifiedPrime :=
  certify 323856017 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_409, p_409] (by decide)

def p_324126013 : CertifiedPrime :=
  certify 324126013 (2) [p_2, p_2, p_3, p_7, p_17, p_353, p_643] (by decide)

def p_324414133 : CertifiedPrime :=
  certify 324414133 (6) [p_2, p_2, p_3, p_7, p_19, p_31, p_79, p_83] (by decide)

def p_324504197 : CertifiedPrime :=
  certify 324504197 (2) [p_2, p_2, p_9007, p_9007] (by decide)

def p_324666343 : CertifiedPrime :=
  certify 324666343 (38) [p_2, p_3, p_3, p_7, p_11, p_13, p_37, p_487] (by decide)

def p_324882601 : CertifiedPrime :=
  certify 324882601 (33) [p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_103, p_751] (by decide)

def p_325171057 : CertifiedPrime :=
  certify 325171057 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_23, p_6011] (by decide)

def p_325225157 : CertifiedPrime :=
  certify 325225157 (2) [p_2, p_2, p_71, p_71, p_127, p_127] (by decide)

def p_325441601 : CertifiedPrime :=
  certify 325441601 (12) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_11, p_11, p_41, p_41] (by decide)

def p_325603981 : CertifiedPrime :=
  certify 325603981 (2) [p_2, p_2, p_3, p_3, p_5, p_13, p_347, p_401] (by decide)

def p_325658117 : CertifiedPrime :=
  certify 325658117 (2) [p_2, p_2, p_7, p_7, p_1289, p_1289] (by decide)

def p_325946917 : CertifiedPrime :=
  certify 325946917 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_17, p_17, p_59, p_59] (by decide)

def p_326163601 : CertifiedPrime :=
  certify 326163601 (31) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_43, p_43] (by decide)

def p_326290033 : CertifiedPrime :=
  certify 326290033 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_223, p_1129] (by decide)

def p_326362291 : CertifiedPrime :=
  certify 326362291 (3) [p_2, p_3, p_5, p_3011, p_3613] (by decide)

def p_326380357 : CertifiedPrime :=
  certify 326380357 (2) [p_2, p_2, p_3, p_3, p_3011, p_3011] (by decide)

def p_326524901 : CertifiedPrime :=
  certify 326524901 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_139, p_139] (by decide)

def p_326615257 : CertifiedPrime :=
  certify 326615257 (5) [p_2, p_2, p_2, p_3, p_3, p_11, p_31, p_53, p_251] (by decide)

def p_326687551 : CertifiedPrime :=
  certify 326687551 (3) [p_2, p_3, p_5, p_5, p_7, p_241, p_1291] (by decide)

def p_326723701 : CertifiedPrime :=
  certify 326723701 (2) [p_2, p_2, p_3, p_5, p_5, p_241, p_4519] (by decide)

def p_326741777 : CertifiedPrime :=
  certify 326741777 (3) [p_2, p_2, p_2, p_2, p_4519, p_4519] (by decide)

def p_326886401 : CertifiedPrime :=
  certify 326886401 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_113, p_113] (by decide)

def p_327049141 : CertifiedPrime :=
  certify 327049141 (6) [p_2, p_2, p_3, p_5, p_11, p_137, p_3617] (by decide)

def p_327230011 : CertifiedPrime :=
  certify 327230011 (11) [p_2, p_3, p_3, p_3, p_5, p_67, p_18089] (by decide)

def p_327392837 : CertifiedPrime :=
  certify 327392837 (2) [p_2, p_2, p_83, p_83, p_109, p_109] (by decide)

def p_327610001 : CertifiedPrime :=
  certify 327610001 (7) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_181, p_181] (by decide)

def p_327664303 : CertifiedPrime :=
  certify 327664303 (3) [p_2, p_3, p_7, p_23, p_431, p_787] (by decide)

def p_327881557 : CertifiedPrime :=
  certify 327881557 (5) [p_2, p_2, p_3, p_3, p_19, p_503, p_953] (by decide)

def p_327917773 : CertifiedPrime :=
  certify 327917773 (11) [p_2, p_2, p_3, p_3, p_7, p_13, p_199, p_503] (by decide)

def p_328207573 : CertifiedPrime :=
  certify 328207573 (2) [p_2, p_2, p_3, p_3, p_3, p_7, p_11, p_61, p_647] (by decide)

def p_328243807 : CertifiedPrime :=
  certify 328243807 (5) [p_2, p_3, p_3, p_3, p_11, p_61, p_9059] (by decide)

def p_328334401 : CertifiedPrime :=
  certify 328334401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_151, p_151] (by decide)

def p_328570003 : CertifiedPrime :=
  certify 328570003 (5) [p_2, p_3, p_3, p_19, p_53, p_18127] (by decide)

def p_328896361 : CertifiedPrime :=
  certify 328896361 (14) [p_2, p_2, p_2, p_3, p_3, p_5, p_13, p_31, p_2267] (by decide)

def p_329204737 : CertifiedPrime :=
  certify 329204737 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7] (by decide)

def p_329277317 : CertifiedPrime :=
  certify 329277317 (2) [p_2, p_2, p_43, p_43, p_211, p_211] (by decide)

def p_329331757 : CertifiedPrime :=
  certify 329331757 (6) [p_2, p_2, p_3, p_13, p_23, p_263, p_349] (by decide)

def p_329404351 : CertifiedPrime :=
  certify 329404351 (3) [p_2, p_3, p_5, p_5, p_11, p_11, p_18149] (by decide)

def p_329440651 : CertifiedPrime :=
  certify 329440651 (7) [p_2, p_3, p_5, p_5, p_7, p_11, p_11, p_2593] (by decide)

def p_329731123 : CertifiedPrime :=
  certify 329731123 (2) [p_2, p_3, p_7, p_1297, p_6053] (by decide)

def p_329930897 : CertifiedPrime :=
  certify 329930897 (3) [p_2, p_2, p_2, p_2, p_19, p_19, p_239, p_239] (by decide)

def p_330294277 : CertifiedPrime :=
  certify 330294277 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_233, p_233] (by decide)

def p_330366977 : CertifiedPrime :=
  certify 330366977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_71, p_71] (by decide)

def p_330494221 : CertifiedPrime :=
  certify 330494221 (11) [p_2, p_2, p_3, p_3, p_5, p_7, p_7, p_7, p_53, p_101] (by decide)

def p_331094417 : CertifiedPrime :=
  certify 331094417 (3) [p_2, p_2, p_2, p_2, p_4549, p_4549] (by decide)

def p_331185403 : CertifiedPrime :=
  certify 331185403 (22) [p_2, p_3, p_3, p_3, p_337, p_18199] (by decide)

def p_331695157 : CertifiedPrime :=
  certify 331695157 (5) [p_2, p_2, p_3, p_13, p_29, p_157, p_467] (by decide)

def p_331822657 : CertifiedPrime :=
  certify 331822657 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_11, p_11, p_23, p_23] (by decide)

def p_332332901 : CertifiedPrime :=
  certify 332332901 (3) [p_2, p_2, p_5, p_5, p_1823, p_1823] (by decide)

def p_333336307 : CertifiedPrime :=
  certify 333336307 (3) [p_2, p_3, p_17, p_179, p_18257] (by decide)

def p_333555433 : CertifiedPrime :=
  certify 333555433 (5) [p_2, p_2, p_2, p_3, p_7, p_761, p_2609] (by decide)

def p_333774631 : CertifiedPrime :=
  certify 333774631 (13) [p_2, p_3, p_3, p_5, p_7, p_29, p_18269] (by decide)

def p_333939077 : CertifiedPrime :=
  certify 333939077 (2) [p_2, p_2, p_9137, p_9137] (by decide)

def p_333993901 : CertifiedPrime :=
  certify 333993901 (6) [p_2, p_2, p_3, p_5, p_5, p_17, p_43, p_1523] (by decide)

def p_334012177 : CertifiedPrime :=
  certify 334012177 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_1523, p_1523] (by decide)

def p_334030453 : CertifiedPrime :=
  certify 334030453 (5) [p_2, p_2, p_3, p_7, p_7, p_373, p_1523] (by decide)

def p_334103563 : CertifiedPrime :=
  certify 334103563 (2) [p_2, p_3, p_3, p_3, p_13, p_19, p_37, p_677] (by decide)

def p_334158401 : CertifiedPrime :=
  certify 334158401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_457, p_457] (by decide)

def p_334249807 : CertifiedPrime :=
  certify 334249807 (3) [p_2, p_3, p_11, p_47, p_277, p_389] (by decide)

def p_334524101 : CertifiedPrime :=
  certify 334524101 (2) [p_2, p_2, p_5, p_5, p_31, p_31, p_59, p_59] (by decide)

def p_334542391 : CertifiedPrime :=
  certify 334542391 (3) [p_2, p_3, p_5, p_7, p_13, p_31, p_59, p_67] (by decide)

def p_335127943 : CertifiedPrime :=
  certify 335127943 (5) [p_2, p_3, p_3, p_3, p_3, p_113, p_18307] (by decide)

def p_335622401 : CertifiedPrime :=
  certify 335622401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_229, p_229] (by decide)

def p_335768977 : CertifiedPrime :=
  certify 335768977 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_509, p_509] (by decide)

def p_335842277 : CertifiedPrime :=
  certify 335842277 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_11, p_11, p_17, p_17] (by decide)

def p_336080557 : CertifiedPrime :=
  certify 336080557 (5) [p_2, p_2, p_3, p_3, p_3, p_7, p_97, p_4583] (by decide)

def p_336135557 : CertifiedPrime :=
  certify 336135557 (2) [p_2, p_2, p_89, p_89, p_103, p_103] (by decide)

def p_336227233 : CertifiedPrime :=
  certify 336227233 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_11, p_191, p_1667] (by decide)

def p_336502337 : CertifiedPrime :=
  certify 336502337 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2293, p_2293] (by decide)

def p_336557371 : CertifiedPrime :=
  certify 336557371 (10) [p_2, p_3, p_5, p_1223, p_9173] (by decide)

def p_336667453 : CertifiedPrime :=
  certify 336667453 (6) [p_2, p_2, p_3, p_11, p_59, p_139, p_311] (by decide)

def p_337089601 : CertifiedPrime :=
  certify 337089601 (29) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_17, p_17] (by decide)

def p_337309957 : CertifiedPrime :=
  certify 337309957 (6) [p_2, p_2, p_3, p_3, p_3061, p_3061] (by decide)

def p_337438531 : CertifiedPrime :=
  certify 337438531 (2) [p_2, p_3, p_3, p_5, p_11, p_13, p_157, p_167] (by decide)

def p_337677377 : CertifiedPrime :=
  certify 337677377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2297, p_2297] (by decide)

def p_337971457 : CertifiedPrime :=
  certify 337971457 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_383, p_383] (by decide)

def p_338044997 : CertifiedPrime :=
  certify 338044997 (2) [p_2, p_2, p_29, p_29, p_317, p_317] (by decide)

def p_338284057 : CertifiedPrime :=
  certify 338284057 (5) [p_2, p_2, p_2, p_3, p_11, p_11, p_19, p_6131] (by decide)

def p_338412817 : CertifiedPrime :=
  certify 338412817 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_73, p_73] (by decide)

def p_338431213 : CertifiedPrime :=
  certify 338431213 (5) [p_2, p_2, p_3, p_3, p_7, p_73, p_18397] (by decide)

def p_338615203 : CertifiedPrime :=
  certify 338615203 (3) [p_2, p_3, p_3067, p_18401] (by decide)

def p_338780837 : CertifiedPrime :=
  certify 338780837 (2) [p_2, p_2, p_9203, p_9203] (by decide)

def p_338928101 : CertifiedPrime :=
  certify 338928101 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_263, p_263] (by decide)

def p_339167473 : CertifiedPrime :=
  certify 339167473 (10) [p_2, p_2, p_2, p_2, p_3, p_7, p_877, p_1151] (by decide)

def p_339425353 : CertifiedPrime :=
  certify 339425353 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_23, p_47, p_89] (by decide)

def p_339664901 : CertifiedPrime :=
  certify 339664901 (2) [p_2, p_2, p_5, p_5, p_19, p_19, p_97, p_97] (by decide)

def p_339978283 : CertifiedPrime :=
  certify 339978283 (3) [p_2, p_3, p_7, p_439, p_18439] (by decide)

def p_340181137 : CertifiedPrime :=
  certify 340181137 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_29, p_29, p_53, p_53] (by decide)

def p_340273363 : CertifiedPrime :=
  certify 340273363 (2) [p_2, p_3, p_11, p_13, p_23, p_43, p_401] (by decide)

def p_340402501 : CertifiedPrime :=
  certify 340402501 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_41, p_41] (by decide)

def p_340919297 : CertifiedPrime :=
  certify 340919297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_577, p_577] (by decide)

def p_340937761 : CertifiedPrime :=
  certify 340937761 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_577, p_1231] (by decide)

def p_341048557 : CertifiedPrime :=
  certify 341048557 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_19, p_59, p_313] (by decide)

def p_341307151 : CertifiedPrime :=
  certify 341307151 (3) [p_2, p_3, p_5, p_5, p_739, p_3079] (by decide)

def p_341510401 : CertifiedPrime :=
  certify 341510401 (19) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_11, p_11] (by decide)

def p_341713711 : CertifiedPrime :=
  certify 341713711 (11) [p_2, p_3, p_3, p_5, p_13, p_79, p_3697] (by decide)

def p_341732197 : CertifiedPrime :=
  certify 341732197 (15) [p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13, p_79, p_79] (by decide)

def p_341750683 : CertifiedPrime :=
  certify 341750683 (5) [p_2, p_3, p_3, p_7, p_13, p_19, p_79, p_139] (by decide)

def p_341824633 : CertifiedPrime :=
  certify 341824633 (5) [p_2, p_2, p_2, p_3, p_2311, p_6163] (by decide)

def p_341972557 : CertifiedPrime :=
  certify 341972557 (5) [p_2, p_2, p_3, p_23, p_67, p_18493] (by decide)

def p_342157507 : CertifiedPrime :=
  certify 342157507 (3) [p_2, p_3, p_53, p_349, p_3083] (by decide)

def p_342268501 : CertifiedPrime :=
  certify 342268501 (2) [p_2, p_2, p_3, p_5, p_5, p_5, p_7, p_37, p_881] (by decide)

def p_342490543 : CertifiedPrime :=
  certify 342490543 (3) [p_2, p_3, p_19, p_31, p_199, p_487] (by decide)

def p_342527557 : CertifiedPrime :=
  certify 342527557 (5) [p_2, p_2, p_3, p_7, p_31, p_199, p_661] (by decide)

def p_342860773 : CertifiedPrime :=
  certify 342860773 (5) [p_2, p_2, p_3, p_1543, p_18517] (by decide)

def p_343138577 : CertifiedPrime :=
  certify 343138577 (3) [p_2, p_2, p_2, p_2, p_11, p_11, p_421, p_421] (by decide)

def p_343194151 : CertifiedPrime :=
  certify 343194151 (6) [p_2, p_3, p_5, p_5, p_13, p_19, p_59, p_157] (by decide)

def p_343268257 : CertifiedPrime :=
  certify 343268257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_97, p_191, p_193] (by decide)

def p_343638907 : CertifiedPrime :=
  certify 343638907 (2) [p_2, p_3, p_13, p_23, p_31, p_37, p_167] (by decide)

def p_343731601 : CertifiedPrime :=
  certify 343731601 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_103, p_103] (by decide)

def p_343824307 : CertifiedPrime :=
  certify 343824307 (3) [p_2, p_3, p_7, p_73, p_127, p_883] (by decide)

def p_343972663 : CertifiedPrime :=
  certify 343972663 (3) [p_2, p_3, p_11, p_17, p_281, p_1091] (by decide)

def p_344083951 : CertifiedPrime :=
  certify 344083951 (6) [p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_53, p_229] (by decide)

def p_344417923 : CertifiedPrime :=
  certify 344417923 (11) [p_2, p_3, p_3, p_67, p_277, p_1031] (by decide)

def p_344640661 : CertifiedPrime :=
  certify 344640661 (10) [p_2, p_2, p_3, p_5, p_7, p_13, p_17, p_47, p_79] (by decide)

def p_344696357 : CertifiedPrime :=
  certify 344696357 (2) [p_2, p_2, p_9283, p_9283] (by decide)

def p_345067777 : CertifiedPrime :=
  certify 345067777 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_43, p_43] (by decide)

def p_345086353 : CertifiedPrime :=
  certify 345086353 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_13, p_43, p_1429] (by decide)

def p_345272143 : CertifiedPrime :=
  certify 345272143 (3) [p_2, p_3, p_17, p_19, p_163, p_1093] (by decide)

def p_345365057 : CertifiedPrime :=
  certify 345365057 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_23, p_23, p_101, p_101] (by decide)

def p_345439397 : CertifiedPrime :=
  certify 345439397 (2) [p_2, p_2, p_9293, p_9293] (by decide)

def p_345606691 : CertifiedPrime :=
  certify 345606691 (2) [p_2, p_3, p_5, p_11, p_13, p_13, p_6197] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
