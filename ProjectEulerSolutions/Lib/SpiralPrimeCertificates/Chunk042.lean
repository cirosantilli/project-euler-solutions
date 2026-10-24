import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk041

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_399940003 : CertifiedPrime :=
  certify 399940003 (7) [p_2, p_3, p_3, p_7, p_11, p_101, p_2857] (by decide)

def p_400300057 : CertifiedPrime :=
  certify 400300057 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_19, p_41, p_61] (by decide)

def p_400500157 : CertifiedPrime :=
  certify 400500157 (2) [p_2, p_2, p_3, p_7, p_953, p_5003] (by decide)

def p_400540183 : CertifiedPrime :=
  certify 400540183 (3) [p_2, p_3, p_7, p_953, p_10007] (by decide)

def p_400560197 : CertifiedPrime :=
  certify 400560197 (2) [p_2, p_2, p_10007, p_10007] (by decide)

def p_401040677 : CertifiedPrime :=
  certify 401040677 (2) [p_2, p_2, p_17, p_17, p_19, p_19, p_31, p_31] (by decide)

def p_401441297 : CertifiedPrime :=
  certify 401441297 (3) [p_2, p_2, p_2, p_2, p_5009, p_5009] (by decide)

def p_401581561 : CertifiedPrime :=
  certify 401581561 (7) [p_2, p_2, p_2, p_3, p_5, p_29, p_167, p_691] (by decide)

def p_401621641 : CertifiedPrime :=
  certify 401621641 (17) [p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_167, p_409] (by decide)

def p_402644357 : CertifiedPrime :=
  certify 402644357 (2) [p_2, p_2, p_79, p_79, p_127, p_127] (by decide)

def p_402704557 : CertifiedPrime :=
  certify 402704557 (2) [p_2, p_2, p_3, p_29, p_173, p_6689] (by decide)

def p_402784831 : CertifiedPrime :=
  certify 402784831 (7) [p_2, p_3, p_3, p_5, p_7, p_47, p_61, p_223] (by decide)

def p_403045777 : CertifiedPrime :=
  certify 403045777 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_239, p_239] (by decide)

def p_403065853 : CertifiedPrime :=
  certify 403065853 (5) [p_2, p_2, p_3, p_7, p_17, p_239, p_1181] (by decide)

def p_403447397 : CertifiedPrime :=
  certify 403447397 (3) [p_2, p_2, p_11, p_11, p_11, p_11, p_83, p_83] (by decide)

def p_403849217 : CertifiedPrime :=
  certify 403849217 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_157, p_157] (by decide)

def p_403869313 : CertifiedPrime :=
  certify 403869313 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_11, p_29, p_157] (by decide)

def p_403909507 : CertifiedPrime :=
  certify 403909507 (2) [p_2, p_3, p_3, p_7, p_11, p_13, p_29, p_773] (by decide)

def p_404010001 : CertifiedPrime :=
  certify 404010001 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_67, p_67] (by decide)

def p_404110507 : CertifiedPrime :=
  certify 404110507 (2) [p_2, p_3, p_19, p_23, p_23, p_6701] (by decide)

def p_404170817 : CertifiedPrime :=
  certify 404170817 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_359, p_359] (by decide)

def p_404231131 : CertifiedPrime :=
  certify 404231131 (10) [p_2, p_3, p_3, p_5, p_1117, p_4021] (by decide)

def p_404472433 : CertifiedPrime :=
  certify 404472433 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_13, p_17, p_419] (by decide)

def p_404754043 : CertifiedPrime :=
  certify 404754043 (13) [p_2, p_3, p_7, p_11, p_31, p_59, p_479] (by decide)

def p_404814401 : CertifiedPrime :=
  certify 404814401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_503, p_503] (by decide)

def p_404955253 : CertifiedPrime :=
  certify 404955253 (5) [p_2, p_2, p_3, p_3, p_13, p_43, p_20123] (by decide)

def p_405438361 : CertifiedPrime :=
  certify 405438361 (17) [p_2, p_2, p_2, p_3, p_5, p_839, p_4027] (by decide)

def p_405458497 : CertifiedPrime :=
  certify 405458497 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_839, p_839] (by decide)

def p_405720307 : CertifiedPrime :=
  certify 405720307 (5) [p_2, p_3, p_3, p_3, p_373, p_20143] (by decide)

def p_405841171 : CertifiedPrime :=
  certify 405841171 (3) [p_2, p_3, p_5, p_7, p_17, p_79, p_1439] (by decide)

def p_405861317 : CertifiedPrime :=
  certify 405861317 (2) [p_2, p_2, p_7, p_7, p_1439, p_1439] (by decide)

def p_406264337 : CertifiedPrime :=
  certify 406264337 (3) [p_2, p_2, p_2, p_2, p_5039, p_5039] (by decide)

def p_406425601 : CertifiedPrime :=
  certify 406425601 (17) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_7] (by decide)

def p_406586897 : CertifiedPrime :=
  certify 406586897 (3) [p_2, p_2, p_2, p_2, p_71, p_71, p_71, p_71] (by decide)

def p_407292943 : CertifiedPrime :=
  certify 407292943 (3) [p_2, p_3, p_7, p_31, p_31, p_10091] (by decide)

def p_407414041 : CertifiedPrime :=
  certify 407414041 (13) [p_2, p_2, p_2, p_3, p_5, p_11, p_29, p_29, p_367] (by decide)

def p_407474597 : CertifiedPrime :=
  certify 407474597 (2) [p_2, p_2, p_10093, p_10093] (by decide)

def p_407777443 : CertifiedPrime :=
  certify 407777443 (2) [p_2, p_3, p_23, p_53, p_127, p_439] (by decide)

def p_407858221 : CertifiedPrime :=
  certify 407858221 (26) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_11, p_17, p_577] (by decide)

def p_408282437 : CertifiedPrime :=
  certify 408282437 (2) [p_2, p_2, p_10103, p_10103] (by decide)

def p_408343057 : CertifiedPrime :=
  certify 408343057 (5) [p_2, p_2, p_2, p_2, p_3, p_11, p_11, p_167, p_421] (by decide)

def p_408605797 : CertifiedPrime :=
  certify 408605797 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_1123, p_1123] (by decide)

def p_408626011 : CertifiedPrime :=
  certify 408626011 (10) [p_2, p_3, p_3, p_5, p_13, p_311, p_1123] (by decide)

def p_408828181 : CertifiedPrime :=
  certify 408828181 (6) [p_2, p_2, p_3, p_5, p_337, p_20219] (by decide)

def p_409657601 : CertifiedPrime :=
  certify 409657601 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_11, p_11, p_23, p_23] (by decide)

def p_409799293 : CertifiedPrime :=
  certify 409799293 (5) [p_2, p_2, p_3, p_7, p_31, p_241, p_653] (by decide)

def p_409819537 : CertifiedPrime :=
  certify 409819537 (15) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_241, p_241] (by decide)

def p_410062501 : CertifiedPrime :=
  certify 410062501 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_5, p_5, p_5, p_5] (by decide)

def p_410305537 : CertifiedPrime :=
  certify 410305537 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_211, p_211] (by decide)

def p_410325793 : CertifiedPrime :=
  certify 410325793 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_47, p_211, p_431] (by decide)

def p_410406823 : CertifiedPrime :=
  certify 410406823 (5) [p_2, p_3, p_3, p_7, p_1447, p_2251] (by decide)

def p_410710757 : CertifiedPrime :=
  certify 410710757 (2) [p_2, p_2, p_10133, p_10133] (by decide)

def p_411177007 : CertifiedPrime :=
  certify 411177007 (5) [p_2, p_3, p_3, p_3, p_751, p_10139] (by decide)

def p_411278401 : CertifiedPrime :=
  certify 411278401 (22) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13, p_13, p_13] (by decide)

def p_411379807 : CertifiedPrime :=
  certify 411379807 (3) [p_2, p_3, p_6761, p_10141] (by decide)

def p_412090001 : CertifiedPrime :=
  certify 412090001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7, p_29, p_29] (by decide)

def p_412150903 : CertifiedPrime :=
  certify 412150903 (3) [p_2, p_3, p_67, p_101, p_10151] (by decide)

def p_412333637 : CertifiedPrime :=
  certify 412333637 (2) [p_2, p_2, p_11, p_11, p_13, p_13, p_71, p_71] (by decide)

def p_412353943 : CertifiedPrime :=
  certify 412353943 (11) [p_2, p_3, p_7, p_11, p_13, p_71, p_967] (by decide)

def p_412394557 : CertifiedPrime :=
  certify 412394557 (2) [p_2, p_2, p_3, p_7, p_967, p_5077] (by decide)

def p_412597657 : CertifiedPrime :=
  certify 412597657 (5) [p_2, p_2, p_2, p_3, p_3, p_37, p_61, p_2539] (by decide)

def p_412638283 : CertifiedPrime :=
  certify 412638283 (2) [p_2, p_3, p_3, p_7, p_37, p_61, p_1451] (by decide)

def p_413085301 : CertifiedPrime :=
  certify 413085301 (2) [p_2, p_2, p_3, p_5, p_5, p_271, p_5081] (by decide)

def p_413735941 : CertifiedPrime :=
  certify 413735941 (13) [p_2, p_2, p_3, p_3, p_5, p_113, p_20341] (by decide)

def p_414285317 : CertifiedPrime :=
  certify 414285317 (2) [p_2, p_2, p_10177, p_10177] (by decide)

def p_414672133 : CertifiedPrime :=
  certify 414672133 (5) [p_2, p_2, p_3, p_7, p_1697, p_2909] (by decide)

def p_414692497 : CertifiedPrime :=
  certify 414692497 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1697, p_1697] (by decide)

def p_414794323 : CertifiedPrime :=
  certify 414794323 (2) [p_2, p_3, p_3, p_17, p_31, p_73, p_599] (by decide)

def p_414916531 : CertifiedPrime :=
  certify 414916531 (15) [p_2, p_3, p_5, p_7, p_97, p_20369] (by decide)

def p_414957271 : CertifiedPrime :=
  certify 414957271 (3) [p_2, p_3, p_5, p_7, p_13, p_97, p_1567] (by decide)

def p_415099877 : CertifiedPrime :=
  certify 415099877 (2) [p_2, p_2, p_61, p_61, p_167, p_167] (by decide)

def p_415283263 : CertifiedPrime :=
  certify 415283263 (3) [p_2, p_3, p_23, p_443, p_6793] (by decide)

def p_415344401 : CertifiedPrime :=
  certify 415344401 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_1019, p_1019] (by decide)

def p_415507457 : CertifiedPrime :=
  certify 415507457 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_7, p_7, p_13, p_13] (by decide)

def p_415527841 : CertifiedPrime :=
  certify 415527841 (46) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_7, p_13, p_151] (by decide)

def p_415568611 : CertifiedPrime :=
  certify 415568611 (2) [p_2, p_3, p_3, p_3, p_5, p_151, p_10193] (by decide)

def p_416139601 : CertifiedPrime :=
  certify 416139601 (21) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_17, p_20399] (by decide)

def p_416160001 : CertifiedPrime :=
  certify 416160001 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_17, p_17] (by decide)

def p_416404837 : CertifiedPrime :=
  certify 416404837 (2) [p_2, p_2, p_3, p_3, p_19, p_19, p_179, p_179] (by decide)

def p_416731397 : CertifiedPrime :=
  certify 416731397 (2) [p_2, p_2, p_59, p_59, p_173, p_173] (by decide)

def p_417160201 : CertifiedPrime :=
  certify 417160201 (11) [p_2, p_2, p_2, p_3, p_5, p_5, p_19, p_23, p_37, p_43] (by decide)

def p_417793601 : CertifiedPrime :=
  certify 417793601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_73, p_73] (by decide)

def p_418100257 : CertifiedPrime :=
  certify 418100257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_23, p_71, p_127] (by decide)

def p_418632061 : CertifiedPrime :=
  certify 418632061 (6) [p_2, p_2, p_3, p_5, p_7, p_11, p_31, p_37, p_79] (by decide)

def p_418713907 : CertifiedPrime :=
  certify 418713907 (2) [p_2, p_3, p_13, p_19, p_359, p_787] (by decide)

def p_419266577 : CertifiedPrime :=
  certify 419266577 (3) [p_2, p_2, p_2, p_2, p_5119, p_5119] (by decide)

def p_419450881 : CertifiedPrime :=
  certify 419450881 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_6827] (by decide)

def p_419594257 : CertifiedPrime :=
  certify 419594257 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_569, p_569] (by decide)

def p_420229501 : CertifiedPrime :=
  certify 420229501 (2) [p_2, p_2, p_3, p_5, p_5, p_5, p_41, p_6833] (by decide)

def p_420250001 : CertifiedPrime :=
  certify 420250001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_41, p_41] (by decide)

def p_420824197 : CertifiedPrime :=
  certify 420824197 (5) [p_2, p_2, p_3, p_3, p_13, p_13, p_263, p_263] (by decide)

def p_421090921 : CertifiedPrime :=
  certify 421090921 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_19, p_20521] (by decide)

def p_421173007 : CertifiedPrime :=
  certify 421173007 (3) [p_2, p_3, p_31, p_331, p_6841] (by decide)

def p_421296151 : CertifiedPrime :=
  certify 421296151 (6) [p_2, p_3, p_5, p_5, p_11, p_311, p_821] (by decide)

def p_421727297 : CertifiedPrime :=
  certify 421727297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_17, p_17, p_151, p_151] (by decide)

def p_421953223 : CertifiedPrime :=
  certify 421953223 (3) [p_2, p_3, p_41, p_167, p_10271] (by decide)

def p_422076481 : CertifiedPrime :=
  certify 422076481 (13) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_107, p_587] (by decide)

def p_422405257 : CertifiedPrime :=
  certify 422405257 (5) [p_2, p_2, p_2, p_3, p_7, p_13, p_17, p_31, p_367] (by decide)

def p_422549137 : CertifiedPrime :=
  certify 422549137 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_571, p_571] (by decide)

def p_422878097 : CertifiedPrime :=
  certify 422878097 (3) [p_2, p_2, p_2, p_2, p_53, p_53, p_97, p_97] (by decide)

def p_423268903 : CertifiedPrime :=
  certify 423268903 (5) [p_2, p_3, p_3, p_3, p_3, p_7, p_127, p_2939] (by decide)

def p_423310051 : CertifiedPrime :=
  certify 423310051 (10) [p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_127, p_823] (by decide)

def p_423392353 : CertifiedPrime :=
  certify 423392353 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_19, p_19, p_19, p_643] (by decide)

def p_423556981 : CertifiedPrime :=
  certify 423556981 (6) [p_2, p_2, p_3, p_5, p_7, p_7, p_7, p_11, p_1871] (by decide)

def p_423680473 : CertifiedPrime :=
  certify 423680473 (7) [p_2, p_2, p_2, p_3, p_3, p_31, p_83, p_2287] (by decide)

def p_423948101 : CertifiedPrime :=
  certify 423948101 (2) [p_2, p_2, p_5, p_5, p_29, p_29, p_71, p_71] (by decide)

def p_424195217 : CertifiedPrime :=
  certify 424195217 (3) [p_2, p_2, p_2, p_2, p_19, p_19, p_271, p_271] (by decide)

def p_424916383 : CertifiedPrime :=
  certify 424916383 (3) [p_2, p_3, p_11, p_937, p_6871] (by decide)

def p_425019457 : CertifiedPrime :=
  certify 425019457 (15) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_859, p_859] (by decide)

def p_425040073 : CertifiedPrime :=
  certify 425040073 (5) [p_2, p_2, p_2, p_3, p_53, p_389, p_859] (by decide)

def p_425184401 : CertifiedPrime :=
  certify 425184401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_1031, p_1031] (by decide)

def p_425349377 : CertifiedPrime :=
  certify 425349377 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1289, p_1289] (by decide)

def p_425535013 : CertifiedPrime :=
  certify 425535013 (14) [p_2, p_2, p_3, p_3, p_3, p_7, p_7, p_191, p_421] (by decide)

def p_425596901 : CertifiedPrime :=
  certify 425596901 (2) [p_2, p_2, p_5, p_5, p_2063, p_2063] (by decide)

def p_425617531 : CertifiedPrime :=
  certify 425617531 (2) [p_2, p_3, p_5, p_13, p_23, p_23, p_2063] (by decide)

def p_425865133 : CertifiedPrime :=
  certify 425865133 (2) [p_2, p_2, p_3, p_3, p_7, p_11, p_67, p_2293] (by decide)

def p_426174737 : CertifiedPrime :=
  certify 426174737 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_397, p_397] (by decide)

def p_426587717 : CertifiedPrime :=
  certify 426587717 (2) [p_2, p_2, p_23, p_23, p_449, p_449] (by decide)

def p_426649681 : CertifiedPrime :=
  certify 426649681 (28) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_5, p_17, p_1291] (by decide)

def p_426670337 : CertifiedPrime :=
  certify 426670337 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1291, p_1291] (by decide)

def p_427228231 : CertifiedPrime :=
  certify 427228231 (3) [p_2, p_3, p_5, p_11, p_13, p_53, p_1879] (by decide)

def p_427248901 : CertifiedPrime :=
  certify 427248901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13, p_53, p_53] (by decide)

def p_427827857 : CertifiedPrime :=
  certify 427827857 (3) [p_2, p_2, p_2, p_2, p_5171, p_5171] (by decide)

def p_427848541 : CertifiedPrime :=
  certify 427848541 (2) [p_2, p_2, p_3, p_5, p_7, p_197, p_5171] (by decide)

def p_428014033 : CertifiedPrime :=
  certify 428014033 (7) [p_2, p_2, p_2, p_2, p_3, p_17, p_431, p_1217] (by decide)

def p_428510701 : CertifiedPrime :=
  certify 428510701 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_23, p_127, p_163] (by decide)

def p_428738437 : CertifiedPrime :=
  certify 428738437 (5) [p_2, p_2, p_3, p_3, p_7, p_7, p_17, p_17, p_29, p_29] (by decide)

def p_428759143 : CertifiedPrime :=
  certify 428759143 (5) [p_2, p_3, p_7, p_17, p_29, p_20707] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
