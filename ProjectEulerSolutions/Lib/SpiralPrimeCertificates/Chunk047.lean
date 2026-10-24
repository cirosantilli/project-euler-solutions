import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk046

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_552720101 : CertifiedPrime :=
  certify 552720101 (2) [p_2, p_2, p_5, p_5, p_2351, p_2351] (by decide)

def p_553025773 : CertifiedPrime :=
  certify 553025773 (2) [p_2, p_2, p_3, p_3, p_3, p_13, p_67, p_5879] (by decide)

def p_553166881 : CertifiedPrime :=
  certify 553166881 (23) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_29, p_811] (by decide)

def p_553449151 : CertifiedPrime :=
  certify 553449151 (13) [p_2, p_3, p_3, p_5, p_5, p_941, p_1307] (by decide)

def p_554343481 : CertifiedPrime :=
  certify 554343481 (7) [p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_17, p_109, p_277] (by decide)

def p_554414117 : CertifiedPrime :=
  certify 554414117 (3) [p_2, p_2, p_61, p_61, p_193, p_193] (by decide)

def p_554602501 : CertifiedPrime :=
  certify 554602501 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_157, p_157] (by decide)

def p_554767363 : CertifiedPrime :=
  certify 554767363 (2) [p_2, p_3, p_3, p_2617, p_11777] (by decide)

def p_554885137 : CertifiedPrime :=
  certify 554885137 (15) [p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13, p_151, p_151] (by decide)

def p_554908693 : CertifiedPrime :=
  certify 554908693 (5) [p_2, p_2, p_3, p_13, p_151, p_23557] (by decide)

def p_555474193 : CertifiedPrime :=
  certify 555474193 (15) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_13, p_37, p_491] (by decide)

def p_555827777 : CertifiedPrime :=
  certify 555827777 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_421, p_421] (by decide)

def p_557148817 : CertifiedPrime :=
  certify 557148817 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_281, p_281] (by decide)

def p_557550157 : CertifiedPrime :=
  certify 557550157 (2) [p_2, p_2, p_3, p_17, p_463, p_5903] (by decide)

def p_557833543 : CertifiedPrime :=
  certify 557833543 (5) [p_2, p_3, p_7, p_7, p_241, p_7873] (by decide)

def p_558447793 : CertifiedPrime :=
  certify 558447793 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_211, p_7877] (by decide)

def p_558825961 : CertifiedPrime :=
  certify 558825961 (17) [p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_197, p_307] (by decide)

def p_558873241 : CertifiedPrime :=
  certify 558873241 (7) [p_2, p_2, p_2, p_3, p_5, p_47, p_197, p_503] (by decide)

def p_558967807 : CertifiedPrime :=
  certify 558967807 (5) [p_2, p_3, p_3, p_37, p_71, p_11821] (by decide)

def p_559109671 : CertifiedPrime :=
  certify 559109671 (6) [p_2, p_3, p_5, p_7, p_563, p_4729] (by decide)

def p_559322501 : CertifiedPrime :=
  certify 559322501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_11, p_11, p_43, p_43] (by decide)

def p_559606337 : CertifiedPrime :=
  certify 559606337 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2957, p_2957] (by decide)

def p_560150557 : CertifiedPrime :=
  certify 560150557 (2) [p_2, p_2, p_3, p_7, p_7, p_7, p_23, p_61, p_97] (by decide)

def p_560268901 : CertifiedPrime :=
  certify 560268901 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_263, p_263] (by decide)

def p_560434603 : CertifiedPrime :=
  certify 560434603 (2) [p_2, p_3, p_7, p_13, p_19, p_89, p_607] (by decide)

def p_560529301 : CertifiedPrime :=
  certify 560529301 (6) [p_2, p_2, p_3, p_5, p_5, p_947, p_1973] (by decide)

def p_560718721 : CertifiedPrime :=
  certify 560718721 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_37, p_877] (by decide)

def p_560931857 : CertifiedPrime :=
  certify 560931857 (5) [p_2, p_2, p_2, p_2, p_31, p_31, p_191, p_191] (by decide)

def p_560955541 : CertifiedPrime :=
  certify 560955541 (2) [p_2, p_2, p_3, p_5, p_31, p_191, p_1579] (by decide)

def p_561097657 : CertifiedPrime :=
  certify 561097657 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_47, p_23687] (by decide)

def p_561287173 : CertifiedPrime :=
  certify 561287173 (2) [p_2, p_2, p_3, p_53, p_149, p_5923] (by decide)

def p_561500417 : CertifiedPrime :=
  certify 561500417 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1481, p_1481] (by decide)

def p_561524113 : CertifiedPrime :=
  certify 561524113 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_1481, p_2633] (by decide)

def p_562282657 : CertifiedPrime :=
  certify 562282657 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_13, p_19, p_23, p_1031] (by decide)

def p_562448657 : CertifiedPrime :=
  certify 562448657 (5) [p_2, p_2, p_2, p_2, p_7, p_7, p_7, p_7, p_11, p_11, p_11, p_11] (by decide)

def p_562519807 : CertifiedPrime :=
  certify 562519807 (3) [p_2, p_3, p_37, p_59, p_67, p_641] (by decide)

def p_562851901 : CertifiedPrime :=
  certify 562851901 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_73, p_659] (by decide)

def p_562923077 : CertifiedPrime :=
  certify 562923077 (2) [p_2, p_2, p_11863, p_11863] (by decide)

def p_563801281 : CertifiedPrime :=
  certify 563801281 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_53, p_1583] (by decide)

def p_563848771 : CertifiedPrime :=
  certify 563848771 (2) [p_2, p_3, p_5, p_31, p_383, p_1583] (by decide)

def p_564086251 : CertifiedPrime :=
  certify 564086251 (2) [p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_7, p_13, p_19, p_29] (by decide)

def p_564822757 : CertifiedPrime :=
  certify 564822757 (2) [p_2, p_2, p_3, p_3, p_17, p_17, p_233, p_233] (by decide)

def p_565298177 : CertifiedPrime :=
  certify 565298177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_743, p_743] (by decide)

def p_565369507 : CertifiedPrime :=
  certify 565369507 (5) [p_2, p_3, p_3, p_13, p_31, p_59, p_1321] (by decide)

def p_565417063 : CertifiedPrime :=
  certify 565417063 (7) [p_2, p_3, p_3, p_7, p_43, p_79, p_1321] (by decide)

def p_565678657 : CertifiedPrime :=
  certify 565678657 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_991, p_991] (by decide)

def p_565702441 : CertifiedPrime :=
  certify 565702441 (11) [p_2, p_2, p_2, p_3, p_5, p_67, p_71, p_991] (by decide)

def p_565773797 : CertifiedPrime :=
  certify 565773797 (2) [p_2, p_2, p_7, p_7, p_1699, p_1699] (by decide)

def p_565964101 : CertifiedPrime :=
  certify 565964101 (11) [p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13, p_61, p_61] (by decide)

def p_566083057 : CertifiedPrime :=
  certify 566083057 (10) [p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_103, p_1487] (by decide)

def p_566725637 : CertifiedPrime :=
  certify 566725637 (2) [p_2, p_2, p_11903, p_11903] (by decide)

def p_566844673 : CertifiedPrime :=
  certify 566844673 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_29, p_31, p_821] (by decide)

def p_567106597 : CertifiedPrime :=
  certify 567106597 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7, p_7, p_7] (by decide)

def p_567201857 : CertifiedPrime :=
  certify 567201857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_13, p_13, p_229, p_229] (by decide)

def p_568846351 : CertifiedPrime :=
  certify 568846351 (33) [p_2, p_3, p_3, p_5, p_5, p_17, p_23, p_53, p_61] (by decide)

def p_569013317 : CertifiedPrime :=
  certify 569013317 (2) [p_2, p_2, p_11927, p_11927] (by decide)

def p_569371183 : CertifiedPrime :=
  certify 569371183 (3) [p_2, p_3, p_41, p_97, p_107, p_223] (by decide)

def p_569585957 : CertifiedPrime :=
  certify 569585957 (2) [p_2, p_2, p_11933, p_11933] (by decide)

def p_569967877 : CertifiedPrime :=
  certify 569967877 (2) [p_2, p_2, p_3, p_3, p_23, p_23, p_173, p_173] (by decide)

def p_570732101 : CertifiedPrime :=
  certify 570732101 (2) [p_2, p_2, p_5, p_5, p_2389, p_2389] (by decide)

def p_570851557 : CertifiedPrime :=
  certify 570851557 (5) [p_2, p_2, p_3, p_11, p_181, p_23893] (by decide)

def p_571210001 : CertifiedPrime :=
  certify 571210001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_239, p_239] (by decide)

def p_571879397 : CertifiedPrime :=
  certify 571879397 (2) [p_2, p_2, p_11, p_11, p_1087, p_1087] (by decide)

def p_571975057 : CertifiedPrime :=
  certify 571975057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1993, p_1993] (by decide)

def p_572644901 : CertifiedPrime :=
  certify 572644901 (3) [p_2, p_2, p_5, p_5, p_2393, p_2393] (by decide)

def p_572836357 : CertifiedPrime :=
  certify 572836357 (5) [p_2, p_2, p_3, p_3, p_3989, p_3989] (by decide)

def p_572932097 : CertifiedPrime :=
  certify 572932097 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_17, p_17] (by decide)

def p_573099661 : CertifiedPrime :=
  certify 573099661 (29) [p_2, p_2, p_3, p_3, p_5, p_7, p_19, p_37, p_647] (by decide)

def p_573315137 : CertifiedPrime :=
  certify 573315137 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_41, p_41, p_73, p_73] (by decide)

def p_573386971 : CertifiedPrime :=
  certify 573386971 (11) [p_2, p_3, p_5, p_13, p_307, p_4789] (by decide)

def p_573410917 : CertifiedPrime :=
  certify 573410917 (2) [p_2, p_2, p_3, p_3, p_13, p_13, p_307, p_307] (by decide)

def p_573602501 : CertifiedPrime :=
  certify 573602501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_479, p_479] (by decide)

def p_574081601 : CertifiedPrime :=
  certify 574081601 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_599, p_599] (by decide)

def p_574441057 : CertifiedPrime :=
  certify 574441057 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_107, p_2663] (by decide)

def p_574536931 : CertifiedPrime :=
  certify 574536931 (3) [p_2, p_3, p_5, p_11, p_17, p_47, p_2179] (by decide)

def p_574968463 : CertifiedPrime :=
  certify 574968463 (3) [p_2, p_3, p_19, p_631, p_7993] (by decide)

def p_575544091 : CertifiedPrime :=
  certify 575544091 (3) [p_2, p_3, p_5, p_11, p_727, p_2399] (by decide)

def p_575832013 : CertifiedPrime :=
  certify 575832013 (14) [p_2, p_2, p_3, p_7, p_19, p_421, p_857] (by decide)

def p_576192017 : CertifiedPrime :=
  certify 576192017 (3) [p_2, p_2, p_2, p_2, p_17, p_17, p_353, p_353] (by decide)

def p_576552133 : CertifiedPrime :=
  certify 576552133 (17) [p_2, p_2, p_3, p_3, p_13, p_23, p_29, p_1847] (by decide)

def p_576672197 : CertifiedPrime :=
  certify 576672197 (2) [p_2, p_2, p_12007, p_12007] (by decide)

def p_577440901 : CertifiedPrime :=
  certify 577440901 (13) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5, p_89, p_89] (by decide)

def p_577849483 : CertifiedPrime :=
  certify 577849483 (2) [p_2, p_3, p_3, p_7, p_17, p_101, p_2671] (by decide)

def p_578330353 : CertifiedPrime :=
  certify 578330353 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_167, p_24049] (by decide)

def p_578402501 : CertifiedPrime :=
  certify 578402501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_13, p_13, p_37, p_37] (by decide)

def p_578474653 : CertifiedPrime :=
  certify 578474653 (2) [p_2, p_2, p_3, p_7, p_859, p_8017] (by decide)

def p_578763307 : CertifiedPrime :=
  certify 578763307 (2) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_11, p_23, p_523] (by decide)

def p_578883601 : CertifiedPrime :=
  certify 578883601 (19) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_401, p_401] (by decide)

def p_579003907 : CertifiedPrime :=
  certify 579003907 (2) [p_2, p_3, p_13, p_53, p_227, p_617] (by decide)

def p_579172357 : CertifiedPrime :=
  certify 579172357 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_7, p_191, p_191] (by decide)

def p_579364901 : CertifiedPrime :=
  certify 579364901 (2) [p_2, p_2, p_5, p_5, p_29, p_29, p_83, p_83] (by decide)

def p_579581551 : CertifiedPrime :=
  certify 579581551 (6) [p_2, p_3, p_3, p_5, p_5, p_107, p_12037] (by decide)

def p_579774163 : CertifiedPrime :=
  certify 579774163 (3) [p_2, p_3, p_11, p_11, p_199, p_4013] (by decide)

def p_580014973 : CertifiedPrime :=
  certify 580014973 (5) [p_2, p_2, p_3, p_3, p_3, p_223, p_24083] (by decide)

def p_580135397 : CertifiedPrime :=
  certify 580135397 (2) [p_2, p_2, p_12043, p_12043] (by decide)

def p_580617217 : CertifiedPrime :=
  certify 580617217 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_251, p_251] (by decide)

def p_580882303 : CertifiedPrime :=
  certify 580882303 (7) [p_2, p_3, p_3, p_7, p_11, p_13, p_103, p_313] (by decide)

def p_580930507 : CertifiedPrime :=
  certify 580930507 (5) [p_2, p_3, p_3, p_13, p_103, p_24103] (by decide)

def p_581171557 : CertifiedPrime :=
  certify 581171557 (5) [p_2, p_2, p_3, p_7, p_7, p_41, p_24107] (by decide)

def p_581509111 : CertifiedPrime :=
  certify 581509111 (6) [p_2, p_3, p_5, p_7, p_13, p_53, p_4019] (by decide)

def p_581967377 : CertifiedPrime :=
  certify 581967377 (3) [p_2, p_2, p_2, p_2, p_37, p_37, p_163, p_163] (by decide)

def p_582039751 : CertifiedPrime :=
  certify 582039751 (3) [p_2, p_3, p_5, p_5, p_5, p_193, p_4021] (by decide)

def p_582184513 : CertifiedPrime :=
  certify 582184513 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_13, p_29, p_383] (by decide)

def p_582449957 : CertifiedPrime :=
  certify 582449957 (2) [p_2, p_2, p_11, p_11, p_1097, p_1097] (by decide)

def p_583053463 : CertifiedPrime :=
  certify 583053463 (6) [p_2, p_3, p_3, p_2683, p_12073] (by decide)

def p_583391563 : CertifiedPrime :=
  certify 583391563 (2) [p_2, p_3, p_13, p_83, p_97, p_929] (by decide)

def p_583488181 : CertifiedPrime :=
  certify 583488181 (10) [p_2, p_2, p_3, p_3, p_5, p_11, p_61, p_4831] (by decide)

def p_583512337 : CertifiedPrime :=
  certify 583512337 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_11, p_11, p_61, p_61] (by decide)

def p_583826407 : CertifiedPrime :=
  certify 583826407 (3) [p_2, p_3, p_73, p_331, p_4027] (by decide)

def p_583923061 : CertifiedPrime :=
  certify 583923061 (2) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_179, p_863] (by decide)

def p_584068057 : CertifiedPrime :=
  certify 584068057 (5) [p_2, p_2, p_2, p_3, p_11, p_13, p_13, p_13, p_19, p_53] (by decide)

def p_584478977 : CertifiedPrime :=
  certify 584478977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1511, p_1511] (by decide)

def p_584865857 : CertifiedPrime :=
  certify 584865857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_3023, p_3023] (by decide)

def p_584986783 : CertifiedPrime :=
  certify 584986783 (5) [p_2, p_3, p_19, p_19, p_29, p_67, p_139] (by decide)

def p_585228673 : CertifiedPrime :=
  certify 585228673 (19) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_7, p_17, p_1423] (by decide)

def p_585422221 : CertifiedPrime :=
  certify 585422221 (2) [p_2, p_2, p_3, p_5, p_23, p_263, p_1613] (by decide)

def p_585446417 : CertifiedPrime :=
  certify 585446417 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_263, p_263] (by decide)

def p_585664201 : CertifiedPrime :=
  certify 585664201 (17) [p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_11, p_11, p_2689] (by decide)

def p_586003057 : CertifiedPrime :=
  certify 586003057 (5) [p_2, p_2, p_2, p_2, p_3, p_17, p_89, p_8069] (by decide)

def p_586245157 : CertifiedPrime :=
  certify 586245157 (2) [p_2, p_2, p_3, p_7, p_1153, p_6053] (by decide)

def p_586535743 : CertifiedPrime :=
  certify 586535743 (5) [p_2, p_3, p_3, p_3, p_3, p_13, p_23, p_12109] (by decide)

def p_586584181 : CertifiedPrime :=
  certify 586584181 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_7, p_13, p_23, p_173] (by decide)

def p_586681063 : CertifiedPrime :=
  certify 586681063 (3) [p_2, p_3, p_11, p_53, p_367, p_457] (by decide)

def p_586874851 : CertifiedPrime :=
  certify 586874851 (10) [p_2, p_3, p_5, p_5, p_17, p_19, p_12113] (by decide)

def p_587286757 : CertifiedPrime :=
  certify 587286757 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_577, p_577] (by decide)

def p_587553361 : CertifiedPrime :=
  certify 587553361 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_101, p_24239] (by decide)

def p_587601841 : CertifiedPrime :=
  certify 587601841 (13) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_101, p_3463] (by decide)

def p_587698807 : CertifiedPrime :=
  certify 587698807 (3) [p_2, p_3, p_17, p_23, p_31, p_8081] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
