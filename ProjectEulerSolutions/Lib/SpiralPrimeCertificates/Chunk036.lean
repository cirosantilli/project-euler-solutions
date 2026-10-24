import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk035

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
set_option maxRecDepth 4096

namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_252825901 : CertifiedPrime :=
  certify 252825901 (6) [p_2, p_2, p_3, p_5, p_5, p_53, p_15901] (by decide)

def p_253175833 : CertifiedPrime :=
  certify 253175833 (23) [p_2, p_2, p_2, p_3, p_3, p_7, p_13, p_17, p_2273] (by decide)

def p_253255397 : CertifiedPrime :=
  certify 253255397 (2) [p_2, p_2, p_73, p_73, p_109, p_109] (by decide)

def p_253557853 : CertifiedPrime :=
  certify 253557853 (5) [p_2, p_2, p_3, p_1327, p_15923] (by decide)

def p_253573777 : CertifiedPrime :=
  certify 253573777 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_1327, p_1327] (by decide)

def p_253653403 : CertifiedPrime :=
  certify 253653403 (2) [p_2, p_3, p_5309, p_7963] (by decide)

def p_253956097 : CertifiedPrime :=
  certify 253956097 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_83, p_83] (by decide)

def p_254211137 : CertifiedPrime :=
  certify 254211137 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_1993, p_1993] (by decide)

def p_254227081 : CertifiedPrime :=
  certify 254227081 (7) [p_2, p_2, p_2, p_3, p_5, p_1063, p_1993] (by decide)

def p_254514163 : CertifiedPrime :=
  certify 254514163 (5) [p_2, p_3, p_7, p_43, p_53, p_2659] (by decide)

def p_254530117 : CertifiedPrime :=
  certify 254530117 (2) [p_2, p_2, p_3, p_3, p_2659, p_2659] (by decide)

def p_254833333 : CertifiedPrime :=
  certify 254833333 (2) [p_2, p_2, p_3, p_13, p_17, p_307, p_313] (by decide)

def p_254929123 : CertifiedPrime :=
  certify 254929123 (5) [p_2, p_3, p_3, p_7, p_887, p_2281] (by decide)

def p_255120757 : CertifiedPrime :=
  certify 255120757 (5) [p_2, p_2, p_3, p_11, p_11, p_11, p_15973] (by decide)

def p_255184651 : CertifiedPrime :=
  certify 255184651 (2) [p_2, p_3, p_3, p_5, p_5, p_7, p_7, p_71, p_163] (by decide)

def p_255360401 : CertifiedPrime :=
  certify 255360401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_47, p_47] (by decide)

def p_255504241 : CertifiedPrime :=
  certify 255504241 (14) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_23, p_37, p_139] (by decide)

def p_255600157 : CertifiedPrime :=
  certify 255600157 (2) [p_2, p_2, p_3, p_7, p_73, p_73, p_571] (by decide)

def p_255888013 : CertifiedPrime :=
  certify 255888013 (5) [p_2, p_2, p_3, p_17, p_31, p_43, p_941] (by decide)

def p_255952003 : CertifiedPrime :=
  certify 255952003 (2) [p_2, p_3, p_19, p_421, p_5333] (by decide)

def p_256000001 : CertifiedPrime :=
  certify 256000001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5] (by decide)

def p_256128017 : CertifiedPrime :=
  certify 256128017 (3) [p_2, p_2, p_2, p_2, p_4001, p_4001] (by decide)

def p_256240057 : CertifiedPrime :=
  certify 256240057 (5) [p_2, p_2, p_2, p_3, p_23, p_29, p_16007] (by decide)

def p_256272073 : CertifiedPrime :=
  certify 256272073 (11) [p_2, p_2, p_2, p_3, p_7, p_23, p_29, p_2287] (by decide)

def p_256464211 : CertifiedPrime :=
  certify 256464211 (3) [p_2, p_3, p_5, p_17, p_157, p_3203] (by decide)

def p_256752553 : CertifiedPrime :=
  certify 256752553 (10) [p_2, p_2, p_2, p_3, p_7, p_7, p_109, p_2003] (by decide)

def p_256832677 : CertifiedPrime :=
  certify 256832677 (6) [p_2, p_2, p_3, p_3, p_2671, p_2671] (by decide)

def p_256960901 : CertifiedPrime :=
  certify 256960901 (11) [p_2, p_2, p_5, p_5, p_7, p_7, p_229, p_229] (by decide)

def p_257137261 : CertifiedPrime :=
  certify 257137261 (2) [p_2, p_2, p_3, p_5, p_19, p_211, p_1069] (by decide)

def p_257201407 : CertifiedPrime :=
  certify 257201407 (5) [p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_11, p_29, p_79] (by decide)

def p_257425981 : CertifiedPrime :=
  certify 257425981 (10) [p_2, p_2, p_3, p_5, p_7, p_191, p_3209] (by decide)

def p_257474117 : CertifiedPrime :=
  certify 257474117 (2) [p_2, p_2, p_71, p_71, p_113, p_113] (by decide)

def p_257586451 : CertifiedPrime :=
  certify 257586451 (11) [p_2, p_3, p_5, p_5, p_11, p_107, p_1459] (by decide)

def p_257907541 : CertifiedPrime :=
  certify 257907541 (2) [p_2, p_2, p_3, p_5, p_11, p_53, p_73, p_101] (by decide)

def p_258260971 : CertifiedPrime :=
  certify 258260971 (2) [p_2, p_3, p_5, p_11, p_487, p_1607] (by decide)

def p_258357403 : CertifiedPrime :=
  certify 258357403 (5) [p_2, p_3, p_3, p_19, p_47, p_16073] (by decide)

def p_258646807 : CertifiedPrime :=
  certify 258646807 (6) [p_2, p_3, p_3, p_11, p_17, p_43, p_1787] (by decide)

def p_258695057 : CertifiedPrime :=
  certify 258695057 (3) [p_2, p_2, p_2, p_2, p_4021, p_4021] (by decide)

def p_259161703 : CertifiedPrime :=
  certify 259161703 (3) [p_2, p_3, p_17, p_947, p_2683] (by decide)

def p_259258303 : CertifiedPrime :=
  certify 259258303 (5) [p_2, p_3, p_3, p_83, p_97, p_1789] (by decide)

def p_259419343 : CertifiedPrime :=
  certify 259419343 (3) [p_2, p_3, p_7, p_13, p_59, p_8053] (by decide)

def p_259532101 : CertifiedPrime :=
  certify 259532101 (7) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_179, p_179] (by decide)

def p_259612657 : CertifiedPrime :=
  certify 259612657 (5) [p_2, p_2, p_2, p_2, p_3, p_19, p_41, p_53, p_131] (by decide)

def p_259644883 : CertifiedPrime :=
  certify 259644883 (2) [p_2, p_3, p_7, p_41, p_131, p_1151] (by decide)

def p_259709341 : CertifiedPrime :=
  certify 259709341 (6) [p_2, p_2, p_3, p_5, p_11, p_17, p_79, p_293] (by decide)

def p_259725457 : CertifiedPrime :=
  certify 259725457 (10) [p_2, p_2, p_2, p_2, p_3, p_3, p_17, p_17, p_79, p_79] (by decide)

def p_260176901 : CertifiedPrime :=
  certify 260176901 (2) [p_2, p_2, p_5, p_5, p_1613, p_1613] (by decide)

def p_260305957 : CertifiedPrime :=
  certify 260305957 (2) [p_2, p_2, p_3, p_3, p_2689, p_2689] (by decide)

def p_260322091 : CertifiedPrime :=
  certify 260322091 (3) [p_2, p_3, p_5, p_7, p_461, p_2689] (by decide)

def p_260515741 : CertifiedPrime :=
  certify 260515741 (6) [p_2, p_2, p_3, p_5, p_269, p_16141] (by decide)

def p_260967871 : CertifiedPrime :=
  certify 260967871 (6) [p_2, p_3, p_3, p_5, p_41, p_197, p_359] (by decide)

def p_261016337 : CertifiedPrime :=
  certify 261016337 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_577, p_577] (by decide)

def p_261485071 : CertifiedPrime :=
  certify 261485071 (3) [p_2, p_3, p_5, p_7, p_7, p_11, p_103, p_157] (by decide)

def p_261582103 : CertifiedPrime :=
  certify 261582103 (6) [p_2, p_3, p_3, p_3, p_599, p_8087] (by decide)

def p_261921857 : CertifiedPrime :=
  certify 261921857 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_17, p_17, p_17, p_17] (by decide)

def p_261970411 : CertifiedPrime :=
  certify 261970411 (2) [p_2, p_3, p_5, p_13, p_83, p_8093] (by decide)

def p_262132291 : CertifiedPrime :=
  certify 262132291 (18) [p_2, p_3, p_3, p_5, p_7, p_257, p_1619] (by decide)

def p_262326613 : CertifiedPrime :=
  certify 262326613 (2) [p_2, p_2, p_3, p_4049, p_5399] (by decide)

def p_262553413 : CertifiedPrime :=
  certify 262553413 (2) [p_2, p_2, p_3, p_11, p_491, p_4051] (by decide)

def p_262650643 : CertifiedPrime :=
  certify 262650643 (3) [p_2, p_3, p_19, p_37, p_73, p_853] (by decide)

def p_262715473 : CertifiedPrime :=
  certify 262715473 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1013, p_1801] (by decide)

def p_263088401 : CertifiedPrime :=
  certify 263088401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_811, p_811] (by decide)

def p_263218177 : CertifiedPrime :=
  certify 263218177 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13, p_13, p_13] (by decide)

def p_263234401 : CertifiedPrime :=
  certify 263234401 (21) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_11, p_13, p_13, p_59] (by decide)

def p_263283077 : CertifiedPrime :=
  certify 263283077 (2) [p_2, p_2, p_7, p_7, p_19, p_19, p_61, p_61] (by decide)

def p_263396671 : CertifiedPrime :=
  certify 263396671 (3) [p_2, p_3, p_5, p_541, p_16229] (by decide)

def p_263412901 : CertifiedPrime :=
  certify 263412901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_541, p_541] (by decide)

def p_263737601 : CertifiedPrime :=
  certify 263737601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_7, p_7, p_29, p_29] (by decide)

def p_263867537 : CertifiedPrime :=
  certify 263867537 (3) [p_2, p_2, p_2, p_2, p_31, p_31, p_131, p_131] (by decide)

def p_264013753 : CertifiedPrime :=
  certify 264013753 (7) [p_2, p_2, p_2, p_3, p_677, p_16249] (by decide)

def p_264062501 : CertifiedPrime :=
  certify 264062501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_5, p_5, p_5, p_5, p_13, p_13] (by decide)

def p_264192517 : CertifiedPrime :=
  certify 264192517 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_7, p_43, p_43] (by decide)

def p_264208771 : CertifiedPrime :=
  certify 264208771 (10) [p_2, p_3, p_3, p_3, p_5, p_7, p_43, p_3251] (by decide)

def p_264306307 : CertifiedPrime :=
  certify 264306307 (2) [p_2, p_3, p_11, p_739, p_5419] (by decide)

def p_264387601 : CertifiedPrime :=
  certify 264387601 (34) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_271, p_271] (by decide)

def p_264468907 : CertifiedPrime :=
  certify 264468907 (2) [p_2, p_3, p_3, p_13, p_47, p_139, p_173] (by decide)

def p_264664093 : CertifiedPrime :=
  certify 264664093 (2) [p_2, p_2, p_3, p_7, p_7, p_11, p_17, p_29, p_83] (by decide)

def p_264989563 : CertifiedPrime :=
  certify 264989563 (3) [p_2, p_3, p_73, p_223, p_2713] (by decide)

def p_265087243 : CertifiedPrime :=
  certify 265087243 (2) [p_2, p_3, p_3, p_3, p_3, p_3, p_7, p_67, p_1163] (by decide)

def p_265152373 : CertifiedPrime :=
  certify 265152373 (5) [p_2, p_2, p_3, p_19, p_23, p_59, p_857] (by decide)

def p_265559617 : CertifiedPrime :=
  certify 265559617 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_97, p_97] (by decide)

def p_265673701 : CertifiedPrime :=
  certify 265673701 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_163, p_1811] (by decide)

def p_265738903 : CertifiedPrime :=
  certify 265738903 (3) [p_2, p_3, p_11, p_13, p_19, p_16301] (by decide)

def p_265771507 : CertifiedPrime :=
  certify 265771507 (5) [p_2, p_3, p_7, p_11, p_13, p_17, p_19, p_137] (by decide)

def p_265820417 : CertifiedPrime :=
  certify 265820417 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_1019, p_1019] (by decide)

def p_266146597 : CertifiedPrime :=
  certify 266146597 (2) [p_2, p_2, p_3, p_3, p_2719, p_2719] (by decide)

def p_266228173 : CertifiedPrime :=
  certify 266228173 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_37, p_4079] (by decide)

def p_266260807 : CertifiedPrime :=
  certify 266260807 (5) [p_2, p_3, p_3, p_7, p_7, p_37, p_41, p_199] (by decide)

def p_266424007 : CertifiedPrime :=
  certify 266424007 (3) [p_2, p_3, p_5441, p_8161] (by decide)

def p_266668901 : CertifiedPrime :=
  certify 266668901 (2) [p_2, p_2, p_5, p_5, p_23, p_23, p_71, p_71] (by decide)

def p_267404257 : CertifiedPrime :=
  certify 267404257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_23, p_73, p_79] (by decide)

def p_267780497 : CertifiedPrime :=
  certify 267780497 (3) [p_2, p_2, p_2, p_2, p_4091, p_4091] (by decide)

def p_267829591 : CertifiedPrime :=
  certify 267829591 (6) [p_2, p_3, p_5, p_7, p_7, p_167, p_1091] (by decide)

def p_267895057 : CertifiedPrime :=
  certify 267895057 (10) [p_2, p_2, p_2, p_2, p_3, p_11, p_13, p_31, p_1259] (by decide)

def p_268026013 : CertifiedPrime :=
  certify 268026013 (2) [p_2, p_2, p_3, p_3, p_17, p_107, p_4093] (by decide)

def p_268107877 : CertifiedPrime :=
  certify 268107877 (13) [p_2, p_2, p_3, p_3, p_2729, p_2729] (by decide)

def p_268288021 : CertifiedPrime :=
  certify 268288021 (22) [p_2, p_2, p_3, p_3, p_5, p_7, p_11, p_13, p_1489] (by decide)

def p_268304401 : CertifiedPrime :=
  certify 268304401 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_7, p_13, p_13] (by decide)

def p_268763237 : CertifiedPrime :=
  certify 268763237 (2) [p_2, p_2, p_7, p_7, p_1171, p_1171] (by decide)

def p_268828817 : CertifiedPrime :=
  certify 268828817 (3) [p_2, p_2, p_2, p_2, p_4099, p_4099] (by decide)

def p_268976401 : CertifiedPrime :=
  certify 268976401 (19) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_7, p_11, p_41, p_71] (by decide)

def p_269091217 : CertifiedPrime :=
  certify 269091217 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_1367, p_1367] (by decide)

def p_269156837 : CertifiedPrime :=
  certify 269156837 (2) [p_2, p_2, p_13, p_13, p_631, p_631] (by decide)

def p_269173243 : CertifiedPrime :=
  certify 269173243 (2) [p_2, p_3, p_3, p_13, p_631, p_1823] (by decide)

def p_269485057 : CertifiedPrime :=
  certify 269485057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_19, p_19] (by decide)

def p_269567143 : CertifiedPrime :=
  certify 269567143 (3) [p_2, p_3, p_13, p_421, p_8209] (by decide)

def p_269665663 : CertifiedPrime :=
  certify 269665663 (3) [p_2, p_3, p_7, p_17, p_23, p_16421] (by decide)

def p_269797051 : CertifiedPrime :=
  certify 269797051 (2) [p_2, p_3, p_3, p_5, p_5, p_43, p_73, p_191] (by decide)

def p_270092791 : CertifiedPrime :=
  certify 270092791 (13) [p_2, p_3, p_3, p_5, p_11, p_19, p_83, p_173] (by decide)

def p_270142097 : CertifiedPrime :=
  certify 270142097 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_587, p_587] (by decide)

def p_270191407 : CertifiedPrime :=
  certify 270191407 (3) [p_2, p_3, p_5479, p_8219] (by decide)

def p_270470917 : CertifiedPrime :=
  certify 270470917 (2) [p_2, p_2, p_3, p_3, p_2741, p_2741] (by decide)

def p_271046833 : CertifiedPrime :=
  certify 271046833 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_7, p_7, p_101, p_163] (by decide)

def p_271079761 : CertifiedPrime :=
  certify 271079761 (11) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_7, p_37, p_89] (by decide)

def p_271277371 : CertifiedPrime :=
  certify 271277371 (7) [p_2, p_3, p_3, p_3, p_5, p_7, p_13, p_61, p_181] (by decide)

def p_271573921 : CertifiedPrime :=
  certify 271573921 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_103, p_1831] (by decide)

def p_271639843 : CertifiedPrime :=
  certify 271639843 (3) [p_2, p_3, p_41, p_67, p_16481] (by decide)

def p_272250001 : CertifiedPrime :=
  certify 272250001 (19) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_5, p_5, p_11, p_11] (by decide)

def p_272580101 : CertifiedPrime :=
  certify 272580101 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_127, p_127] (by decide)

def p_272728711 : CertifiedPrime :=
  certify 272728711 (6) [p_2, p_3, p_3, p_5, p_23, p_359, p_367] (by decide)

def p_273026053 : CertifiedPrime :=
  certify 273026053 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_13, p_17, p_31, p_41] (by decide)

def p_273158257 : CertifiedPrime :=
  certify 273158257 (13) [p_2, p_2, p_2, p_2, p_3, p_7, p_787, p_1033] (by decide)

def p_273240901 : CertifiedPrime :=
  certify 273240901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_19, p_29, p_29] (by decide)

def p_273257431 : CertifiedPrime :=
  certify 273257431 (12) [p_2, p_3, p_5, p_19, p_29, p_61, p_271] (by decide)

def p_273356623 : CertifiedPrime :=
  certify 273356623 (6) [p_2, p_3, p_3, p_7, p_11, p_167, p_1181] (by decide)

def p_273422761 : CertifiedPrime :=
  certify 273422761 (11) [p_2, p_2, p_2, p_3, p_5, p_13, p_53, p_3307] (by decide)

def p_273439297 : CertifiedPrime :=
  certify 273439297 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_13, p_13, p_53, p_53] (by decide)

def p_274018363 : CertifiedPrime :=
  certify 274018363 (3) [p_2, p_3, p_31, p_89, p_16553] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
