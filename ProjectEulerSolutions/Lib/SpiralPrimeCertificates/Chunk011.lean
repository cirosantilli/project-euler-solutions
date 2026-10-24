import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk010

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_224677 : CertifiedPrime :=
  certify 224677 (2) [p_2, p_2, p_3, p_3, p_79, p_79] (by decide)

def p_227053 : CertifiedPrime :=
  certify 227053 (2) [p_2, p_2, p_3, p_3, p_7, p_17, p_53] (by decide)

def p_239611 : CertifiedPrime :=
  certify 239611 (2) [p_2, p_3, p_5, p_7, p_7, p_163] (by decide)

def p_240101 : CertifiedPrime :=
  certify 240101 (3) [p_2, p_2, p_5, p_5, p_7, p_7, p_7, p_7] (by decide)

def p_245521 : CertifiedPrime :=
  certify 245521 (14) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_11, p_31] (by decide)

def p_246017 : CertifiedPrime :=
  certify 246017 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_31, p_31] (by decide)

def p_250501 : CertifiedPrime :=
  certify 250501 (2) [p_2, p_2, p_3, p_5, p_5, p_5, p_167] (by decide)

def p_262657 : CertifiedPrime :=
  certify 262657 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_19] (by decide)

def p_268843 : CertifiedPrime :=
  certify 268843 (2) [p_2, p_3, p_7, p_37, p_173] (by decide)

def p_276151 : CertifiedPrime :=
  certify 276151 (6) [p_2, p_3, p_5, p_5, p_7, p_263] (by decide)

def p_281431 : CertifiedPrime :=
  certify 281431 (12) [p_2, p_3, p_3, p_5, p_53, p_59] (by decide)

def p_282493 : CertifiedPrime :=
  certify 282493 (2) [p_2, p_2, p_3, p_3, p_7, p_19, p_59] (by decide)

def p_284623 : CertifiedPrime :=
  certify 284623 (3) [p_2, p_3, p_13, p_41, p_89] (by decide)

def p_287297 : CertifiedPrime :=
  certify 287297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_67, p_67] (by decide)

def p_288907 : CertifiedPrime :=
  certify 288907 (2) [p_2, p_3, p_179, p_269] (by decide)

def p_292141 : CertifiedPrime :=
  certify 292141 (7) [p_2, p_2, p_3, p_3, p_3, p_5, p_541] (by decide)

def p_295937 : CertifiedPrime :=
  certify 295937 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_17, p_17] (by decide)

def p_304153 : CertifiedPrime :=
  certify 304153 (7) [p_2, p_2, p_2, p_3, p_19, p_23, p_29] (by decide)

def p_307471 : CertifiedPrime :=
  certify 307471 (3) [p_2, p_3, p_5, p_37, p_277] (by decide)

def p_309137 : CertifiedPrime :=
  certify 309137 (3) [p_2, p_2, p_2, p_2, p_139, p_139] (by decide)

def p_314161 : CertifiedPrime :=
  certify 314161 (37) [p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_17] (by decide)

def p_320923 : CertifiedPrime :=
  certify 320923 (2) [p_2, p_3, p_3, p_3, p_3, p_7, p_283] (by decide)

def p_322057 : CertifiedPrime :=
  certify 322057 (10) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_7, p_71] (by decide)

def p_324901 : CertifiedPrime :=
  certify 324901 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_19, p_19] (by decide)

def p_327757 : CertifiedPrime :=
  certify 327757 (2) [p_2, p_2, p_3, p_11, p_13, p_191] (by decide)

def p_331777 : CertifiedPrime :=
  certify 331777 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3] (by decide)

def p_335821 : CertifiedPrime :=
  certify 335821 (13) [p_2, p_2, p_3, p_5, p_29, p_193] (by decide)

def p_339307 : CertifiedPrime :=
  certify 339307 (3) [p_2, p_3, p_11, p_53, p_97] (by decide)

def p_341057 : CertifiedPrime :=
  certify 341057 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_73, p_73] (by decide)

def p_341641 : CertifiedPrime :=
  certify 341641 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_13, p_73] (by decide)

def p_352837 : CertifiedPrime :=
  certify 352837 (14) [p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_11, p_11] (by decide)

def p_364213 : CertifiedPrime :=
  certify 364213 (2) [p_2, p_2, p_3, p_3, p_67, p_151] (by decide)

def p_366631 : CertifiedPrime :=
  certify 366631 (6) [p_2, p_3, p_5, p_11, p_11, p_101] (by decide)

def p_371491 : CertifiedPrime :=
  certify 371491 (2) [p_2, p_3, p_5, p_7, p_29, p_61] (by decide)

def p_375157 : CertifiedPrime :=
  certify 375157 (5) [p_2, p_2, p_3, p_3, p_17, p_613] (by decide)

def p_386263 : CertifiedPrime :=
  certify 386263 (5) [p_2, p_3, p_3, p_3, p_23, p_311] (by decide)

def p_390001 : CertifiedPrime :=
  certify 390001 (7) [p_2, p_2, p_2, p_2, p_3, p_5, p_5, p_5, p_5, p_13] (by decide)

def p_392503 : CertifiedPrime :=
  certify 392503 (3) [p_2, p_3, p_11, p_19, p_313] (by decide)

def p_401957 : CertifiedPrime :=
  certify 401957 (2) [p_2, p_2, p_317, p_317] (by decide)

def p_403861 : CertifiedPrime :=
  certify 403861 (6) [p_2, p_2, p_3, p_5, p_53, p_127] (by decide)

def p_404497 : CertifiedPrime :=
  certify 404497 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_53, p_53] (by decide)

def p_412807 : CertifiedPrime :=
  certify 412807 (3) [p_2, p_3, p_107, p_643] (by decide)

def p_414737 : CertifiedPrime :=
  certify 414737 (3) [p_2, p_2, p_2, p_2, p_7, p_7, p_23, p_23] (by decide)

def p_415381 : CertifiedPrime :=
  certify 415381 (10) [p_2, p_2, p_3, p_5, p_7, p_23, p_43] (by decide)

def p_417317 : CertifiedPrime :=
  certify 417317 (2) [p_2, p_2, p_17, p_17, p_19, p_19] (by decide)

def p_427717 : CertifiedPrime :=
  certify 427717 (2) [p_2, p_2, p_3, p_3, p_109, p_109] (by decide)

def p_446893 : CertifiedPrime :=
  certify 446893 (2) [p_2, p_2, p_3, p_167, p_223] (by decide)

def p_450913 : CertifiedPrime :=
  certify 450913 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_7, p_11, p_61] (by decide)

def p_454277 : CertifiedPrime :=
  certify 454277 (2) [p_2, p_2, p_337, p_337] (by decide)

def p_459007 : CertifiedPrime :=
  certify 459007 (3) [p_2, p_3, p_113, p_677] (by decide)

def p_462401 : CertifiedPrime :=
  certify 462401 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17] (by decide)

def p_470597 : CertifiedPrime :=
  certify 470597 (2) [p_2, p_2, p_7, p_7, p_7, p_7, p_7, p_7] (by decide)

def p_471283 : CertifiedPrime :=
  certify 471283 (2) [p_2, p_3, p_7, p_7, p_7, p_229] (by decide)

def p_476101 : CertifiedPrime :=
  certify 476101 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_23, p_23] (by decide)

def p_484417 : CertifiedPrime :=
  certify 484417 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_29, p_29] (by decide)

def p_485113 : CertifiedPrime :=
  certify 485113 (5) [p_2, p_2, p_2, p_3, p_17, p_29, p_41] (by decide)

def p_490001 : CertifiedPrime :=
  certify 490001 (6) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_7, p_7] (by decide)

def p_492103 : CertifiedPrime :=
  certify 492103 (23) [p_2, p_3, p_3, p_3, p_13, p_701] (by decide)

def p_495617 : CertifiedPrime :=
  certify 495617 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11] (by decide)

def p_509797 : CertifiedPrime :=
  certify 509797 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_17, p_17] (by decide)

def p_512657 : CertifiedPrime :=
  certify 512657 (3) [p_2, p_2, p_2, p_2, p_179, p_179] (by decide)

def p_519121 : CertifiedPrime :=
  certify 519121 (23) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_103] (by decide)

def p_527803 : CertifiedPrime :=
  certify 527803 (3) [p_2, p_3, p_11, p_11, p_727] (by decide)

def p_530713 : CertifiedPrime :=
  certify 530713 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_7, p_13] (by decide)

def p_540961 : CertifiedPrime :=
  certify 540961 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_7, p_23] (by decide)

def p_547601 : CertifiedPrime :=
  certify 547601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_37, p_37] (by decide)

def p_552793 : CertifiedPrime :=
  certify 552793 (5) [p_2, p_2, p_2, p_3, p_31, p_743] (by decide)

def p_558757 : CertifiedPrime :=
  certify 558757 (5) [p_2, p_2, p_3, p_3, p_11, p_17, p_83] (by decide)

def p_562501 : CertifiedPrime :=
  certify 562501 (14) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_5, p_5] (by decide)

def p_570781 : CertifiedPrime :=
  certify 570781 (17) [p_2, p_2, p_3, p_3, p_3, p_5, p_7, p_151] (by decide)

def p_577601 : CertifiedPrime :=
  certify 577601 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_19, p_19] (by decide)

def p_579883 : CertifiedPrime :=
  certify 579883 (3) [p_2, p_3, p_127, p_761] (by decide)

def p_581407 : CertifiedPrime :=
  certify 581407 (3) [p_2, p_3, p_7, p_109, p_127] (by decide)

def p_583697 : CertifiedPrime :=
  certify 583697 (3) [p_2, p_2, p_2, p_2, p_191, p_191] (by decide)

def p_590593 : CertifiedPrime :=
  certify 590593 (15) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_769] (by decide)

def p_598303 : CertifiedPrime :=
  certify 598303 (5) [p_2, p_3, p_3, p_43, p_773] (by decide)

def p_608401 : CertifiedPrime :=
  certify 608401 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13] (by decide)

def p_612307 : CertifiedPrime :=
  certify 612307 (2) [p_2, p_3, p_3, p_3, p_17, p_23, p_29] (by decide)

def p_614657 : CertifiedPrime :=
  certify 614657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_7, p_7] (by decide)

def p_617011 : CertifiedPrime :=
  certify 617011 (10) [p_2, p_3, p_5, p_131, p_157] (by decide)

def p_628057 : CertifiedPrime :=
  certify 628057 (5) [p_2, p_2, p_2, p_3, p_3, p_11, p_13, p_61] (by decide)

def p_637603 : CertifiedPrime :=
  certify 637603 (3) [p_2, p_3, p_7, p_17, p_19, p_47] (by decide)

def p_642403 : CertifiedPrime :=
  certify 642403 (2) [p_2, p_3, p_3, p_89, p_401] (by decide)

def p_660157 : CertifiedPrime :=
  certify 660157 (2) [p_2, p_2, p_3, p_7, p_29, p_271] (by decide)

def p_665857 : CertifiedPrime :=
  certify 665857 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_17, p_17] (by decide)

def p_669943 : CertifiedPrime :=
  certify 669943 (6) [p_2, p_3, p_3, p_7, p_13, p_409] (by decide)

def p_671581 : CertifiedPrime :=
  certify 671581 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_13, p_41] (by decide)

def p_681451 : CertifiedPrime :=
  certify 681451 (2) [p_2, p_3, p_5, p_5, p_7, p_11, p_59] (by decide)

def p_682277 : CertifiedPrime :=
  certify 682277 (2) [p_2, p_2, p_7, p_7, p_59, p_59] (by decide)

def p_684757 : CertifiedPrime :=
  certify 684757 (5) [p_2, p_2, p_3, p_3, p_23, p_827] (by decide)

def p_699733 : CertifiedPrime :=
  certify 699733 (2) [p_2, p_2, p_3, p_3, p_3, p_11, p_19, p_31] (by decide)

def p_704761 : CertifiedPrime :=
  certify 704761 (19) [p_2, p_2, p_2, p_3, p_5, p_7, p_839] (by decide)

def p_716563 : CertifiedPrime :=
  certify 716563 (7) [p_2, p_3, p_3, p_7, p_11, p_11, p_47] (by decide)

def p_731881 : CertifiedPrime :=
  certify 731881 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_19, p_107] (by decide)

def p_735307 : CertifiedPrime :=
  certify 735307 (5) [p_2, p_3, p_11, p_13, p_857] (by decide)

def p_739601 : CertifiedPrime :=
  certify 739601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_43, p_43] (by decide)

def p_740461 : CertifiedPrime :=
  certify 740461 (2) [p_2, p_2, p_3, p_5, p_7, p_41, p_43] (by decide)

def p_746497 : CertifiedPrime :=
  certify 746497 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3] (by decide)

def p_747361 : CertifiedPrime :=
  certify 747361 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_5, p_173] (by decide)

def p_766501 : CertifiedPrime :=
  certify 766501 (6) [p_2, p_2, p_3, p_5, p_5, p_5, p_7, p_73] (by decide)

def p_771763 : CertifiedPrime :=
  certify 771763 (2) [p_2, p_3, p_293, p_439] (by decide)

def p_792101 : CertifiedPrime :=
  certify 792101 (2) [p_2, p_2, p_5, p_5, p_89, p_89] (by decide)

def p_792991 : CertifiedPrime :=
  certify 792991 (6) [p_2, p_3, p_3, p_3, p_3, p_5, p_11, p_89] (by decide)

def p_800131 : CertifiedPrime :=
  certify 800131 (3) [p_2, p_3, p_5, p_149, p_179] (by decide)

def p_805507 : CertifiedPrime :=
  certify 805507 (2) [p_2, p_3, p_13, p_23, p_449] (by decide)

def p_809101 : CertifiedPrime :=
  certify 809101 (7) [p_2, p_2, p_3, p_3, p_5, p_5, p_29, p_31] (by decide)

def p_820837 : CertifiedPrime :=
  certify 820837 (5) [p_2, p_2, p_3, p_3, p_151, p_151] (by decide)

def p_828101 : CertifiedPrime :=
  certify 828101 (2) [p_2, p_2, p_5, p_5, p_7, p_7, p_13, p_13] (by decide)

def p_830833 : CertifiedPrime :=
  certify 830833 (15) [p_2, p_2, p_2, p_2, p_3, p_19, p_911] (by decide)

def p_838141 : CertifiedPrime :=
  certify 838141 (2) [p_2, p_2, p_3, p_5, p_61, p_229] (by decide)

def p_843643 : CertifiedPrime :=
  certify 843643 (13) [p_2, p_3, p_3, p_3, p_17, p_919] (by decide)

def p_846401 : CertifiedPrime :=
  certify 846401 (6) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_23, p_23] (by decide)

def p_847321 : CertifiedPrime :=
  certify 847321 (7) [p_2, p_2, p_2, p_3, p_5, p_23, p_307] (by decide)

def p_860257 : CertifiedPrime :=
  certify 860257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_29, p_103] (by decide)

def p_864901 : CertifiedPrime :=
  certify 864901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_31, p_31] (by decide)

def p_876097 : CertifiedPrime :=
  certify 876097 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13] (by decide)

def p_894917 : CertifiedPrime :=
  certify 894917 (2) [p_2, p_2, p_11, p_11, p_43, p_43] (by decide)

def p_902501 : CertifiedPrime :=
  certify 902501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_19, p_19] (by decide)

def p_903451 : CertifiedPrime :=
  certify 903451 (2) [p_2, p_3, p_5, p_5, p_19, p_317] (by decide)

def p_920641 : CertifiedPrime :=
  certify 920641 (23) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_137] (by decide)

def p_921601 : CertifiedPrime :=
  certify 921601 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5] (by decide)

def p_922561 : CertifiedPrime :=
  certify 922561 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_31, p_31] (by decide)

def p_933157 : CertifiedPrime :=
  certify 933157 (17) [p_2, p_2, p_3, p_3, p_7, p_7, p_23, p_23] (by decide)

def p_939931 : CertifiedPrime :=
  certify 939931 (3) [p_2, p_3, p_5, p_17, p_19, p_97] (by decide)

def p_949651 : CertifiedPrime :=
  certify 949651 (2) [p_2, p_3, p_5, p_5, p_13, p_487] (by decide)

def p_963343 : CertifiedPrime :=
  certify 963343 (5) [p_2, p_3, p_3, p_109, p_491] (by decide)

def p_972197 : CertifiedPrime :=
  certify 972197 (3) [p_2, p_2, p_17, p_17, p_29, p_29] (by decide)

def p_975157 : CertifiedPrime :=
  certify 975157 (2) [p_2, p_2, p_3, p_7, p_13, p_19, p_47] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
