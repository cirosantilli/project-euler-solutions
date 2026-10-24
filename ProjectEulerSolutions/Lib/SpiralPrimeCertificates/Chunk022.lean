import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk021

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_43243777 : CertifiedPrime :=
  certify 43243777 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_137, p_137] (by decide)

def p_43250353 : CertifiedPrime :=
  certify 43250353 (5) [p_2, p_2, p_2, p_2, p_3, p_137, p_6577] (by decide)

def p_43289821 : CertifiedPrime :=
  certify 43289821 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_17, p_43, p_47] (by decide)

def p_43428101 : CertifiedPrime :=
  certify 43428101 (2) [p_2, p_2, p_5, p_5, p_659, p_659] (by decide)

def p_43434691 : CertifiedPrime :=
  certify 43434691 (2) [p_2, p_3, p_5, p_13, p_13, p_13, p_659] (by decide)

def p_43474243 : CertifiedPrime :=
  certify 43474243 (3) [p_2, p_3, p_7, p_19, p_157, p_347] (by decide)

def p_43553401 : CertifiedPrime :=
  certify 43553401 (7) [p_2, p_2, p_2, p_3, p_5, p_5, p_11, p_6599] (by decide)

def p_43593007 : CertifiedPrime :=
  certify 43593007 (3) [p_2, p_3, p_31, p_71, p_3301] (by decide)

def p_43612817 : CertifiedPrime :=
  certify 43612817 (3) [p_2, p_2, p_2, p_2, p_13, p_13, p_127, p_127] (by decide)

def p_43744997 : CertifiedPrime :=
  certify 43744997 (2) [p_2, p_2, p_3307, p_3307] (by decide)

def p_43751611 : CertifiedPrime :=
  certify 43751611 (12) [p_2, p_3, p_3, p_3, p_5, p_7, p_7, p_3307] (by decide)

def p_43923757 : CertifiedPrime :=
  certify 43923757 (2) [p_2, p_2, p_3, p_47, p_47, p_1657] (by decide)

def p_43990057 : CertifiedPrime :=
  certify 43990057 (5) [p_2, p_2, p_2, p_3, p_3, p_11, p_67, p_829] (by decide)

def p_44036497 : CertifiedPrime :=
  certify 44036497 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_79, p_79] (by decide)

def p_44043133 : CertifiedPrime :=
  certify 44043133 (5) [p_2, p_2, p_3, p_7, p_79, p_6637] (by decide)

def p_44149381 : CertifiedPrime :=
  certify 44149381 (17) [p_2, p_2, p_3, p_5, p_11, p_151, p_443] (by decide)

def p_44169317 : CertifiedPrime :=
  certify 44169317 (2) [p_2, p_2, p_3323, p_3323] (by decide)

def p_44269063 : CertifiedPrime :=
  certify 44269063 (3) [p_2, p_3, p_1109, p_6653] (by decide)

def p_44482231 : CertifiedPrime :=
  certify 44482231 (6) [p_2, p_3, p_3, p_3, p_5, p_13, p_19, p_23, p_29] (by decide)

def p_44642443 : CertifiedPrime :=
  certify 44642443 (2) [p_2, p_3, p_13, p_17, p_131, p_257] (by decide)

def p_44789557 : CertifiedPrime :=
  certify 44789557 (2) [p_2, p_2, p_3, p_7, p_23, p_97, p_239] (by decide)

def p_44923507 : CertifiedPrime :=
  certify 44923507 (3) [p_2, p_3, p_1117, p_6703] (by decide)

def p_44943617 : CertifiedPrime :=
  certify 44943617 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_419, p_419] (by decide)

def p_45024101 : CertifiedPrime :=
  certify 45024101 (2) [p_2, p_2, p_5, p_5, p_11, p_11, p_61, p_61] (by decide)

def p_45077797 : CertifiedPrime :=
  certify 45077797 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_373, p_373] (by decide)

def p_45151681 : CertifiedPrime :=
  certify 45151681 (61) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_6719] (by decide)

def p_45165121 : CertifiedPrime :=
  certify 45165121 (23) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_13, p_47] (by decide)

def p_45212177 : CertifiedPrime :=
  certify 45212177 (3) [p_2, p_2, p_2, p_2, p_41, p_41, p_41, p_41] (by decide)

def p_45232351 : CertifiedPrime :=
  certify 45232351 (3) [p_2, p_3, p_5, p_5, p_19, p_59, p_269] (by decide)

def p_45346757 : CertifiedPrime :=
  certify 45346757 (2) [p_2, p_2, p_7, p_7, p_13, p_13, p_37, p_37] (by decide)

def p_45353491 : CertifiedPrime :=
  certify 45353491 (2) [p_2, p_3, p_5, p_7, p_13, p_37, p_449] (by decide)

def p_45434341 : CertifiedPrime :=
  certify 45434341 (2) [p_2, p_2, p_3, p_3, p_5, p_7, p_107, p_337] (by decide)

def p_45555751 : CertifiedPrime :=
  certify 45555751 (57) [p_2, p_3, p_3, p_3, p_5, p_5, p_5, p_17, p_397] (by decide)

def p_45717883 : CertifiedPrime :=
  certify 45717883 (3) [p_2, p_3, p_7, p_7, p_23, p_6761] (by decide)

def p_45751697 : CertifiedPrime :=
  certify 45751697 (3) [p_2, p_2, p_2, p_2, p_19, p_19, p_89, p_89] (by decide)

def p_45799057 : CertifiedPrime :=
  certify 45799057 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_47, p_67, p_101] (by decide)

def p_45893851 : CertifiedPrime :=
  certify 45893851 (3) [p_2, p_3, p_5, p_5, p_271, p_1129] (by decide)

def p_45914177 : CertifiedPrime :=
  certify 45914177 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_11, p_11, p_11, p_11] (by decide)

def p_45968401 : CertifiedPrime :=
  certify 45968401 (21) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_113, p_113] (by decide)

def p_46022657 : CertifiedPrime :=
  certify 46022657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_53, p_53] (by decide)

def p_46049797 : CertifiedPrime :=
  certify 46049797 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_13, p_13, p_29, p_29] (by decide)

def p_46083733 : CertifiedPrime :=
  certify 46083733 (2) [p_2, p_2, p_3, p_31, p_73, p_1697] (by decide)

def p_46124473 : CertifiedPrime :=
  certify 46124473 (13) [p_2, p_2, p_2, p_3, p_283, p_6791] (by decide)

def p_46219603 : CertifiedPrime :=
  certify 46219603 (3) [p_2, p_3, p_11, p_13, p_103, p_523] (by decide)

def p_46240001 : CertifiedPrime :=
  certify 46240001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_17, p_17] (by decide)

def p_46321637 : CertifiedPrime :=
  certify 46321637 (2) [p_2, p_2, p_41, p_41, p_83, p_83] (by decide)

def p_46342057 : CertifiedPrime :=
  certify 46342057 (5) [p_2, p_2, p_2, p_3, p_23, p_37, p_2269] (by decide)

def p_46382911 : CertifiedPrime :=
  certify 46382911 (12) [p_2, p_3, p_5, p_7, p_7, p_139, p_227] (by decide)

def p_46410157 : CertifiedPrime :=
  certify 46410157 (2) [p_2, p_2, p_3, p_3, p_13, p_131, p_757] (by decide)

def p_46464673 : CertifiedPrime :=
  certify 46464673 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_17, p_71, p_401] (by decide)

def p_46566977 : CertifiedPrime :=
  certify 46566977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_853, p_853] (by decide)

def p_46594277 : CertifiedPrime :=
  certify 46594277 (2) [p_2, p_2, p_3413, p_3413] (by decide)

def p_46696723 : CertifiedPrime :=
  certify 46696723 (3) [p_2, p_3, p_17, p_67, p_6833] (by decide)

def p_46751407 : CertifiedPrime :=
  certify 46751407 (3) [p_2, p_3, p_13, p_43, p_53, p_263] (by decide)

def p_46819807 : CertifiedPrime :=
  certify 46819807 (5) [p_2, p_3, p_11, p_311, p_2281] (by decide)

def p_46922501 : CertifiedPrime :=
  certify 46922501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_137, p_137] (by decide)

def p_46977317 : CertifiedPrime :=
  certify 46977317 (2) [p_2, p_2, p_23, p_23, p_149, p_149] (by decide)

def p_47141957 : CertifiedPrime :=
  certify 47141957 (2) [p_2, p_2, p_3433, p_3433] (by decide)

def p_47251877 : CertifiedPrime :=
  certify 47251877 (2) [p_2, p_2, p_7, p_7, p_491, p_491] (by decide)

def p_47272501 : CertifiedPrime :=
  certify 47272501 (22) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_11, p_191] (by decide)

def p_47368807 : CertifiedPrime :=
  certify 47368807 (3) [p_2, p_3, p_31, p_37, p_6883] (by decide)

def p_47389457 : CertifiedPrime :=
  certify 47389457 (3) [p_2, p_2, p_2, p_2, p_1721, p_1721] (by decide)

def p_47396341 : CertifiedPrime :=
  certify 47396341 (6) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_17, p_1721] (by decide)

def p_47451433 : CertifiedPrime :=
  certify 47451433 (15) [p_2, p_2, p_2, p_3, p_7, p_41, p_83, p_83] (by decide)

def p_47685931 : CertifiedPrime :=
  certify 47685931 (3) [p_2, p_3, p_5, p_1151, p_1381] (by decide)

def p_47741191 : CertifiedPrime :=
  certify 47741191 (11) [p_2, p_3, p_5, p_7, p_7, p_47, p_691] (by decide)

def p_47748101 : CertifiedPrime :=
  certify 47748101 (2) [p_2, p_2, p_5, p_5, p_691, p_691] (by decide)

def p_47810311 : CertifiedPrime :=
  certify 47810311 (3) [p_2, p_3, p_5, p_461, p_3457] (by decide)

def p_47824141 : CertifiedPrime :=
  certify 47824141 (2) [p_2, p_2, p_3, p_5, p_7, p_13, p_19, p_461] (by decide)

def p_47851807 : CertifiedPrime :=
  certify 47851807 (3) [p_2, p_3, p_1153, p_6917] (by decide)

def p_47907163 : CertifiedPrime :=
  certify 47907163 (2) [p_2, p_3, p_3, p_769, p_3461] (by decide)

def p_47969477 : CertifiedPrime :=
  certify 47969477 (2) [p_2, p_2, p_3463, p_3463] (by decide)

def p_48024901 : CertifiedPrime :=
  certify 48024901 (2) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_7, p_7, p_11, p_11] (by decide)

def p_48219137 : CertifiedPrime :=
  certify 48219137 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_7, p_7, p_31, p_31] (by decide)

def p_48226081 : CertifiedPrime :=
  certify 48226081 (26) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_31, p_463] (by decide)

def p_48365071 : CertifiedPrime :=
  certify 48365071 (3) [p_2, p_3, p_5, p_13, p_19, p_61, p_107] (by decide)

def p_48385937 : CertifiedPrime :=
  certify 48385937 (5) [p_2, p_2, p_2, p_2, p_37, p_37, p_47, p_47] (by decide)

def p_48392893 : CertifiedPrime :=
  certify 48392893 (2) [p_2, p_2, p_3, p_3, p_37, p_47, p_773] (by decide)

def p_48490333 : CertifiedPrime :=
  certify 48490333 (2) [p_2, p_2, p_3, p_11, p_211, p_1741] (by decide)

def p_48559993 : CertifiedPrime :=
  certify 48559993 (5) [p_2, p_2, p_2, p_3, p_13, p_23, p_67, p_101] (by decide)

def p_48580901 : CertifiedPrime :=
  certify 48580901 (2) [p_2, p_2, p_5, p_5, p_17, p_17, p_41, p_41] (by decide)

def p_48720401 : CertifiedPrime :=
  certify 48720401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_349, p_349] (by decide)

def p_48776257 : CertifiedPrime :=
  certify 48776257 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_97, p_97] (by decide)

def p_48811183 : CertifiedPrime :=
  certify 48811183 (3) [p_2, p_3, p_7, p_17, p_137, p_499] (by decide)

def p_48825157 : CertifiedPrime :=
  certify 48825157 (2) [p_2, p_2, p_3, p_17, p_137, p_1747] (by decide)

def p_48860101 : CertifiedPrime :=
  certify 48860101 (6) [p_2, p_2, p_3, p_3, p_5, p_5, p_233, p_233] (by decide)

def p_48909043 : CertifiedPrime :=
  certify 48909043 (2) [p_2, p_3, p_3, p_3, p_7, p_13, p_37, p_269] (by decide)

def p_48944017 : CertifiedPrime :=
  certify 48944017 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_11, p_11, p_53, p_53] (by decide)

def p_49140101 : CertifiedPrime :=
  certify 49140101 (2) [p_2, p_2, p_5, p_5, p_701, p_701] (by decide)

def p_49189183 : CertifiedPrime :=
  certify 49189183 (3) [p_2, p_3, p_7, p_167, p_7013] (by decide)

def p_49196197 : CertifiedPrime :=
  certify 49196197 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_167, p_167] (by decide)

def p_49203211 : CertifiedPrime :=
  certify 49203211 (3) [p_2, p_3, p_5, p_7, p_23, p_61, p_167] (by decide)

def p_49224257 : CertifiedPrime :=
  certify 49224257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_877, p_877] (by decide)

def p_49371703 : CertifiedPrime :=
  certify 49371703 (3) [p_2, p_3, p_1171, p_7027] (by decide)

def p_49484191 : CertifiedPrime :=
  certify 49484191 (3) [p_2, p_3, p_5, p_7, p_67, p_3517] (by decide)

def p_49617937 : CertifiedPrime :=
  certify 49617937 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_587, p_587] (by decide)

def p_49667257 : CertifiedPrime :=
  certify 49667257 (5) [p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_29, p_881] (by decide)

def p_49702501 : CertifiedPrime :=
  certify 49702501 (10) [p_2, p_2, p_3, p_3, p_5, p_5, p_5, p_5, p_47, p_47] (by decide)

def p_49836541 : CertifiedPrime :=
  certify 49836541 (2) [p_2, p_2, p_3, p_5, p_13, p_181, p_353] (by decide)

def p_49928357 : CertifiedPrime :=
  certify 49928357 (2) [p_2, p_2, p_3533, p_3533] (by decide)

def p_50133481 : CertifiedPrime :=
  certify 50133481 (7) [p_2, p_2, p_2, p_3, p_5, p_59, p_73, p_97] (by decide)

def p_50175973 : CertifiedPrime :=
  certify 50175973 (6) [p_2, p_2, p_3, p_3, p_7, p_11, p_23, p_787] (by decide)

def p_50289373 : CertifiedPrime :=
  certify 50289373 (11) [p_2, p_2, p_3, p_3, p_7, p_197, p_1013] (by decide)

def p_50410001 : CertifiedPrime :=
  certify 50410001 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5, p_71, p_71] (by decide)

def p_50502343 : CertifiedPrime :=
  certify 50502343 (3) [p_2, p_3, p_11, p_17, p_19, p_23, p_103] (by decide)

def p_50559211 : CertifiedPrime :=
  certify 50559211 (22) [p_2, p_3, p_3, p_5, p_13, p_79, p_547] (by decide)

def p_50587657 : CertifiedPrime :=
  certify 50587657 (5) [p_2, p_2, p_2, p_3, p_7, p_127, p_2371] (by decide)

def p_50608997 : CertifiedPrime :=
  certify 50608997 (2) [p_2, p_2, p_3557, p_3557] (by decide)

def p_50630341 : CertifiedPrime :=
  certify 50630341 (6) [p_2, p_2, p_3, p_5, p_593, p_1423] (by decide)

def p_50673043 : CertifiedPrime :=
  certify 50673043 (2) [p_2, p_3, p_3, p_7, p_113, p_3559] (by decide)

def p_50730007 : CertifiedPrime :=
  certify 50730007 (3) [p_2, p_3, p_17, p_419, p_1187] (by decide)

def p_50836901 : CertifiedPrime :=
  certify 50836901 (13) [p_2, p_2, p_5, p_5, p_23, p_23, p_31, p_31] (by decide)

def p_51029593 : CertifiedPrime :=
  certify 51029593 (7) [p_2, p_2, p_2, p_3, p_19, p_47, p_2381] (by decide)

def p_51122501 : CertifiedPrime :=
  certify 51122501 (3) [p_2, p_2, p_5, p_5, p_5, p_5, p_11, p_11, p_13, p_13] (by decide)

def p_51158257 : CertifiedPrime :=
  certify 51158257 (5) [p_2, p_2, p_2, p_2, p_3, p_23, p_149, p_311] (by decide)

def p_51229807 : CertifiedPrime :=
  certify 51229807 (3) [p_2, p_3, p_17, p_421, p_1193] (by decide)

def p_51265601 : CertifiedPrime :=
  certify 51265601 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_179, p_179] (by decide)

def p_51272761 : CertifiedPrime :=
  certify 51272761 (13) [p_2, p_2, p_2, p_3, p_5, p_7, p_11, p_31, p_179] (by decide)

def p_51322897 : CertifiedPrime :=
  certify 51322897 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_199, p_199] (by decide)

def p_51330061 : CertifiedPrime :=
  certify 51330061 (7) [p_2, p_2, p_3, p_3, p_5, p_199, p_1433] (by decide)

def p_51573943 : CertifiedPrime :=
  certify 51573943 (11) [p_2, p_3, p_3, p_3, p_7, p_19, p_43, p_167] (by decide)

def p_51588307 : CertifiedPrime :=
  certify 51588307 (5) [p_2, p_3, p_3, p_3, p_7, p_11, p_19, p_653] (by decide)

def p_51696101 : CertifiedPrime :=
  certify 51696101 (2) [p_2, p_2, p_5, p_5, p_719, p_719] (by decide)

def p_51703291 : CertifiedPrime :=
  certify 51703291 (14) [p_2, p_3, p_3, p_5, p_17, p_47, p_719] (by decide)

def p_51789613 : CertifiedPrime :=
  certify 51789613 (2) [p_2, p_2, p_3, p_7, p_257, p_2399] (by decide)

def p_51832801 : CertifiedPrime :=
  certify 51832801 (14) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_23, p_313] (by decide)

def p_51890413 : CertifiedPrime :=
  certify 51890413 (2) [p_2, p_2, p_3, p_7, p_7, p_7, p_7, p_1801] (by decide)

def p_52070657 : CertifiedPrime :=
  certify 52070657 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_41, p_41] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
