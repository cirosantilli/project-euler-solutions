import ProjectEulerSolutions.Lib.PrattCertificates
import ProjectEulerSolutions.Lib.SpiralPrimeCertificates.Chunk020

/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/
namespace ProjectEulerSolutions.P58PrimeNodes
open ProjectEulerSolutions.PrattCertificates

def p_35569297 : CertifiedPrime :=
  certify 35569297 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_71, p_71] (by decide)

def p_35634931 : CertifiedPrime :=
  certify 35634931 (3) [p_2, p_3, p_5, p_47, p_127, p_199] (by decide)

def p_35640901 : CertifiedPrime :=
  certify 35640901 (2) [p_2, p_2, p_3, p_3, p_5, p_5, p_199, p_199] (by decide)

def p_35754421 : CertifiedPrime :=
  certify 35754421 (6) [p_2, p_2, p_3, p_5, p_13, p_23, p_1993] (by decide)

def p_35808257 : CertifiedPrime :=
  certify 35808257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_11, p_11, p_17, p_17] (by decide)

def p_35814241 : CertifiedPrime :=
  certify 35814241 (26) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_7, p_11, p_17, p_19] (by decide)

def p_35880101 : CertifiedPrime :=
  certify 35880101 (2) [p_2, p_2, p_5, p_5, p_599, p_599] (by decide)

def p_35922043 : CertifiedPrime :=
  certify 35922043 (5) [p_2, p_3, p_3, p_3, p_3, p_13, p_37, p_461] (by decide)

def p_35952017 : CertifiedPrime :=
  certify 35952017 (3) [p_2, p_2, p_2, p_2, p_1499, p_1499] (by decide)

def p_36066031 : CertifiedPrime :=
  certify 36066031 (6) [p_2, p_3, p_5, p_7, p_11, p_13, p_1201] (by decide)

def p_36072037 : CertifiedPrime :=
  certify 36072037 (2) [p_2, p_2, p_3, p_3, p_7, p_7, p_11, p_11, p_13, p_13] (by decide)

def p_36078043 : CertifiedPrime :=
  certify 36078043 (3) [p_2, p_3, p_7, p_11, p_13, p_6007] (by decide)

def p_36120101 : CertifiedPrime :=
  certify 36120101 (2) [p_2, p_2, p_5, p_5, p_601, p_601] (by decide)

def p_36174211 : CertifiedPrime :=
  certify 36174211 (2) [p_2, p_3, p_5, p_31, p_97, p_401] (by decide)

def p_36192257 : CertifiedPrime :=
  certify 36192257 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_47, p_47] (by decide)

def p_36222343 : CertifiedPrime :=
  certify 36222343 (3) [p_2, p_3, p_13, p_17, p_59, p_463] (by decide)

def p_36282553 : CertifiedPrime :=
  certify 36282553 (5) [p_2, p_2, p_2, p_3, p_19, p_251, p_317] (by decide)

def p_36360901 : CertifiedPrime :=
  certify 36360901 (7) [p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_67, p_67] (by decide)

def p_36403123 : CertifiedPrime :=
  certify 36403123 (2) [p_2, p_3, p_7, p_431, p_2011] (by decide)

def p_36511807 : CertifiedPrime :=
  certify 36511807 (3) [p_2, p_3, p_19, p_53, p_6043] (by decide)

def p_36554117 : CertifiedPrime :=
  certify 36554117 (2) [p_2, p_2, p_3023, p_3023] (by decide)

def p_36572257 : CertifiedPrime :=
  certify 36572257 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_7, p_6047] (by decide)

def p_36620653 : CertifiedPrime :=
  certify 36620653 (2) [p_2, p_2, p_3, p_17, p_89, p_2017] (by decide)

def p_36723601 : CertifiedPrime :=
  certify 36723601 (11) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_101, p_101] (by decide)

def p_36766033 : CertifiedPrime :=
  certify 36766033 (7) [p_2, p_2, p_2, p_2, p_3, p_43, p_47, p_379] (by decide)

def p_36790291 : CertifiedPrime :=
  certify 36790291 (10) [p_2, p_3, p_3, p_5, p_337, p_1213] (by decide)

def p_36936007 : CertifiedPrime :=
  certify 36936007 (3) [p_2, p_3, p_59, p_103, p_1013] (by decide)

def p_37094191 : CertifiedPrime :=
  certify 37094191 (19) [p_2, p_3, p_5, p_7, p_29, p_6091] (by decide)

def p_37155121 : CertifiedPrime :=
  certify 37155121 (17) [p_2, p_2, p_2, p_2, p_3, p_5, p_23, p_53, p_127] (by decide)

def p_37161217 : CertifiedPrime :=
  certify 37161217 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_127, p_127] (by decide)

def p_37167313 : CertifiedPrime :=
  certify 37167313 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_67, p_127] (by decide)

def p_37240507 : CertifiedPrime :=
  certify 37240507 (5) [p_2, p_3, p_3, p_3, p_17, p_113, p_359] (by decide)

def p_37301557 : CertifiedPrime :=
  certify 37301557 (5) [p_2, p_2, p_3, p_31, p_197, p_509] (by decide)

def p_37332101 : CertifiedPrime :=
  certify 37332101 (2) [p_2, p_2, p_5, p_5, p_13, p_13, p_47, p_47] (by decide)

def p_37350433 : CertifiedPrime :=
  certify 37350433 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_97, p_191] (by decide)

def p_37387111 : CertifiedPrime :=
  certify 37387111 (3) [p_2, p_3, p_5, p_1019, p_1223] (by decide)

def p_37423807 : CertifiedPrime :=
  certify 37423807 (3) [p_2, p_3, p_7, p_19, p_23, p_2039] (by decide)

def p_37454401 : CertifiedPrime :=
  certify 37454401 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_5, p_5, p_17, p_17] (by decide)

def p_37527877 : CertifiedPrime :=
  certify 37527877 (2) [p_2, p_2, p_3, p_3, p_1021, p_1021] (by decide)

def p_37558513 : CertifiedPrime :=
  certify 37558513 (5) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_227, p_383] (by decide)

def p_37576901 : CertifiedPrime :=
  certify 37576901 (2) [p_2, p_2, p_5, p_5, p_613, p_613] (by decide)

def p_37625957 : CertifiedPrime :=
  certify 37625957 (2) [p_2, p_2, p_3067, p_3067] (by decide)

def p_37668907 : CertifiedPrime :=
  certify 37668907 (5) [p_2, p_3, p_3, p_11, p_17, p_19, p_19, p_31] (by decide)

def p_37699601 : CertifiedPrime :=
  certify 37699601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_307, p_307] (by decide)

def p_37896337 : CertifiedPrime :=
  certify 37896337 (7) [p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_19, p_19] (by decide)

def p_37902493 : CertifiedPrime :=
  certify 37902493 (5) [p_2, p_2, p_3, p_3, p_3, p_3, p_19, p_47, p_131] (by decide)

def p_37994897 : CertifiedPrime :=
  certify 37994897 (3) [p_2, p_2, p_2, p_2, p_23, p_23, p_67, p_67] (by decide)

def p_38019557 : CertifiedPrime :=
  certify 38019557 (2) [p_2, p_2, p_3083, p_3083] (by decide)

def p_38075071 : CertifiedPrime :=
  certify 38075071 (13) [p_2, p_3, p_5, p_11, p_11, p_17, p_617] (by decide)

def p_38112103 : CertifiedPrime :=
  certify 38112103 (5) [p_2, p_3, p_3, p_7, p_7, p_7, p_6173] (by decide)

def p_38142977 : CertifiedPrime :=
  certify 38142977 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_193, p_193] (by decide)

def p_38161507 : CertifiedPrime :=
  certify 38161507 (2) [p_2, p_3, p_29, p_71, p_3089] (by decide)

def p_38316101 : CertifiedPrime :=
  certify 38316101 (3) [p_2, p_2, p_5, p_5, p_619, p_619] (by decide)

def p_38446201 : CertifiedPrime :=
  certify 38446201 (7) [p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_31, p_53] (by decide)

def p_38533057 : CertifiedPrime :=
  certify 38533057 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_97, p_2069] (by decide)

def p_38638657 : CertifiedPrime :=
  certify 38638657 (5) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_37, p_37] (by decide)

def p_38644873 : CertifiedPrime :=
  certify 38644873 (5) [p_2, p_2, p_2, p_3, p_7, p_37, p_6217] (by decide)

def p_38682181 : CertifiedPrime :=
  certify 38682181 (2) [p_2, p_2, p_3, p_3, p_5, p_311, p_691] (by decide)

def p_38688401 : CertifiedPrime :=
  certify 38688401 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_311, p_311] (by decide)

def p_38719507 : CertifiedPrime :=
  certify 38719507 (3) [p_2, p_3, p_7, p_7, p_17, p_61, p_127] (by decide)

def p_38831593 : CertifiedPrime :=
  certify 38831593 (5) [p_2, p_2, p_2, p_3, p_19, p_31, p_41, p_67] (by decide)

def p_38862757 : CertifiedPrime :=
  certify 38862757 (2) [p_2, p_2, p_3, p_3, p_1039, p_1039] (by decide)

def p_38887697 : CertifiedPrime :=
  certify 38887697 (3) [p_2, p_2, p_2, p_2, p_1559, p_1559] (by decide)

def p_38906407 : CertifiedPrime :=
  certify 38906407 (5) [p_2, p_3, p_3, p_3, p_3, p_7, p_11, p_3119] (by decide)

def p_38937601 : CertifiedPrime :=
  certify 38937601 (14) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_13, p_13] (by decide)

def p_39081253 : CertifiedPrime :=
  certify 39081253 (14) [p_2, p_2, p_3, p_7, p_19, p_47, p_521] (by decide)

def p_39112517 : CertifiedPrime :=
  certify 39112517 (2) [p_2, p_2, p_53, p_53, p_59, p_59] (by decide)

def p_39262757 : CertifiedPrime :=
  certify 39262757 (2) [p_2, p_2, p_13, p_13, p_241, p_241] (by decide)

def p_39344257 : CertifiedPrime :=
  certify 39344257 (10) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_7, p_7, p_17, p_41] (by decide)

def p_39507511 : CertifiedPrime :=
  certify 39507511 (3) [p_2, p_3, p_5, p_7, p_419, p_449] (by decide)

def p_39734113 : CertifiedPrime :=
  certify 39734113 (7) [p_2, p_2, p_2, p_2, p_2, p_3, p_11, p_191, p_197] (by decide)

def p_39759331 : CertifiedPrime :=
  certify 39759331 (3) [p_2, p_3, p_5, p_13, p_97, p_1051] (by decide)

def p_39765637 : CertifiedPrime :=
  certify 39765637 (2) [p_2, p_2, p_3, p_3, p_1051, p_1051] (by decide)

def p_39771943 : CertifiedPrime :=
  certify 39771943 (3) [p_2, p_3, p_7, p_17, p_53, p_1051] (by decide)

def p_39866597 : CertifiedPrime :=
  certify 39866597 (2) [p_2, p_2, p_7, p_7, p_11, p_11, p_41, p_41] (by decide)

def p_39999301 : CertifiedPrime :=
  certify 39999301 (7) [p_2, p_2, p_3, p_5, p_5, p_11, p_17, p_23, p_31] (by decide)

def p_40024603 : CertifiedPrime :=
  certify 40024603 (2) [p_2, p_3, p_3, p_19, p_37, p_3163] (by decide)

def p_40195601 : CertifiedPrime :=
  certify 40195601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_317, p_317] (by decide)

def p_40214623 : CertifiedPrime :=
  certify 40214623 (3) [p_2, p_3, p_7, p_17, p_151, p_373] (by decide)

def p_40303453 : CertifiedPrime :=
  certify 40303453 (6) [p_2, p_2, p_3, p_7, p_23, p_23, p_907] (by decide)

def p_40322501 : CertifiedPrime :=
  certify 40322501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_127, p_127] (by decide)

def p_40328851 : CertifiedPrime :=
  certify 40328851 (2) [p_2, p_3, p_5, p_5, p_29, p_73, p_127] (by decide)

def p_40379671 : CertifiedPrime :=
  certify 40379671 (19) [p_2, p_3, p_3, p_5, p_31, p_41, p_353] (by decide)

def p_40449601 : CertifiedPrime :=
  certify 40449601 (7) [p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_53, p_53] (by decide)

def p_40494133 : CertifiedPrime :=
  certify 40494133 (2) [p_2, p_2, p_3, p_3, p_7, p_37, p_43, p_101] (by decide)

def p_40525957 : CertifiedPrime :=
  certify 40525957 (2) [p_2, p_2, p_3, p_3, p_1061, p_1061] (by decide)

def p_40557793 : CertifiedPrime :=
  certify 40557793 (5) [p_2, p_2, p_2, p_2, p_2, p_3, p_11, p_193, p_199] (by decide)

def p_40596013 : CertifiedPrime :=
  certify 40596013 (11) [p_2, p_2, p_3, p_3, p_3, p_23, p_59, p_277] (by decide)

def p_40672507 : CertifiedPrime :=
  certify 40672507 (5) [p_2, p_3, p_7, p_911, p_1063] (by decide)

def p_40685263 : CertifiedPrime :=
  certify 40685263 (3) [p_2, p_3, p_1063, p_6379] (by decide)

def p_40825711 : CertifiedPrime :=
  certify 40825711 (11) [p_2, p_3, p_3, p_5, p_71, p_6389] (by decide)

def p_40915213 : CertifiedPrime :=
  certify 40915213 (5) [p_2, p_2, p_3, p_13, p_41, p_6397] (by decide)

def p_40960001 : CertifiedPrime :=
  certify 40960001 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_5, p_5, p_5, p_5] (by decide)

def p_41068873 : CertifiedPrime :=
  certify 41068873 (7) [p_2, p_2, p_2, p_3, p_3, p_13, p_17, p_29, p_89] (by decide)

def p_41184307 : CertifiedPrime :=
  certify 41184307 (2) [p_2, p_3, p_3, p_23, p_31, p_3209] (by decide)

def p_41216401 : CertifiedPrime :=
  certify 41216401 (17) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_107, p_107] (by decide)

def p_41222821 : CertifiedPrime :=
  certify 41222821 (6) [p_2, p_2, p_3, p_5, p_107, p_6421] (by decide)

def p_41299903 : CertifiedPrime :=
  certify 41299903 (5) [p_2, p_3, p_3, p_3, p_7, p_17, p_6427] (by decide)

def p_41396357 : CertifiedPrime :=
  certify 41396357 (2) [p_2, p_2, p_3217, p_3217] (by decide)

def p_41402791 : CertifiedPrime :=
  certify 41402791 (6) [p_2, p_3, p_3, p_5, p_11, p_13, p_3217] (by decide)

def p_41480041 : CertifiedPrime :=
  certify 41480041 (13) [p_2, p_2, p_2, p_3, p_5, p_7, p_19, p_23, p_113] (by decide)

def p_41492923 : CertifiedPrime :=
  certify 41492923 (3) [p_2, p_3, p_19, p_113, p_3221] (by decide)

def p_41557363 : CertifiedPrime :=
  certify 41557363 (2) [p_2, p_3, p_7, p_11, p_293, p_307] (by decide)

def p_41570257 : CertifiedPrime :=
  certify 41570257 (5) [p_2, p_2, p_2, p_2, p_3, p_7, p_13, p_31, p_307] (by decide)

def p_41608951 : CertifiedPrime :=
  certify 41608951 (3) [p_2, p_3, p_5, p_5, p_43, p_6451] (by decide)

def p_41673481 : CertifiedPrime :=
  certify 41673481 (21) [p_2, p_2, p_2, p_3, p_5, p_269, p_1291] (by decide)

def p_41725141 : CertifiedPrime :=
  certify 41725141 (2) [p_2, p_2, p_3, p_5, p_17, p_19, p_2153] (by decide)

def p_41731601 : CertifiedPrime :=
  certify 41731601 (3) [p_2, p_2, p_2, p_2, p_5, p_5, p_17, p_17, p_19, p_19] (by decide)

def p_41763907 : CertifiedPrime :=
  certify 41763907 (5) [p_2, p_3, p_3, p_23, p_281, p_359] (by decide)

def p_41990401 : CertifiedPrime :=
  certify 41990401 (11) [p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_2, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_3, p_5, p_5] (by decide)

def p_42022807 : CertifiedPrime :=
  certify 42022807 (5) [p_2, p_3, p_7, p_463, p_2161] (by decide)

def p_42113611 : CertifiedPrime :=
  certify 42113611 (11) [p_2, p_3, p_3, p_5, p_7, p_11, p_59, p_103] (by decide)

def p_42191521 : CertifiedPrime :=
  certify 42191521 (13) [p_2, p_2, p_2, p_2, p_2, p_3, p_5, p_7, p_29, p_433] (by decide)

def p_42217507 : CertifiedPrime :=
  certify 42217507 (5) [p_2, p_3, p_3, p_19, p_19, p_73, p_89] (by decide)

def p_42386611 : CertifiedPrime :=
  certify 42386611 (3) [p_2, p_3, p_5, p_7, p_17, p_31, p_383] (by decide)

def p_42432197 : CertifiedPrime :=
  certify 42432197 (2) [p_2, p_2, p_3257, p_3257] (by decide)

def p_42640901 : CertifiedPrime :=
  certify 42640901 (3) [p_2, p_2, p_5, p_5, p_653, p_653] (by decide)

def p_42719297 : CertifiedPrime :=
  certify 42719297 (3) [p_2, p_2, p_2, p_2, p_2, p_2, p_19, p_19, p_43, p_43] (by decide)

def p_42738907 : CertifiedPrime :=
  certify 42738907 (2) [p_2, p_3, p_7, p_467, p_2179] (by decide)

def p_42765061 : CertifiedPrime :=
  certify 42765061 (6) [p_2, p_2, p_3, p_5, p_13, p_109, p_503] (by decide)

def p_42771601 : CertifiedPrime :=
  certify 42771601 (13) [p_2, p_2, p_2, p_2, p_3, p_3, p_5, p_5, p_109, p_109] (by decide)

def p_42850117 : CertifiedPrime :=
  certify 42850117 (2) [p_2, p_2, p_3, p_3, p_1091, p_1091] (by decide)

def p_42902501 : CertifiedPrime :=
  certify 42902501 (2) [p_2, p_2, p_5, p_5, p_5, p_5, p_131, p_131] (by decide)

def p_42922153 : CertifiedPrime :=
  certify 42922153 (11) [p_2, p_2, p_2, p_3, p_3, p_7, p_13, p_6551] (by decide)

def p_42935257 : CertifiedPrime :=
  certify 42935257 (10) [p_2, p_2, p_2, p_3, p_3, p_7, p_13, p_6553] (by decide)

def p_43132057 : CertifiedPrime :=
  certify 43132057 (5) [p_2, p_2, p_2, p_3, p_11, p_199, p_821] (by decide)

def p_43158331 : CertifiedPrime :=
  certify 43158331 (13) [p_2, p_3, p_3, p_5, p_73, p_6569] (by decide)

def p_43210903 : CertifiedPrime :=
  certify 43210903 (10) [p_2, p_3, p_7, p_19, p_173, p_313] (by decide)

end ProjectEulerSolutions.P58PrimeNodes
