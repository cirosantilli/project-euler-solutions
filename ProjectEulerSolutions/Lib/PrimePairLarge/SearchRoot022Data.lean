import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root22Candidates : List Nat := Data.row97
set_option maxRecDepth 8192 in
theorem root22Candidates_checked : candidates edge 5 0 97 tail23 winner.bound = root22Candidates := by
  apply candidates_eq_row edge 5 0 97 tail23 root22Candidates winner.bound tail23_ordered
  · decide
  · decide
  · funext q
    rfl
def root22Tail103 : List Nat := []
def root22Tail102 : List Nat := 25801 :: root22Tail103
def root22Tail101 : List Nat := 25609 :: root22Tail102
def root22Tail100 : List Nat := 25171 :: root22Tail101
def root22Tail99 : List Nat := 25153 :: root22Tail100
def root22Tail98 : List Nat := 25087 :: root22Tail99
def root22Tail97 : List Nat := 24859 :: root22Tail98
def root22Tail96 : List Nat := 24151 :: root22Tail97
def root22Tail95 : List Nat := 23887 :: root22Tail96
def root22Tail94 : List Nat := 23071 :: root22Tail95
def root22Tail93 : List Nat := 23017 :: root22Tail94
def root22Tail92 : List Nat := 23011 :: root22Tail93
def root22Tail91 : List Nat := 21871 :: root22Tail92
def root22Tail90 : List Nat := 21577 :: root22Tail91
def root22Tail89 : List Nat := 21247 :: root22Tail90
def root22Tail88 : List Nat := 21169 :: root22Tail89
def root22Tail87 : List Nat := 21163 :: root22Tail88
def root22Tail86 : List Nat := 20959 :: root22Tail87
def root22Tail85 : List Nat := 20551 :: root22Tail86
def root22Tail84 : List Nat := 20149 :: root22Tail85
def root22Tail83 : List Nat := 19867 :: root22Tail84
def root22Tail82 : List Nat := 19603 :: root22Tail83
def root22Tail81 : List Nat := 19441 :: root22Tail82
def root22Tail80 : List Nat := 19231 :: root22Tail81
def root22Tail79 : List Nat := 19141 :: root22Tail80
def root22Tail78 : List Nat := 19081 :: root22Tail79
def root22Tail77 : List Nat := 17989 :: root22Tail78
def root22Tail76 : List Nat := 17551 :: root22Tail77
def root22Tail75 : List Nat := 17443 :: root22Tail76
def root22Tail74 : List Nat := 17401 :: root22Tail75
def root22Tail73 : List Nat := 16747 :: root22Tail74
def root22Tail72 : List Nat := 16633 :: root22Tail73
def root22Tail71 : List Nat := 16477 :: root22Tail72
def root22Tail70 : List Nat := 15331 :: root22Tail71
def root22Tail69 : List Nat := 14437 :: root22Tail70
def root22Tail68 : List Nat := 13789 :: root22Tail69
def root22Tail67 : List Nat := 13309 :: root22Tail68
def root22Tail66 : List Nat := 13219 :: root22Tail67
def root22Tail65 : List Nat := 12799 :: root22Tail66
def root22Tail64 : List Nat := 12697 :: root22Tail65
def root22Tail63 : List Nat := 11971 :: root22Tail64
def root22Tail62 : List Nat := 11833 :: root22Tail63
def root22Tail61 : List Nat := 11503 :: root22Tail62
def root22Tail60 : List Nat := 11467 :: root22Tail61
def root22Tail59 : List Nat := 11257 :: root22Tail60
def root22Tail58 : List Nat := 11059 :: root22Tail59
def root22Tail57 : List Nat := 10711 :: root22Tail58
def root22Tail56 : List Nat := 10627 :: root22Tail57
def root22Tail55 : List Nat := 10513 :: root22Tail56
def root22Tail54 : List Nat := 9949 :: root22Tail55
def root22Tail53 : List Nat := 9907 :: root22Tail54
def root22Tail52 : List Nat := 9787 :: root22Tail53
def root22Tail51 : List Nat := 9337 :: root22Tail52
def root22Tail50 : List Nat := 8689 :: root22Tail51
def root22Tail49 : List Nat := 8521 :: root22Tail50
def root22Tail48 : List Nat := 8287 :: root22Tail49
def root22Tail47 : List Nat := 8269 :: root22Tail48
def root22Tail46 : List Nat := 8209 :: root22Tail47
def root22Tail45 : List Nat := 8053 :: root22Tail46
def root22Tail44 : List Nat := 8011 :: root22Tail45
def root22Tail43 : List Nat := 7927 :: root22Tail44
def root22Tail42 : List Nat := 7681 :: root22Tail43
def root22Tail41 : List Nat := 7507 :: root22Tail42
def root22Tail40 : List Nat := 6991 :: root22Tail41
def root22Tail39 : List Nat := 6883 :: root22Tail40
def root22Tail38 : List Nat := 6637 :: root22Tail39
def root22Tail37 : List Nat := 6571 :: root22Tail38
def root22Tail36 : List Nat := 6301 :: root22Tail37
def root22Tail35 : List Nat := 6271 :: root22Tail36
def root22Tail34 : List Nat := 5581 :: root22Tail35
def root22Tail33 : List Nat := 5281 :: root22Tail34
def root22Tail32 : List Nat := 5011 :: root22Tail33
def root22Tail31 : List Nat := 4969 :: root22Tail32
def root22Tail30 : List Nat := 4957 :: root22Tail31
def root22Tail29 : List Nat := 4657 :: root22Tail30
def root22Tail28 : List Nat := 4513 :: root22Tail29
def root22Tail27 : List Nat := 4507 :: root22Tail28
def root22Tail26 : List Nat := 4261 :: root22Tail27
def root22Tail25 : List Nat := 3853 :: root22Tail26
def root22Tail24 : List Nat := 3727 :: root22Tail25
def root22Tail23 : List Nat := 3691 :: root22Tail24
def root22Tail22 : List Nat := 3373 :: root22Tail23
def root22Tail21 : List Nat := 3331 :: root22Tail22
def root22Tail20 : List Nat := 2833 :: root22Tail21
def root22Tail19 : List Nat := 2221 :: root22Tail20
def root22Tail18 : List Nat := 2017 :: root22Tail19
def root22Tail17 : List Nat := 1951 :: root22Tail18
def root22Tail16 : List Nat := 1783 :: root22Tail17
def root22Tail15 : List Nat := 1381 :: root22Tail16
def root22Tail14 : List Nat := 1291 :: root22Tail15
def root22Tail13 : List Nat := 1279 :: root22Tail14
def root22Tail12 : List Nat := 1093 :: root22Tail13
def root22Tail11 : List Nat := 1063 :: root22Tail12
def root22Tail10 : List Nat := 1039 :: root22Tail11
def root22Tail9 : List Nat := 1021 :: root22Tail10
def root22Tail8 : List Nat := 967 :: root22Tail9
def root22Tail7 : List Nat := 919 :: root22Tail8
def root22Tail6 : List Nat := 883 :: root22Tail7
def root22Tail5 : List Nat := 829 :: root22Tail6
def root22Tail4 : List Nat := 787 :: root22Tail5
def root22Tail3 : List Nat := 379 :: root22Tail4
def root22Tail2 : List Nat := 373 :: root22Tail3
def root22Tail1 : List Nat := 241 :: root22Tail2
def root22Tail0 : List Nat := 157 :: root22Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
