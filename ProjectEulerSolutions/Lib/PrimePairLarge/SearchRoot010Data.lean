import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root10Candidates : List Nat := Data.row41
set_option maxRecDepth 8192 in
theorem root10Candidates_checked : candidates edge 5 0 41 tail11 winner.bound = root10Candidates := by
  apply candidates_eq_row edge 5 0 41 tail11 root10Candidates winner.bound tail11_ordered
  · decide
  · decide
  · funext q
    rfl
def root10Tail103 : List Nat := []
def root10Tail102 : List Nat := 25229 :: root10Tail103
def root10Tail101 : List Nat := 24443 :: root10Tail102
def root10Tail100 : List Nat := 24179 :: root10Tail101
def root10Tail99 : List Nat := 23957 :: root10Tail100
def root10Tail98 : List Nat := 23813 :: root10Tail99
def root10Tail97 : List Nat := 23753 :: root10Tail98
def root10Tail96 : List Nat := 23087 :: root10Tail97
def root10Tail95 : List Nat := 22973 :: root10Tail96
def root10Tail94 : List Nat := 22709 :: root10Tail95
def root10Tail93 : List Nat := 22343 :: root10Tail94
def root10Tail92 : List Nat := 22247 :: root10Tail93
def root10Tail91 : List Nat := 22067 :: root10Tail92
def root10Tail90 : List Nat := 21503 :: root10Tail91
def root10Tail89 : List Nat := 21017 :: root10Tail90
def root10Tail88 : List Nat := 20231 :: root10Tail89
def root10Tail87 : List Nat := 19889 :: root10Tail88
def root10Tail86 : List Nat := 19697 :: root10Tail87
def root10Tail85 : List Nat := 18749 :: root10Tail86
def root10Tail84 : List Nat := 18251 :: root10Tail85
def root10Tail83 : List Nat := 17783 :: root10Tail84
def root10Tail82 : List Nat := 17657 :: root10Tail83
def root10Tail81 : List Nat := 17489 :: root10Tail82
def root10Tail80 : List Nat := 17291 :: root10Tail81
def root10Tail79 : List Nat := 16649 :: root10Tail80
def root10Tail78 : List Nat := 16187 :: root10Tail79
def root10Tail77 : List Nat := 15791 :: root10Tail78
def root10Tail76 : List Nat := 15773 :: root10Tail77
def root10Tail75 : List Nat := 15443 :: root10Tail76
def root10Tail74 : List Nat := 15263 :: root10Tail75
def root10Tail73 : List Nat := 15101 :: root10Tail74
def root10Tail72 : List Nat := 15053 :: root10Tail73
def root10Tail71 : List Nat := 14741 :: root10Tail72
def root10Tail70 : List Nat := 14087 :: root10Tail71
def root10Tail69 : List Nat := 14009 :: root10Tail70
def root10Tail68 : List Nat := 13931 :: root10Tail69
def root10Tail67 : List Nat := 13259 :: root10Tail68
def root10Tail66 : List Nat := 12497 :: root10Tail67
def root10Tail65 : List Nat := 12413 :: root10Tail66
def root10Tail64 : List Nat := 12263 :: root10Tail65
def root10Tail63 : List Nat := 11867 :: root10Tail64
def root10Tail62 : List Nat := 11801 :: root10Tail63
def root10Tail61 : List Nat := 11489 :: root10Tail62
def root10Tail60 : List Nat := 10163 :: root10Tail61
def root10Tail59 : List Nat := 9929 :: root10Tail60
def root10Tail58 : List Nat := 9623 :: root10Tail59
def root10Tail57 : List Nat := 9473 :: root10Tail58
def root10Tail56 : List Nat := 9467 :: root10Tail57
def root10Tail55 : List Nat := 9281 :: root10Tail56
def root10Tail54 : List Nat := 9161 :: root10Tail55
def root10Tail53 : List Nat := 8933 :: root10Tail54
def root10Tail52 : List Nat := 8867 :: root10Tail53
def root10Tail51 : List Nat := 8423 :: root10Tail52
def root10Tail50 : List Nat := 8219 :: root10Tail51
def root10Tail49 : List Nat := 8069 :: root10Tail50
def root10Tail48 : List Nat := 7793 :: root10Tail49
def root10Tail47 : List Nat := 7643 :: root10Tail48
def root10Tail46 : List Nat := 7583 :: root10Tail47
def root10Tail45 : List Nat := 7457 :: root10Tail46
def root10Tail44 : List Nat := 7451 :: root10Tail45
def root10Tail43 : List Nat := 7331 :: root10Tail44
def root10Tail42 : List Nat := 7187 :: root10Tail43
def root10Tail41 : List Nat := 6833 :: root10Tail42
def root10Tail40 : List Nat := 6719 :: root10Tail41
def root10Tail39 : List Nat := 6491 :: root10Tail40
def root10Tail38 : List Nat := 6473 :: root10Tail39
def root10Tail37 : List Nat := 6263 :: root10Tail38
def root10Tail36 : List Nat := 6089 :: root10Tail37
def root10Tail35 : List Nat := 5717 :: root10Tail36
def root10Tail34 : List Nat := 5477 :: root10Tail35
def root10Tail33 : List Nat := 5147 :: root10Tail34
def root10Tail32 : List Nat := 4691 :: root10Tail33
def root10Tail31 : List Nat := 4679 :: root10Tail32
def root10Tail30 : List Nat := 4457 :: root10Tail31
def root10Tail29 : List Nat := 4451 :: root10Tail30
def root10Tail28 : List Nat := 4217 :: root10Tail29
def root10Tail27 : List Nat := 4049 :: root10Tail28
def root10Tail26 : List Nat := 4013 :: root10Tail27
def root10Tail25 : List Nat := 3719 :: root10Tail26
def root10Tail24 : List Nat := 3533 :: root10Tail25
def root10Tail23 : List Nat := 3527 :: root10Tail24
def root10Tail22 : List Nat := 3461 :: root10Tail23
def root10Tail21 : List Nat := 3299 :: root10Tail22
def root10Tail20 : List Nat := 2939 :: root10Tail21
def root10Tail19 : List Nat := 2609 :: root10Tail20
def root10Tail18 : List Nat := 2411 :: root10Tail19
def root10Tail17 : List Nat := 2339 :: root10Tail18
def root10Tail16 : List Nat := 2333 :: root10Tail17
def root10Tail15 : List Nat := 2081 :: root10Tail16
def root10Tail14 : List Nat := 1823 :: root10Tail15
def root10Tail13 : List Nat := 1667 :: root10Tail14
def root10Tail12 : List Nat := 1637 :: root10Tail13
def root10Tail11 : List Nat := 1613 :: root10Tail12
def root10Tail10 : List Nat := 1583 :: root10Tail11
def root10Tail9 : List Nat := 1259 :: root10Tail10
def root10Tail8 : List Nat := 1031 :: root10Tail9
def root10Tail7 : List Nat := 1013 :: root10Tail8
def root10Tail6 : List Nat := 911 :: root10Tail7
def root10Tail5 : List Nat := 887 :: root10Tail6
def root10Tail4 : List Nat := 863 :: root10Tail5
def root10Tail3 : List Nat := 719 :: root10Tail4
def root10Tail2 : List Nat := 593 :: root10Tail3
def root10Tail1 : List Nat := 257 :: root10Tail2
def root10Tail0 : List Nat := 227 :: root10Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
