import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root18Candidates : List Nat := Data.row73
set_option maxRecDepth 8192 in
theorem root18Candidates_checked : candidates edge 5 0 73 tail19 winner.bound = root18Candidates := by
  apply candidates_eq_row edge 5 0 73 tail19 root18Candidates winner.bound tail19_ordered
  · decide
  · decide
  · funext q
    rfl
def root18Tail103 : List Nat := []
def root18Tail102 : List Nat := 25471 :: root18Tail103
def root18Tail101 : List Nat := 25411 :: root18Tail102
def root18Tail100 : List Nat := 25243 :: root18Tail101
def root18Tail99 : List Nat := 24469 :: root18Tail100
def root18Tail98 : List Nat := 24379 :: root18Tail99
def root18Tail97 : List Nat := 24337 :: root18Tail98
def root18Tail96 : List Nat := 24133 :: root18Tail97
def root18Tail95 : List Nat := 24007 :: root18Tail96
def root18Tail94 : List Nat := 23497 :: root18Tail95
def root18Tail93 : List Nat := 23143 :: root18Tail94
def root18Tail92 : List Nat := 22669 :: root18Tail93
def root18Tail91 : List Nat := 21757 :: root18Tail92
def root18Tail90 : List Nat := 21529 :: root18Tail91
def root18Tail89 : List Nat := 20359 :: root18Tail90
def root18Tail88 : List Nat := 20023 :: root18Tail89
def root18Tail87 : List Nat := 19597 :: root18Tail88
def root18Tail86 : List Nat := 19387 :: root18Tail87
def root18Tail85 : List Nat := 19381 :: root18Tail86
def root18Tail84 : List Nat := 18757 :: root18Tail85
def root18Tail83 : List Nat := 18523 :: root18Tail84
def root18Tail82 : List Nat := 18379 :: root18Tail83
def root18Tail81 : List Nat := 17977 :: root18Tail82
def root18Tail80 : List Nat := 17911 :: root18Tail81
def root18Tail79 : List Nat := 16567 :: root18Tail80
def root18Tail78 : List Nat := 15739 :: root18Tail79
def root18Tail77 : List Nat := 15727 :: root18Tail78
def root18Tail76 : List Nat := 15313 :: root18Tail77
def root18Tail75 : List Nat := 15271 :: root18Tail76
def root18Tail74 : List Nat := 14143 :: root18Tail75
def root18Tail73 : List Nat := 13693 :: root18Tail74
def root18Tail72 : List Nat := 13513 :: root18Tail73
def root18Tail71 : List Nat := 13063 :: root18Tail72
def root18Tail70 : List Nat := 12979 :: root18Tail71
def root18Tail69 : List Nat := 12583 :: root18Tail70
def root18Tail68 : List Nat := 12037 :: root18Tail69
def root18Tail67 : List Nat := 11701 :: root18Tail68
def root18Tail66 : List Nat := 11497 :: root18Tail67
def root18Tail65 : List Nat := 11467 :: root18Tail66
def root18Tail64 : List Nat := 11287 :: root18Tail65
def root18Tail63 : List Nat := 11149 :: root18Tail64
def root18Tail62 : List Nat := 10651 :: root18Tail63
def root18Tail61 : List Nat := 9859 :: root18Tail62
def root18Tail60 : List Nat := 9829 :: root18Tail61
def root18Tail59 : List Nat := 9787 :: root18Tail60
def root18Tail58 : List Nat := 9649 :: root18Tail59
def root18Tail57 : List Nat := 9631 :: root18Tail58
def root18Tail56 : List Nat := 9601 :: root18Tail57
def root18Tail55 : List Nat := 9433 :: root18Tail56
def root18Tail54 : List Nat := 9397 :: root18Tail55
def root18Tail53 : List Nat := 9241 :: root18Tail54
def root18Tail52 : List Nat := 8839 :: root18Tail53
def root18Tail51 : List Nat := 8707 :: root18Tail52
def root18Tail50 : List Nat := 8677 :: root18Tail51
def root18Tail49 : List Nat := 7687 :: root18Tail50
def root18Tail48 : List Nat := 7591 :: root18Tail49
def root18Tail47 : List Nat := 7537 :: root18Tail48
def root18Tail46 : List Nat := 7351 :: root18Tail47
def root18Tail45 : List Nat := 7309 :: root18Tail46
def root18Tail44 : List Nat := 7207 :: root18Tail45
def root18Tail43 : List Nat := 7159 :: root18Tail44
def root18Tail42 : List Nat := 7129 :: root18Tail43
def root18Tail41 : List Nat := 6823 :: root18Tail42
def root18Tail40 : List Nat := 6793 :: root18Tail41
def root18Tail39 : List Nat := 6691 :: root18Tail40
def root18Tail38 : List Nat := 6367 :: root18Tail39
def root18Tail37 : List Nat := 6277 :: root18Tail38
def root18Tail36 : List Nat := 6121 :: root18Tail37
def root18Tail35 : List Nat := 6091 :: root18Tail36
def root18Tail34 : List Nat := 5953 :: root18Tail35
def root18Tail33 : List Nat := 5821 :: root18Tail34
def root18Tail32 : List Nat := 5659 :: root18Tail33
def root18Tail31 : List Nat := 5443 :: root18Tail32
def root18Tail30 : List Nat := 5107 :: root18Tail31
def root18Tail29 : List Nat := 4957 :: root18Tail30
def root18Tail28 : List Nat := 4813 :: root18Tail29
def root18Tail27 : List Nat := 4759 :: root18Tail28
def root18Tail26 : List Nat := 4663 :: root18Tail27
def root18Tail25 : List Nat := 4549 :: root18Tail26
def root18Tail24 : List Nat := 4177 :: root18Tail25
def root18Tail23 : List Nat := 3847 :: root18Tail24
def root18Tail22 : List Nat := 3823 :: root18Tail23
def root18Tail21 : List Nat := 3517 :: root18Tail22
def root18Tail20 : List Nat := 3433 :: root18Tail21
def root18Tail19 : List Nat := 3391 :: root18Tail20
def root18Tail18 : List Nat := 2749 :: root18Tail19
def root18Tail17 : List Nat := 2689 :: root18Tail18
def root18Tail16 : List Nat := 2617 :: root18Tail17
def root18Tail15 : List Nat := 2521 :: root18Tail16
def root18Tail14 : List Nat := 2467 :: root18Tail15
def root18Tail13 : List Nat := 2293 :: root18Tail14
def root18Tail12 : List Nat := 2287 :: root18Tail13
def root18Tail11 : List Nat := 2029 :: root18Tail12
def root18Tail10 : List Nat := 1933 :: root18Tail11
def root18Tail9 : List Nat := 1597 :: root18Tail10
def root18Tail8 : List Nat := 1447 :: root18Tail9
def root18Tail7 : List Nat := 1117 :: root18Tail8
def root18Tail6 : List Nat := 823 :: root18Tail7
def root18Tail5 : List Nat := 757 :: root18Tail6
def root18Tail4 : List Nat := 643 :: root18Tail5
def root18Tail3 : List Nat := 607 :: root18Tail4
def root18Tail2 : List Nat := 571 :: root18Tail3
def root18Tail1 : List Nat := 547 :: root18Tail2
def root18Tail0 : List Nat := 277 :: root18Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
