import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root15Candidates : List Nat := Data.row61
set_option maxRecDepth 8192 in
theorem root15Candidates_checked : candidates edge 5 0 61 tail16 winner.bound = root15Candidates := by
  apply candidates_eq_row edge 5 0 61 tail16 root15Candidates winner.bound tail16_ordered
  · decide
  · decide
  · funext q
    rfl
def root15Tail104 : List Nat := []
def root15Tail103 : List Nat := 25939 :: root15Tail104
def root15Tail102 : List Nat := 25447 :: root15Tail103
def root15Tail101 : List Nat := 25309 :: root15Tail102
def root15Tail100 : List Nat := 25303 :: root15Tail101
def root15Tail99 : List Nat := 24889 :: root15Tail100
def root15Tail98 : List Nat := 24793 :: root15Tail99
def root15Tail97 : List Nat := 24691 :: root15Tail98
def root15Tail96 : List Nat := 23629 :: root15Tail97
def root15Tail95 : List Nat := 23203 :: root15Tail96
def root15Tail94 : List Nat := 22447 :: root15Tail95
def root15Tail93 : List Nat := 22381 :: root15Tail94
def root15Tail92 : List Nat := 22273 :: root15Tail93
def root15Tail91 : List Nat := 21727 :: root15Tail92
def root15Tail90 : List Nat := 21487 :: root15Tail91
def root15Tail89 : List Nat := 21121 :: root15Tail90
def root15Tail88 : List Nat := 20809 :: root15Tail89
def root15Tail87 : List Nat := 20533 :: root15Tail88
def root15Tail86 : List Nat := 20347 :: root15Tail87
def root15Tail85 : List Nat := 19543 :: root15Tail86
def root15Tail84 : List Nat := 19423 :: root15Tail85
def root15Tail83 : List Nat := 18517 :: root15Tail84
def root15Tail82 : List Nat := 17959 :: root15Tail83
def root15Tail81 : List Nat := 17581 :: root15Tail82
def root15Tail80 : List Nat := 17299 :: root15Tail81
def root15Tail79 : List Nat := 17239 :: root15Tail80
def root15Tail78 : List Nat := 17203 :: root15Tail79
def root15Tail77 : List Nat := 16921 :: root15Tail78
def root15Tail76 : List Nat := 16657 :: root15Tail77
def root15Tail75 : List Nat := 16567 :: root15Tail76
def root15Tail74 : List Nat := 16273 :: root15Tail75
def root15Tail73 : List Nat := 16111 :: root15Tail74
def root15Tail72 : List Nat := 16057 :: root15Tail73
def root15Tail71 : List Nat := 15973 :: root15Tail72
def root15Tail70 : List Nat := 15643 :: root15Tail71
def root15Tail69 : List Nat := 14887 :: root15Tail70
def root15Tail68 : List Nat := 13729 :: root15Tail69
def root15Tail67 : List Nat := 13309 :: root15Tail68
def root15Tail66 : List Nat := 13297 :: root15Tail67
def root15Tail65 : List Nat := 12763 :: root15Tail66
def root15Tail64 : List Nat := 12703 :: root15Tail65
def root15Tail63 : List Nat := 12553 :: root15Tail64
def root15Tail62 : List Nat := 12541 :: root15Tail63
def root15Tail61 : List Nat := 11839 :: root15Tail62
def root15Tail60 : List Nat := 11617 :: root15Tail61
def root15Tail59 : List Nat := 11587 :: root15Tail60
def root15Tail58 : List Nat := 11527 :: root15Tail59
def root15Tail57 : List Nat := 10861 :: root15Tail58
def root15Tail56 : List Nat := 9511 :: root15Tail57
def root15Tail55 : List Nat := 9181 :: root15Tail56
def root15Tail54 : List Nat := 9007 :: root15Tail55
def root15Tail53 : List Nat := 8941 :: root15Tail54
def root15Tail52 : List Nat := 8581 :: root15Tail53
def root15Tail51 : List Nat := 8311 :: root15Tail52
def root15Tail50 : List Nat := 8161 :: root15Tail51
def root15Tail49 : List Nat := 7963 :: root15Tail50
def root15Tail48 : List Nat := 7951 :: root15Tail49
def root15Tail47 : List Nat := 7873 :: root15Tail48
def root15Tail46 : List Nat := 7699 :: root15Tail47
def root15Tail45 : List Nat := 7681 :: root15Tail46
def root15Tail44 : List Nat := 7369 :: root15Tail45
def root15Tail43 : List Nat := 7237 :: root15Tail44
def root15Tail42 : List Nat := 7129 :: root15Tail43
def root15Tail41 : List Nat := 6997 :: root15Tail42
def root15Tail40 : List Nat := 6871 :: root15Tail41
def root15Tail39 : List Nat := 6793 :: root15Tail40
def root15Tail38 : List Nat := 6673 :: root15Tail39
def root15Tail37 : List Nat := 6367 :: root15Tail38
def root15Tail36 : List Nat := 6079 :: root15Tail37
def root15Tail35 : List Nat := 5869 :: root15Tail36
def root15Tail34 : List Nat := 5827 :: root15Tail35
def root15Tail33 : List Nat := 5821 :: root15Tail34
def root15Tail32 : List Nat := 5623 :: root15Tail33
def root15Tail31 : List Nat := 5557 :: root15Tail32
def root15Tail30 : List Nat := 5449 :: root15Tail31
def root15Tail29 : List Nat := 5431 :: root15Tail30
def root15Tail28 : List Nat := 5413 :: root15Tail29
def root15Tail27 : List Nat := 4657 :: root15Tail28
def root15Tail26 : List Nat := 3889 :: root15Tail27
def root15Tail25 : List Nat := 3733 :: root15Tail26
def root15Tail24 : List Nat := 3637 :: root15Tail25
def root15Tail23 : List Nat := 3469 :: root15Tail24
def root15Tail22 : List Nat := 3463 :: root15Tail23
def root15Tail21 : List Nat := 3181 :: root15Tail22
def root15Tail20 : List Nat := 3169 :: root15Tail21
def root15Tail19 : List Nat := 3049 :: root15Tail20
def root15Tail18 : List Nat := 2971 :: root15Tail19
def root15Tail17 : List Nat := 2797 :: root15Tail18
def root15Tail16 : List Nat := 2383 :: root15Tail17
def root15Tail15 : List Nat := 2371 :: root15Tail16
def root15Tail14 : List Nat := 2341 :: root15Tail15
def root15Tail13 : List Nat := 1999 :: root15Tail14
def root15Tail12 : List Nat := 1951 :: root15Tail13
def root15Tail11 : List Nat := 1873 :: root15Tail12
def root15Tail10 : List Nat := 1801 :: root15Tail11
def root15Tail9 : List Nat := 1753 :: root15Tail10
def root15Tail8 : List Nat := 1693 :: root15Tail9
def root15Tail7 : List Nat := 1483 :: root15Tail8
def root15Tail6 : List Nat := 1453 :: root15Tail7
def root15Tail5 : List Nat := 1069 :: root15Tail6
def root15Tail4 : List Nat := 751 :: root15Tail5
def root15Tail3 : List Nat := 487 :: root15Tail4
def root15Tail2 : List Nat := 409 :: root15Tail3
def root15Tail1 : List Nat := 331 :: root15Tail2
def root15Tail0 : List Nat := 151 :: root15Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
