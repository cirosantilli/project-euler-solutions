import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root20Candidates : List Nat := Data.row83
set_option maxRecDepth 8192 in
theorem root20Candidates_checked : candidates edge 5 0 83 tail21 winner.bound = root20Candidates := by
  apply candidates_eq_row edge 5 0 83 tail21 root20Candidates winner.bound tail21_ordered
  · decide
  · decide
  · funext q
    rfl
def root20Tail102 : List Nat := []
def root20Tail101 : List Nat := 25307 :: root20Tail102
def root20Tail100 : List Nat := 25073 :: root20Tail101
def root20Tail99 : List Nat := 25013 :: root20Tail100
def root20Tail98 : List Nat := 24923 :: root20Tail99
def root20Tail97 : List Nat := 24023 :: root20Tail98
def root20Tail96 : List Nat := 23981 :: root20Tail97
def root20Tail95 : List Nat := 23873 :: root20Tail96
def root20Tail94 : List Nat := 23753 :: root20Tail95
def root20Tail93 : List Nat := 23201 :: root20Tail94
def root20Tail92 : List Nat := 22871 :: root20Tail93
def root20Tail91 : List Nat := 22679 :: root20Tail92
def root20Tail90 : List Nat := 22283 :: root20Tail91
def root20Tail89 : List Nat := 21503 :: root20Tail90
def root20Tail88 : List Nat := 20849 :: root20Tail89
def root20Tail87 : List Nat := 20693 :: root20Tail88
def root20Tail86 : List Nat := 20393 :: root20Tail87
def root20Tail85 : List Nat := 20177 :: root20Tail86
def root20Tail84 : List Nat := 20051 :: root20Tail85
def root20Tail83 : List Nat := 19841 :: root20Tail84
def root20Tail82 : List Nat := 19559 :: root20Tail83
def root20Tail81 : List Nat := 19391 :: root20Tail82
def root20Tail80 : List Nat := 19319 :: root20Tail81
def root20Tail79 : List Nat := 18413 :: root20Tail80
def root20Tail78 : List Nat := 18329 :: root20Tail79
def root20Tail77 : List Nat := 16619 :: root20Tail78
def root20Tail76 : List Nat := 16193 :: root20Tail77
def root20Tail75 : List Nat := 15773 :: root20Tail76
def root20Tail74 : List Nat := 15731 :: root20Tail75
def root20Tail73 : List Nat := 14897 :: root20Tail74
def root20Tail72 : List Nat := 13763 :: root20Tail73
def root20Tail71 : List Nat := 13457 :: root20Tail72
def root20Tail70 : List Nat := 12713 :: root20Tail71
def root20Tail69 : List Nat := 12377 :: root20Tail70
def root20Tail68 : List Nat := 12251 :: root20Tail69
def root20Tail67 : List Nat := 12197 :: root20Tail68
def root20Tail66 : List Nat := 12119 :: root20Tail67
def root20Tail65 : List Nat := 11909 :: root20Tail66
def root20Tail64 : List Nat := 11447 :: root20Tail65
def root20Tail63 : List Nat := 11279 :: root20Tail64
def root20Tail62 : List Nat := 10979 :: root20Tail63
def root20Tail61 : List Nat := 10781 :: root20Tail62
def root20Tail60 : List Nat := 10691 :: root20Tail61
def root20Tail59 : List Nat := 9887 :: root20Tail60
def root20Tail58 : List Nat := 9551 :: root20Tail59
def root20Tail57 : List Nat := 9539 :: root20Tail58
def root20Tail56 : List Nat := 9473 :: root20Tail57
def root20Tail55 : List Nat := 9437 :: root20Tail56
def root20Tail54 : List Nat := 9413 :: root20Tail55
def root20Tail53 : List Nat := 9161 :: root20Tail54
def root20Tail52 : List Nat := 8969 :: root20Tail53
def root20Tail51 : List Nat := 8837 :: root20Tail52
def root20Tail50 : List Nat := 8597 :: root20Tail51
def root20Tail49 : List Nat := 8171 :: root20Tail50
def root20Tail48 : List Nat := 8093 :: root20Tail49
def root20Tail47 : List Nat := 7583 :: root20Tail48
def root20Tail46 : List Nat := 7307 :: root20Tail47
def root20Tail45 : List Nat := 7283 :: root20Tail46
def root20Tail44 : List Nat := 7079 :: root20Tail45
def root20Tail43 : List Nat := 6491 :: root20Tail44
def root20Tail42 : List Nat := 6203 :: root20Tail43
def root20Tail41 : List Nat := 5987 :: root20Tail42
def root20Tail40 : List Nat := 5897 :: root20Tail41
def root20Tail39 : List Nat := 5717 :: root20Tail40
def root20Tail38 : List Nat := 5591 :: root20Tail39
def root20Tail37 : List Nat := 5441 :: root20Tail38
def root20Tail36 : List Nat := 5039 :: root20Tail37
def root20Tail35 : List Nat := 4643 :: root20Tail36
def root20Tail34 : List Nat := 3923 :: root20Tail35
def root20Tail33 : List Nat := 3659 :: root20Tail34
def root20Tail32 : List Nat := 3557 :: root20Tail33
def root20Tail31 : List Nat := 3491 :: root20Tail32
def root20Tail30 : List Nat := 3347 :: root20Tail31
def root20Tail29 : List Nat := 3191 :: root20Tail30
def root20Tail28 : List Nat := 2969 :: root20Tail29
def root20Tail27 : List Nat := 2801 :: root20Tail28
def root20Tail26 : List Nat := 2729 :: root20Tail27
def root20Tail25 : List Nat := 2693 :: root20Tail26
def root20Tail24 : List Nat := 2687 :: root20Tail25
def root20Tail23 : List Nat := 2633 :: root20Tail24
def root20Tail22 : List Nat := 2591 :: root20Tail23
def root20Tail21 : List Nat := 2411 :: root20Tail22
def root20Tail20 : List Nat := 2339 :: root20Tail21
def root20Tail19 : List Nat := 2063 :: root20Tail20
def root20Tail18 : List Nat := 2003 :: root20Tail19
def root20Tail17 : List Nat := 1889 :: root20Tail18
def root20Tail16 : List Nat := 1811 :: root20Tail17
def root20Tail15 : List Nat := 1697 :: root20Tail16
def root20Tail14 : List Nat := 1619 :: root20Tail15
def root20Tail13 : List Nat := 1553 :: root20Tail14
def root20Tail12 : List Nat := 1409 :: root20Tail13
def root20Tail11 : List Nat := 1373 :: root20Tail12
def root20Tail10 : List Nat := 1301 :: root20Tail11
def root20Tail9 : List Nat := 1031 :: root20Tail10
def root20Tail8 : List Nat := 911 :: root20Tail9
def root20Tail7 : List Nat := 773 :: root20Tail8
def root20Tail6 : List Nat := 719 :: root20Tail7
def root20Tail5 : List Nat := 701 :: root20Tail6
def root20Tail4 : List Nat := 563 :: root20Tail5
def root20Tail3 : List Nat := 449 :: root20Tail4
def root20Tail2 : List Nat := 443 :: root20Tail3
def root20Tail1 : List Nat := 311 :: root20Tail2
def root20Tail0 : List Nat := 227 :: root20Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
