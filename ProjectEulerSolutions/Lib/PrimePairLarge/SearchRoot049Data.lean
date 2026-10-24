import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root49Candidates : List Nat := Data.row239
set_option maxRecDepth 8192 in
theorem root49Candidates_checked : candidates edge 5 0 239 tail50 winner.bound = root49Candidates := by
  apply candidates_eq_row edge 5 0 239 tail50 root49Candidates winner.bound tail50_ordered
  · decide
  · decide
  · funext q
    rfl
def root49Tail79 : List Nat := []
def root49Tail78 : List Nat := 25127 :: root49Tail79
def root49Tail77 : List Nat := 24917 :: root49Tail78
def root49Tail76 : List Nat := 24413 :: root49Tail77
def root49Tail75 : List Nat := 23957 :: root49Tail76
def root49Tail74 : List Nat := 23819 :: root49Tail75
def root49Tail73 : List Nat := 23531 :: root49Tail74
def root49Tail72 : List Nat := 23357 :: root49Tail73
def root49Tail71 : List Nat := 23099 :: root49Tail72
def root49Tail70 : List Nat := 22727 :: root49Tail71
def root49Tail69 : List Nat := 22193 :: root49Tail70
def root49Tail68 : List Nat := 21773 :: root49Tail69
def root49Tail67 : List Nat := 21383 :: root49Tail68
def root49Tail66 : List Nat := 21011 :: root49Tail67
def root49Tail65 : List Nat := 20201 :: root49Tail66
def root49Tail64 : List Nat := 20177 :: root49Tail65
def root49Tail63 : List Nat := 19319 :: root49Tail64
def root49Tail62 : List Nat := 18773 :: root49Tail63
def root49Tail61 : List Nat := 18743 :: root49Tail62
def root49Tail60 : List Nat := 17987 :: root49Tail61
def root49Tail59 : List Nat := 17909 :: root49Tail60
def root49Tail58 : List Nat := 17189 :: root49Tail59
def root49Tail57 : List Nat := 16691 :: root49Tail58
def root49Tail56 : List Nat := 16301 :: root49Tail57
def root49Tail55 : List Nat := 16007 :: root49Tail56
def root49Tail54 : List Nat := 15887 :: root49Tail55
def root49Tail53 : List Nat := 15299 :: root49Tail54
def root49Tail52 : List Nat := 15077 :: root49Tail53
def root49Tail51 : List Nat := 14207 :: root49Tail52
def root49Tail50 : List Nat := 14081 :: root49Tail51
def root49Tail49 : List Nat := 13103 :: root49Tail50
def root49Tail48 : List Nat := 11717 :: root49Tail49
def root49Tail47 : List Nat := 11057 :: root49Tail48
def root49Tail46 : List Nat := 10433 :: root49Tail47
def root49Tail45 : List Nat := 10247 :: root49Tail46
def root49Tail44 : List Nat := 10079 :: root49Tail45
def root49Tail43 : List Nat := 9539 :: root49Tail44
def root49Tail42 : List Nat := 8699 :: root49Tail43
def root49Tail41 : List Nat := 8573 :: root49Tail42
def root49Tail40 : List Nat := 8537 :: root49Tail41
def root49Tail39 : List Nat := 8243 :: root49Tail40
def root49Tail38 : List Nat := 8111 :: root49Tail39
def root49Tail37 : List Nat := 8087 :: root49Tail38
def root49Tail36 : List Nat := 7823 :: root49Tail37
def root49Tail35 : List Nat := 7541 :: root49Tail36
def root49Tail34 : List Nat := 7523 :: root49Tail35
def root49Tail33 : List Nat := 6659 :: root49Tail34
def root49Tail32 : List Nat := 6257 :: root49Tail33
def root49Tail31 : List Nat := 6197 :: root49Tail32
def root49Tail30 : List Nat := 6131 :: root49Tail31
def root49Tail29 : List Nat := 5927 :: root49Tail30
def root49Tail28 : List Nat := 5879 :: root49Tail29
def root49Tail27 : List Nat := 5807 :: root49Tail28
def root49Tail26 : List Nat := 5303 :: root49Tail27
def root49Tail25 : List Nat := 5039 :: root49Tail26
def root49Tail24 : List Nat := 4787 :: root49Tail25
def root49Tail23 : List Nat := 4241 :: root49Tail24
def root49Tail22 : List Nat := 4157 :: root49Tail23
def root49Tail21 : List Nat := 3761 :: root49Tail22
def root49Tail20 : List Nat := 3467 :: root49Tail21
def root49Tail19 : List Nat := 3257 :: root49Tail20
def root49Tail18 : List Nat := 2861 :: root49Tail19
def root49Tail17 : List Nat := 2729 :: root49Tail18
def root49Tail16 : List Nat := 2459 :: root49Tail17
def root49Tail15 : List Nat := 2447 :: root49Tail16
def root49Tail14 : List Nat := 2333 :: root49Tail15
def root49Tail13 : List Nat := 1847 :: root49Tail14
def root49Tail12 : List Nat := 1637 :: root49Tail13
def root49Tail11 : List Nat := 1439 :: root49Tail12
def root49Tail10 : List Nat := 1091 :: root49Tail11
def root49Tail9 : List Nat := 1049 :: root49Tail10
def root49Tail8 : List Nat := 1013 :: root49Tail9
def root49Tail7 : List Nat := 977 :: root49Tail8
def root49Tail6 : List Nat := 947 :: root49Tail7
def root49Tail5 : List Nat := 929 :: root49Tail6
def root49Tail4 : List Nat := 641 :: root49Tail5
def root49Tail3 : List Nat := 509 :: root49Tail4
def root49Tail2 : List Nat := 461 :: root49Tail3
def root49Tail1 : List Nat := 347 :: root49Tail2
def root49Tail0 : List Nat := 263 :: root49Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
