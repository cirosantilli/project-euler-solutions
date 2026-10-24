import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root16Candidates : List Nat := Data.row67
set_option maxRecDepth 8192 in
theorem root16Candidates_checked : candidates edge 5 0 67 tail17 winner.bound = root16Candidates := by
  apply candidates_eq_row edge 5 0 67 tail17 root16Candidates winner.bound tail17_ordered
  · decide
  · decide
  · funext q
    rfl
def root16Tail105 : List Nat := []
def root16Tail104 : List Nat := 25951 :: root16Tail105
def root16Tail103 : List Nat := 25633 :: root16Tail104
def root16Tail102 : List Nat := 25339 :: root16Tail103
def root16Tail101 : List Nat := 25153 :: root16Tail102
def root16Tail100 : List Nat := 24247 :: root16Tail101
def root16Tail99 : List Nat := 24007 :: root16Tail100
def root16Tail98 : List Nat := 23971 :: root16Tail99
def root16Tail97 : List Nat := 23071 :: root16Tail98
def root16Tail96 : List Nat := 22717 :: root16Tail97
def root16Tail95 : List Nat := 22291 :: root16Tail96
def root16Tail94 : List Nat := 22273 :: root16Tail95
def root16Tail93 : List Nat := 22129 :: root16Tail94
def root16Tail92 : List Nat := 22039 :: root16Tail93
def root16Tail91 : List Nat := 21961 :: root16Tail92
def root16Tail90 : List Nat := 21739 :: root16Tail91
def root16Tail89 : List Nat := 21661 :: root16Tail90
def root16Tail88 : List Nat := 21577 :: root16Tail89
def root16Tail87 : List Nat := 21157 :: root16Tail88
def root16Tail86 : List Nat := 20947 :: root16Tail87
def root16Tail85 : List Nat := 20773 :: root16Tail86
def root16Tail84 : List Nat := 19861 :: root16Tail85
def root16Tail83 : List Nat := 19501 :: root16Tail84
def root16Tail82 : List Nat := 19231 :: root16Tail83
def root16Tail81 : List Nat := 18379 :: root16Tail82
def root16Tail80 : List Nat := 18223 :: root16Tail81
def root16Tail79 : List Nat := 18211 :: root16Tail80
def root16Tail78 : List Nat := 17959 :: root16Tail79
def root16Tail77 : List Nat := 17341 :: root16Tail78
def root16Tail76 : List Nat := 17077 :: root16Tail77
def root16Tail75 : List Nat := 16453 :: root16Tail76
def root16Tail74 : List Nat := 16363 :: root16Tail75
def root16Tail73 : List Nat := 15913 :: root16Tail74
def root16Tail72 : List Nat := 15859 :: root16Tail73
def root16Tail71 : List Nat := 14503 :: root16Tail72
def root16Tail70 : List Nat := 14107 :: root16Tail71
def root16Tail69 : List Nat := 12457 :: root16Tail70
def root16Tail68 : List Nat := 12421 :: root16Tail69
def root16Tail67 : List Nat := 11491 :: root16Tail68
def root16Tail66 : List Nat := 11251 :: root16Tail67
def root16Tail65 : List Nat := 11119 :: root16Tail66
def root16Tail64 : List Nat := 11071 :: root16Tail65
def root16Tail63 : List Nat := 10909 :: root16Tail64
def root16Tail62 : List Nat := 10723 :: root16Tail63
def root16Tail61 : List Nat := 10159 :: root16Tail62
def root16Tail60 : List Nat := 9907 :: root16Tail61
def root16Tail59 : List Nat := 9883 :: root16Tail60
def root16Tail58 : List Nat := 9829 :: root16Tail59
def root16Tail57 : List Nat := 9733 :: root16Tail58
def root16Tail56 : List Nat := 9463 :: root16Tail57
def root16Tail55 : List Nat := 9433 :: root16Tail56
def root16Tail54 : List Nat := 9319 :: root16Tail55
def root16Tail53 : List Nat := 9067 :: root16Tail54
def root16Tail52 : List Nat := 8941 :: root16Tail53
def root16Tail51 : List Nat := 8641 :: root16Tail52
def root16Tail50 : List Nat := 8581 :: root16Tail51
def root16Tail49 : List Nat := 8221 :: root16Tail50
def root16Tail48 : List Nat := 8191 :: root16Tail49
def root16Tail47 : List Nat := 7687 :: root16Tail48
def root16Tail46 : List Nat := 7681 :: root16Tail47
def root16Tail45 : List Nat := 7639 :: root16Tail46
def root16Tail44 : List Nat := 7561 :: root16Tail45
def root16Tail43 : List Nat := 6829 :: root16Tail44
def root16Tail42 : List Nat := 6661 :: root16Tail43
def root16Tail41 : List Nat := 6337 :: root16Tail42
def root16Tail40 : List Nat := 5923 :: root16Tail41
def root16Tail39 : List Nat := 5881 :: root16Tail40
def root16Tail38 : List Nat := 5827 :: root16Tail39
def root16Tail37 : List Nat := 5743 :: root16Tail38
def root16Tail36 : List Nat := 5581 :: root16Tail37
def root16Tail35 : List Nat := 5569 :: root16Tail36
def root16Tail34 : List Nat := 5419 :: root16Tail35
def root16Tail33 : List Nat := 5179 :: root16Tail34
def root16Tail32 : List Nat := 4987 :: root16Tail33
def root16Tail31 : List Nat := 4903 :: root16Tail32
def root16Tail30 : List Nat := 4831 :: root16Tail31
def root16Tail29 : List Nat := 4789 :: root16Tail30
def root16Tail28 : List Nat := 4483 :: root16Tail29
def root16Tail27 : List Nat := 4099 :: root16Tail28
def root16Tail26 : List Nat := 4057 :: root16Tail27
def root16Tail25 : List Nat := 3943 :: root16Tail26
def root16Tail24 : List Nat := 3637 :: root16Tail25
def root16Tail23 : List Nat := 3499 :: root16Tail24
def root16Tail22 : List Nat := 3313 :: root16Tail23
def root16Tail21 : List Nat := 2767 :: root16Tail22
def root16Tail20 : List Nat := 2557 :: root16Tail21
def root16Tail19 : List Nat := 2377 :: root16Tail20
def root16Tail18 : List Nat := 2341 :: root16Tail19
def root16Tail17 : List Nat := 2251 :: root16Tail18
def root16Tail16 : List Nat := 2029 :: root16Tail17
def root16Tail15 : List Nat := 1999 :: root16Tail16
def root16Tail14 : List Nat := 1933 :: root16Tail15
def root16Tail13 : List Nat := 1831 :: root16Tail14
def root16Tail12 : List Nat := 1609 :: root16Tail13
def root16Tail11 : List Nat := 1303 :: root16Tail12
def root16Tail10 : List Nat := 1201 :: root16Tail11
def root16Tail9 : List Nat := 1093 :: root16Tail10
def root16Tail8 : List Nat := 1063 :: root16Tail9
def root16Tail7 : List Nat := 1039 :: root16Tail8
def root16Tail6 : List Nat := 757 :: root16Tail7
def root16Tail5 : List Nat := 751 :: root16Tail6
def root16Tail4 : List Nat := 619 :: root16Tail5
def root16Tail3 : List Nat := 601 :: root16Tail4
def root16Tail2 : List Nat := 547 :: root16Tail3
def root16Tail1 : List Nat := 157 :: root16Tail2
def root16Tail0 : List Nat := 139 :: root16Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
