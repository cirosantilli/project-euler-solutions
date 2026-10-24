import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root17Candidates : List Nat := Data.row71
set_option maxRecDepth 8192 in
theorem root17Candidates_checked : candidates edge 5 0 71 tail18 winner.bound = root17Candidates := by
  apply candidates_eq_row edge 5 0 71 tail18 root17Candidates winner.bound tail18_ordered
  · decide
  · decide
  · funext q
    rfl
def root17Tail103 : List Nat := []
def root17Tail102 : List Nat := 25673 :: root17Tail103
def root17Tail101 : List Nat := 25169 :: root17Tail102
def root17Tail100 : List Nat := 25121 :: root17Tail101
def root17Tail99 : List Nat := 24977 :: root17Tail100
def root17Tail98 : List Nat := 24179 :: root17Tail99
def root17Tail97 : List Nat := 23981 :: root17Tail98
def root17Tail96 : List Nat := 22961 :: root17Tail97
def root17Tail95 : List Nat := 22691 :: root17Tail96
def root17Tail94 : List Nat := 22679 :: root17Tail95
def root17Tail93 : List Nat := 22469 :: root17Tail94
def root17Tail92 : List Nat := 22091 :: root17Tail93
def root17Tail91 : List Nat := 22013 :: root17Tail92
def root17Tail90 : List Nat := 21767 :: root17Tail91
def root17Tail89 : List Nat := 21377 :: root17Tail90
def root17Tail88 : List Nat := 20123 :: root17Tail89
def root17Tail87 : List Nat := 19403 :: root17Tail88
def root17Tail86 : List Nat := 17939 :: root17Tail87
def root17Tail85 : List Nat := 17729 :: root17Tail86
def root17Tail84 : List Nat := 17183 :: root17Tail85
def root17Tail83 : List Nat := 17021 :: root17Tail84
def root17Tail82 : List Nat := 16871 :: root17Tail83
def root17Tail81 : List Nat := 15971 :: root17Tail82
def root17Tail80 : List Nat := 15737 :: root17Tail81
def root17Tail79 : List Nat := 14957 :: root17Tail80
def root17Tail78 : List Nat := 14753 :: root17Tail79
def root17Tail77 : List Nat := 14621 :: root17Tail78
def root17Tail76 : List Nat := 13907 :: root17Tail77
def root17Tail75 : List Nat := 13721 :: root17Tail76
def root17Tail74 : List Nat := 13553 :: root17Tail75
def root17Tail73 : List Nat := 12983 :: root17Tail74
def root17Tail72 : List Nat := 12041 :: root17Tail73
def root17Tail71 : List Nat := 12011 :: root17Tail72
def root17Tail70 : List Nat := 11789 :: root17Tail71
def root17Tail69 : List Nat := 11549 :: root17Tail70
def root17Tail68 : List Nat := 11411 :: root17Tail69
def root17Tail67 : List Nat := 11087 :: root17Tail68
def root17Tail66 : List Nat := 10847 :: root17Tail67
def root17Tail65 : List Nat := 9833 :: root17Tail66
def root17Tail64 : List Nat := 9791 :: root17Tail65
def root17Tail63 : List Nat := 9749 :: root17Tail64
def root17Tail62 : List Nat := 9689 :: root17Tail63
def root17Tail61 : List Nat := 9419 :: root17Tail62
def root17Tail60 : List Nat := 9239 :: root17Tail61
def root17Tail59 : List Nat := 9203 :: root17Tail60
def root17Tail58 : List Nat := 9011 :: root17Tail59
def root17Tail57 : List Nat := 8999 :: root17Tail58
def root17Tail56 : List Nat := 8747 :: root17Tail57
def root17Tail55 : List Nat := 8423 :: root17Tail56
def root17Tail54 : List Nat := 8387 :: root17Tail55
def root17Tail53 : List Nat := 8087 :: root17Tail54
def root17Tail52 : List Nat := 7919 :: root17Tail53
def root17Tail51 : List Nat := 7841 :: root17Tail52
def root17Tail50 : List Nat := 7817 :: root17Tail51
def root17Tail49 : List Nat := 7589 :: root17Tail50
def root17Tail48 : List Nat := 7229 :: root17Tail49
def root17Tail47 : List Nat := 7151 :: root17Tail48
def root17Tail46 : List Nat := 7109 :: root17Tail47
def root17Tail45 : List Nat := 7103 :: root17Tail46
def root17Tail44 : List Nat := 7001 :: root17Tail45
def root17Tail43 : List Nat := 6869 :: root17Tail44
def root17Tail42 : List Nat := 6563 :: root17Tail43
def root17Tail41 : List Nat := 6389 :: root17Tail42
def root17Tail40 : List Nat := 5879 :: root17Tail41
def root17Tail39 : List Nat := 5849 :: root17Tail40
def root17Tail38 : List Nat := 5657 :: root17Tail39
def root17Tail37 : List Nat := 5651 :: root17Tail38
def root17Tail36 : List Nat := 5639 :: root17Tail37
def root17Tail35 : List Nat := 5441 :: root17Tail36
def root17Tail34 : List Nat := 5417 :: root17Tail35
def root17Tail33 : List Nat := 5237 :: root17Tail34
def root17Tail32 : List Nat := 5153 :: root17Tail33
def root17Tail31 : List Nat := 5087 :: root17Tail32
def root17Tail30 : List Nat := 4673 :: root17Tail31
def root17Tail29 : List Nat := 4517 :: root17Tail30
def root17Tail28 : List Nat := 3863 :: root17Tail29
def root17Tail27 : List Nat := 3833 :: root17Tail28
def root17Tail26 : List Nat := 3491 :: root17Tail27
def root17Tail25 : List Nat := 3347 :: root17Tail26
def root17Tail24 : List Nat := 2909 :: root17Tail25
def root17Tail23 : List Nat := 2837 :: root17Tail24
def root17Tail22 : List Nat := 2819 :: root17Tail23
def root17Tail21 : List Nat := 2687 :: root17Tail22
def root17Tail20 : List Nat := 2477 :: root17Tail21
def root17Tail19 : List Nat := 2417 :: root17Tail20
def root17Tail18 : List Nat := 1973 :: root17Tail19
def root17Tail17 : List Nat := 1709 :: root17Tail18
def root17Tail16 : List Nat := 1583 :: root17Tail17
def root17Tail15 : List Nat := 1499 :: root17Tail16
def root17Tail14 : List Nat := 1427 :: root17Tail15
def root17Tail13 : List Nat := 1181 :: root17Tail14
def root17Tail12 : List Nat := 1163 :: root17Tail13
def root17Tail11 : List Nat := 1049 :: root17Tail12
def root17Tail10 : List Nat := 971 :: root17Tail11
def root17Tail9 : List Nat := 947 :: root17Tail10
def root17Tail8 : List Nat := 887 :: root17Tail9
def root17Tail7 : List Nat := 821 :: root17Tail8
def root17Tail6 : List Nat := 719 :: root17Tail7
def root17Tail5 : List Nat := 443 :: root17Tail6
def root17Tail4 : List Nat := 389 :: root17Tail5
def root17Tail3 : List Nat := 317 :: root17Tail4
def root17Tail2 : List Nat := 263 :: root17Tail3
def root17Tail1 : List Nat := 257 :: root17Tail2
def root17Tail0 : List Nat := 233 :: root17Tail1
end ProjectEulerSolutions.PrimePairLarge.Search
