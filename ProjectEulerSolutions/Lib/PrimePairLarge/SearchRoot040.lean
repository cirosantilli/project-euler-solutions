import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root40Candidates : List Nat := Data.row191
set_option maxRecDepth 8192 in
theorem root40Candidates_checked : candidates edge 5 0 191 tail41 winner.bound = root40Candidates := by
  apply candidates_eq_row edge 5 0 191 tail41 root40Candidates winner.bound tail41_ordered
  · decide
  · decide
  · funext q
    rfl
def root40Tail78 : List Nat := []
def root40Tail77 : List Nat := 25523 :: root40Tail78
def root40Tail76 : List Nat := 24239 :: root40Tail77
def root40Tail75 : List Nat := 23801 :: root40Tail76
def root40Tail74 : List Nat := 22907 :: root40Tail75
def root40Tail73 : List Nat := 22751 :: root40Tail74
def root40Tail72 : List Nat := 22613 :: root40Tail73
def root40Tail71 : List Nat := 22541 :: root40Tail72
def root40Tail70 : List Nat := 22079 :: root40Tail71
def root40Tail69 : List Nat := 21881 :: root40Tail70
def root40Tail68 : List Nat := 21803 :: root40Tail69
def root40Tail67 : List Nat := 20807 :: root40Tail68
def root40Tail66 : List Nat := 20201 :: root40Tail67
def root40Tail65 : List Nat := 19949 :: root40Tail66
def root40Tail64 : List Nat := 19853 :: root40Tail65
def root40Tail63 : List Nat := 18311 :: root40Tail64
def root40Tail62 : List Nat := 18119 :: root40Tail63
def root40Tail61 : List Nat := 17783 :: root40Tail62
def root40Tail60 : List Nat := 17729 :: root40Tail61
def root40Tail59 : List Nat := 17573 :: root40Tail60
def root40Tail58 : List Nat := 17321 :: root40Tail59
def root40Tail57 : List Nat := 17291 :: root40Tail58
def root40Tail56 : List Nat := 16787 :: root40Tail57
def root40Tail55 : List Nat := 16763 :: root40Tail56
def root40Tail54 : List Nat := 16631 :: root40Tail55
def root40Tail53 : List Nat := 16217 :: root40Tail54
def root40Tail52 : List Nat := 15923 :: root40Tail53
def root40Tail51 : List Nat := 15641 :: root40Tail52
def root40Tail50 : List Nat := 14591 :: root40Tail51
def root40Tail49 : List Nat := 14369 :: root40Tail50
def root40Tail48 : List Nat := 13649 :: root40Tail49
def root40Tail47 : List Nat := 13397 :: root40Tail48
def root40Tail46 : List Nat := 13187 :: root40Tail47
def root40Tail45 : List Nat := 12923 :: root40Tail46
def root40Tail44 : List Nat := 12911 :: root40Tail45
def root40Tail43 : List Nat := 12251 :: root40Tail44
def root40Tail42 : List Nat := 11987 :: root40Tail43
def root40Tail41 : List Nat := 11933 :: root40Tail42
def root40Tail40 : List Nat := 11471 :: root40Tail41
def root40Tail39 : List Nat := 11243 :: root40Tail40
def root40Tail38 : List Nat := 10781 :: root40Tail39
def root40Tail37 : List Nat := 10151 :: root40Tail38
def root40Tail36 : List Nat := 9677 :: root40Tail37
def root40Tail35 : List Nat := 9479 :: root40Tail36
def root40Tail34 : List Nat := 9461 :: root40Tail35
def root40Tail33 : List Nat := 8933 :: root40Tail34
def root40Tail32 : List Nat := 8747 :: root40Tail33
def root40Tail31 : List Nat := 8237 :: root40Tail32
def root40Tail30 : List Nat := 6917 :: root40Tail31
def root40Tail29 : List Nat := 6599 :: root40Tail30
def root40Tail28 : List Nat := 6311 :: root40Tail29
def root40Tail27 : List Nat := 6287 :: root40Tail28
def root40Tail26 : List Nat := 5981 :: root40Tail27
def root40Tail25 : List Nat := 5843 :: root40Tail26
def root40Tail24 : List Nat := 5813 :: root40Tail25
def root40Tail23 : List Nat := 5153 :: root40Tail24
def root40Tail22 : List Nat := 5099 :: root40Tail23
def root40Tail21 : List Nat := 4691 :: root40Tail22
def root40Tail20 : List Nat := 4673 :: root40Tail21
def root40Tail19 : List Nat := 4259 :: root40Tail20
def root40Tail18 : List Nat := 4127 :: root40Tail19
def root40Tail17 : List Nat := 3581 :: root40Tail18
def root40Tail16 : List Nat := 3251 :: root40Tail17
def root40Tail15 : List Nat := 2903 :: root40Tail16
def root40Tail14 : List Nat := 2423 :: root40Tail15
def root40Tail13 : List Nat := 2213 :: root40Tail14
def root40Tail12 : List Nat := 2069 :: root40Tail13
def root40Tail11 : List Nat := 1697 :: root40Tail12
def root40Tail10 : List Nat := 1583 :: root40Tail11
def root40Tail9 : List Nat := 1373 :: root40Tail10
def root40Tail8 : List Nat := 1319 :: root40Tail9
def root40Tail7 : List Nat := 1103 :: root40Tail8
def root40Tail6 : List Nat := 977 :: root40Tail7
def root40Tail5 : List Nat := 953 :: root40Tail6
def root40Tail4 : List Nat := 599 :: root40Tail5
def root40Tail3 : List Nat := 461 :: root40Tail4
def root40Tail2 : List Nat := 281 :: root40Tail3
def root40Tail1 : List Nat := 251 :: root40Tail2
def root40Tail0 : List Nat := 227 :: root40Tail1
def root40Candidates0 : List Nat := [281, 2903, 4691, 22079, 23801]
set_option maxRecDepth 8192 in
theorem root40Candidates0_checked : candidates edge 4 191 227 root40Tail1 winner.bound = root40Candidates0 := by
  unfold candidates
  have row : edge 227 = (SortedNatTree.contains FastRows.row227) := by
    change Data.row227.contains = (SortedNatTree.contains FastRows.row227)
    exact FastRows.row227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child0 : search edge 3 [191, 227] root40Candidates0 418 winner = winner := by decide
theorem root40Step0 : scan edge (search edge 3) 4 [191] 191 root40Tail0 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail1 winner := by
  rw [root40Tail0, scan, if_neg (by decide), root40Candidates0_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail1
    (if 3 ≤ root40Candidates0.length then search edge 3 [191, 227] root40Candidates0 418 winner else winner) = _
  rw [if_pos (by decide), root40Child0]
def root40Candidates1 : List Nat := [8933, 12911, 15641, 22751]
set_option maxRecDepth 8192 in
theorem root40Candidates1_checked : candidates edge 4 191 251 root40Tail2 winner.bound = root40Candidates1 := by
  unfold candidates
  have row : edge 251 = (SortedNatTree.contains FastRows.row251) := by
    change Data.row251.contains = (SortedNatTree.contains FastRows.row251)
    exact FastRows.row251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child1 : search edge 3 [191, 251] root40Candidates1 442 winner = winner := by decide
theorem root40Step1 : scan edge (search edge 3) 4 [191] 191 root40Tail1 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail2 winner := by
  rw [root40Tail1, scan, if_neg (by decide), root40Candidates1_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail2
    (if 3 ≤ root40Candidates1.length then search edge 3 [191, 251] root40Candidates1 442 winner else winner) = _
  rw [if_pos (by decide), root40Child1]
def root40Candidates2 : List Nat := [5843, 21803]
set_option maxRecDepth 8192 in
theorem root40Candidates2_checked : candidates edge 4 191 281 root40Tail3 winner.bound = root40Candidates2 := by
  unfold candidates
  have row : edge 281 = (SortedNatTree.contains FastRows.row281) := by
    change Data.row281.contains = (SortedNatTree.contains FastRows.row281)
    exact FastRows.row281_checked.symm
  rw [row]
  decide
theorem root40Step2 : scan edge (search edge 3) 4 [191] 191 root40Tail2 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail3 winner := by
  rw [root40Tail2, scan, if_neg (by decide), root40Candidates2_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail3
    (if 3 ≤ root40Candidates2.length then search edge 3 [191, 281] root40Candidates2 472 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates3 : List Nat := [1697, 10781]
set_option maxRecDepth 8192 in
theorem root40Candidates3_checked : candidates edge 4 191 461 root40Tail4 winner.bound = root40Candidates3 := by
  unfold candidates
  have row : edge 461 = (SortedNatTree.contains FastRows.row461) := by
    change Data.row461.contains = (SortedNatTree.contains FastRows.row461)
    exact FastRows.row461_checked.symm
  rw [row]
  decide
theorem root40Step3 : scan edge (search edge 3) 4 [191] 191 root40Tail3 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail4 winner := by
  rw [root40Tail3, scan, if_neg (by decide), root40Candidates3_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail4
    (if 3 ≤ root40Candidates3.length then search edge 3 [191, 461] root40Candidates3 652 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates4 : List Nat := [9479, 15923, 21803]
set_option maxRecDepth 8192 in
theorem root40Candidates4_checked : candidates edge 4 191 599 root40Tail5 winner.bound = root40Candidates4 := by
  unfold candidates
  have row : edge 599 = (SortedNatTree.contains FastRows.row599) := by
    change Data.row599.contains = (SortedNatTree.contains FastRows.row599)
    exact FastRows.row599_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child4 : search edge 3 [191, 599] root40Candidates4 790 winner = winner := by decide
theorem root40Step4 : scan edge (search edge 3) 4 [191] 191 root40Tail4 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail5 winner := by
  rw [root40Tail4, scan, if_neg (by decide), root40Candidates4_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail5
    (if 3 ≤ root40Candidates4.length then search edge 3 [191, 599] root40Candidates4 790 winner else winner) = _
  rw [if_pos (by decide), root40Child4]
def root40Candidates5 : List Nat := [13649, 14591]
set_option maxRecDepth 8192 in
theorem root40Candidates5_checked : candidates edge 4 191 953 root40Tail6 winner.bound = root40Candidates5 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root40Step5 : scan edge (search edge 3) 4 [191] 191 root40Tail5 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail6 winner := by
  rw [root40Tail5, scan, if_neg (by decide), root40Candidates5_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail6
    (if 3 ≤ root40Candidates5.length then search edge 3 [191, 953] root40Candidates5 1144 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates6 : List Nat := [16763, 17573, 19949]
set_option maxRecDepth 8192 in
theorem root40Candidates6_checked : candidates edge 4 191 977 root40Tail7 winner.bound = root40Candidates6 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child6 : search edge 3 [191, 977] root40Candidates6 1168 winner = winner := by decide
theorem root40Step6 : scan edge (search edge 3) 4 [191] 191 root40Tail6 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail7 winner := by
  rw [root40Tail6, scan, if_neg (by decide), root40Candidates6_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail7
    (if 3 ≤ root40Candidates6.length then search edge 3 [191, 977] root40Candidates6 1168 winner else winner) = _
  rw [if_pos (by decide), root40Child6]
def root40Candidates7 : List Nat := [1373, 5813, 15641]
set_option maxRecDepth 8192 in
theorem root40Candidates7_checked : candidates edge 4 191 1103 root40Tail8 winner.bound = root40Candidates7 := by
  unfold candidates
  have row : edge 1103 = (SortedNatTree.contains FastRows.row1103) := by
    change Data.row1103.contains = (SortedNatTree.contains FastRows.row1103)
    exact FastRows.row1103_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child7 : search edge 3 [191, 1103] root40Candidates7 1294 winner = winner := by decide
theorem root40Step7 : scan edge (search edge 3) 4 [191] 191 root40Tail7 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail8 winner := by
  rw [root40Tail7, scan, if_neg (by decide), root40Candidates7_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail8
    (if 3 ≤ root40Candidates7.length then search edge 3 [191, 1103] root40Candidates7 1294 winner else winner) = _
  rw [if_pos (by decide), root40Child7]
def root40Candidates8 : List Nat := [5843, 5981, 11933, 20807, 22613]
set_option maxRecDepth 8192 in
theorem root40Candidates8_checked : candidates edge 4 191 1319 root40Tail9 winner.bound = root40Candidates8 := by
  unfold candidates
  have row : edge 1319 = (SortedNatTree.contains FastRows.row1319) := by
    change Data.row1319.contains = (SortedNatTree.contains FastRows.row1319)
    exact FastRows.row1319_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child8 : search edge 3 [191, 1319] root40Candidates8 1510 winner = winner := by decide
theorem root40Step8 : scan edge (search edge 3) 4 [191] 191 root40Tail8 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail9 winner := by
  rw [root40Tail8, scan, if_neg (by decide), root40Candidates8_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail9
    (if 3 ≤ root40Candidates8.length then search edge 3 [191, 1319] root40Candidates8 1510 winner else winner) = _
  rw [if_pos (by decide), root40Child8]
def root40Candidates9 : List Nat := [2423, 14591]
set_option maxRecDepth 8192 in
theorem root40Candidates9_checked : candidates edge 4 191 1373 root40Tail10 winner.bound = root40Candidates9 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
theorem root40Step9 : scan edge (search edge 3) 4 [191] 191 root40Tail9 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail10 winner := by
  rw [root40Tail9, scan, if_neg (by decide), root40Candidates9_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail10
    (if 3 ≤ root40Candidates9.length then search edge 3 [191, 1373] root40Candidates9 1564 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates10 : List Nat := [6287, 19853]
set_option maxRecDepth 8192 in
theorem root40Candidates10_checked : candidates edge 4 191 1583 root40Tail11 winner.bound = root40Candidates10 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root40Step10 : scan edge (search edge 3) 4 [191] 191 root40Tail10 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail11 winner := by
  rw [root40Tail10, scan, if_neg (by decide), root40Candidates10_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail11
    (if 3 ≤ root40Candidates10.length then search edge 3 [191, 1583] root40Candidates10 1774 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates11 : List Nat := [5099, 8237]
set_option maxRecDepth 8192 in
theorem root40Candidates11_checked : candidates edge 4 191 1697 root40Tail12 winner.bound = root40Candidates11 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root40Step11 : scan edge (search edge 3) 4 [191] 191 root40Tail11 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail12 winner := by
  rw [root40Tail11, scan, if_neg (by decide), root40Candidates11_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail12
    (if 3 ≤ root40Candidates11.length then search edge 3 [191, 1697] root40Candidates11 1888 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates12 : List Nat := [4691, 8747, 17573, 20807]
set_option maxRecDepth 8192 in
theorem root40Candidates12_checked : candidates edge 4 191 2069 root40Tail13 winner.bound = root40Candidates12 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child12 : search edge 3 [191, 2069] root40Candidates12 2260 winner = winner := by decide
theorem root40Step12 : scan edge (search edge 3) 4 [191] 191 root40Tail12 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail13 winner := by
  rw [root40Tail12, scan, if_neg (by decide), root40Candidates12_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail13
    (if 3 ≤ root40Candidates12.length then search edge 3 [191, 2069] root40Candidates12 2260 winner else winner) = _
  rw [if_pos (by decide), root40Child12]
def root40Candidates13 : List Nat := [3581, 4691, 12251]
set_option maxRecDepth 8192 in
theorem root40Candidates13_checked : candidates edge 4 191 2213 root40Tail14 winner.bound = root40Candidates13 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child13 : search edge 3 [191, 2213] root40Candidates13 2404 winner = winner := by decide
theorem root40Step13 : scan edge (search edge 3) 4 [191] 191 root40Tail13 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail14 winner := by
  rw [root40Tail13, scan, if_neg (by decide), root40Candidates13_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail14
    (if 3 ≤ root40Candidates13.length then search edge 3 [191, 2213] root40Candidates13 2404 winner else winner) = _
  rw [if_pos (by decide), root40Child13]
def root40Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root40Candidates14_checked : candidates edge 4 191 2423 root40Tail15 winner.bound = root40Candidates14 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root40Step14 : scan edge (search edge 3) 4 [191] 191 root40Tail14 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail15 winner := by
  rw [root40Tail14, scan, if_neg (by decide), root40Candidates14_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail15
    (if 3 ≤ root40Candidates14.length then search edge 3 [191, 2423] root40Candidates14 2614 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates15 : List Nat := [19853, 20201]
set_option maxRecDepth 8192 in
theorem root40Candidates15_checked : candidates edge 4 191 2903 root40Tail16 winner.bound = root40Candidates15 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root40Step15 : scan edge (search edge 3) 4 [191] 191 root40Tail15 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail16 winner := by
  rw [root40Tail15, scan, if_neg (by decide), root40Candidates15_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail16
    (if 3 ≤ root40Candidates15.length then search edge 3 [191, 2903] root40Candidates15 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates16 : List Nat := [17321]
set_option maxRecDepth 8192 in
theorem root40Candidates16_checked : candidates edge 4 191 3251 root40Tail17 winner.bound = root40Candidates16 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root40Step16 : scan edge (search edge 3) 4 [191] 191 root40Tail16 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail17 winner := by
  rw [root40Tail16, scan, if_neg (by decide), root40Candidates16_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail17
    (if 3 ≤ root40Candidates16.length then search edge 3 [191, 3251] root40Candidates16 3442 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates17 : List Nat := [16217, 16763]
set_option maxRecDepth 8192 in
theorem root40Candidates17_checked : candidates edge 4 191 3581 root40Tail18 winner.bound = root40Candidates17 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root40Step17 : scan edge (search edge 3) 4 [191] 191 root40Tail17 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail18 winner := by
  rw [root40Tail17, scan, if_neg (by decide), root40Candidates17_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail18
    (if 3 ≤ root40Candidates17.length then search edge 3 [191, 3581] root40Candidates17 3772 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates18 : List Nat := [8933, 20807]
set_option maxRecDepth 8192 in
theorem root40Candidates18_checked : candidates edge 4 191 4127 root40Tail19 winner.bound = root40Candidates18 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root40Step18 : scan edge (search edge 3) 4 [191] 191 root40Tail18 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail19 winner := by
  rw [root40Tail18, scan, if_neg (by decide), root40Candidates18_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail19
    (if 3 ≤ root40Candidates18.length then search edge 3 [191, 4127] root40Candidates18 4318 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates19 : List Nat := [9677, 17729]
set_option maxRecDepth 8192 in
theorem root40Candidates19_checked : candidates edge 4 191 4259 root40Tail20 winner.bound = root40Candidates19 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root40Step19 : scan edge (search edge 3) 4 [191] 191 root40Tail19 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail20 winner := by
  rw [root40Tail19, scan, if_neg (by decide), root40Candidates19_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail20
    (if 3 ≤ root40Candidates19.length then search edge 3 [191, 4259] root40Candidates19 4450 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root40Candidates20_checked : candidates edge 4 191 4673 root40Tail21 winner.bound = root40Candidates20 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root40Step20 : scan edge (search edge 3) 4 [191] 191 root40Tail20 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail21 winner := by
  rw [root40Tail20, scan, if_neg (by decide), root40Candidates20_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail21
    (if 3 ≤ root40Candidates20.length then search edge 3 [191, 4673] root40Candidates20 4864 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates21 : List Nat := [18311]
set_option maxRecDepth 8192 in
theorem root40Candidates21_checked : candidates edge 4 191 4691 root40Tail22 winner.bound = root40Candidates21 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root40Step21 : scan edge (search edge 3) 4 [191] 191 root40Tail21 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail22 winner := by
  rw [root40Tail21, scan, if_neg (by decide), root40Candidates21_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail22
    (if 3 ≤ root40Candidates21.length then search edge 3 [191, 4691] root40Candidates21 4882 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root40Candidates22_checked : candidates edge 4 191 5099 root40Tail23 winner.bound = root40Candidates22 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root40Step22 : scan edge (search edge 3) 4 [191] 191 root40Tail22 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail23 winner := by
  rw [root40Tail22, scan, if_neg (by decide), root40Candidates22_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail23
    (if 3 ≤ root40Candidates22.length then search edge 3 [191, 5099] root40Candidates22 5290 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates23 : List Nat := [12251, 12923]
set_option maxRecDepth 8192 in
theorem root40Candidates23_checked : candidates edge 4 191 5153 root40Tail24 winner.bound = root40Candidates23 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root40Step23 : scan edge (search edge 3) 4 [191] 191 root40Tail23 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail24 winner := by
  rw [root40Tail23, scan, if_neg (by decide), root40Candidates23_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail24
    (if 3 ≤ root40Candidates23.length then search edge 3 [191, 5153] root40Candidates23 5344 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root40Candidates24_checked : candidates edge 4 191 5813 root40Tail25 winner.bound = root40Candidates24 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root40Step24 : scan edge (search edge 3) 4 [191] 191 root40Tail24 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail25 winner := by
  rw [root40Tail24, scan, if_neg (by decide), root40Candidates24_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail25
    (if 3 ≤ root40Candidates24.length then search edge 3 [191, 5813] root40Candidates24 6004 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates25 : List Nat := [16217]
set_option maxRecDepth 8192 in
theorem root40Candidates25_checked : candidates edge 4 191 5843 root40Tail26 winner.bound = root40Candidates25 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root40Step25 : scan edge (search edge 3) 4 [191] 191 root40Tail25 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail26 winner := by
  rw [root40Tail25, scan, if_neg (by decide), root40Candidates25_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail26
    (if 3 ≤ root40Candidates25.length then search edge 3 [191, 5843] root40Candidates25 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates26 : List Nat := [8237, 11471, 11933]
set_option maxRecDepth 8192 in
theorem root40Candidates26_checked : candidates edge 4 191 5981 root40Tail27 winner.bound = root40Candidates26 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root40Child26 : search edge 3 [191, 5981] root40Candidates26 6172 winner = winner := by decide
theorem root40Step26 : scan edge (search edge 3) 4 [191] 191 root40Tail26 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail27 winner := by
  rw [root40Tail26, scan, if_neg (by decide), root40Candidates26_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail27
    (if 3 ≤ root40Candidates26.length then search edge 3 [191, 5981] root40Candidates26 6172 winner else winner) = _
  rw [if_pos (by decide), root40Child26]
def root40Candidates27 : List Nat := [9461]
set_option maxRecDepth 8192 in
theorem root40Candidates27_checked : candidates edge 4 191 6287 root40Tail28 winner.bound = root40Candidates27 := by
  unfold candidates
  have row : edge 6287 = (SortedNatTree.contains FastRows.row6287) := by
    change Data.row6287.contains = (SortedNatTree.contains FastRows.row6287)
    exact FastRows.row6287_checked.symm
  rw [row]
  decide
theorem root40Step27 : scan edge (search edge 3) 4 [191] 191 root40Tail27 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail28 winner := by
  rw [root40Tail27, scan, if_neg (by decide), root40Candidates27_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail28
    (if 3 ≤ root40Candidates27.length then search edge 3 [191, 6287] root40Candidates27 6478 winner else winner) = _
  rw [if_neg (by decide)]
def root40Candidates28 : List Nat := []
set_option maxRecDepth 8192 in
theorem root40Candidates28_checked : candidates edge 4 191 6311 root40Tail29 winner.bound = root40Candidates28 := by
  unfold candidates
  have row : edge 6311 = (SortedNatTree.contains FastRows.row6311) := by
    change Data.row6311.contains = (SortedNatTree.contains FastRows.row6311)
    exact FastRows.row6311_checked.symm
  rw [row]
  decide
theorem root40Step28 : scan edge (search edge 3) 4 [191] 191 root40Tail28 winner =
    scan edge (search edge 3) 4 [191] 191 root40Tail29 winner := by
  rw [root40Tail28, scan, if_neg (by decide), root40Candidates28_checked]
  change scan edge (search edge 3) 4 [191] 191 root40Tail29
    (if 3 ≤ root40Candidates28.length then search edge 3 [191, 6311] root40Candidates28 6502 winner else winner) = _
  rw [if_neg (by decide)]
theorem root40Scan29 : scan edge (search edge 3) 4 [191] 191 root40Tail29 winner = winner := by rw [root40Tail29, scan, if_pos (by decide)]
theorem root40Scan28 : scan edge (search edge 3) 4 [191] 191 root40Tail28 winner = winner :=
  (root40Step28).trans root40Scan29
theorem root40Scan27 : scan edge (search edge 3) 4 [191] 191 root40Tail27 winner = winner :=
  (root40Step27).trans root40Scan28
theorem root40Scan26 : scan edge (search edge 3) 4 [191] 191 root40Tail26 winner = winner :=
  (root40Step26).trans root40Scan27
theorem root40Scan25 : scan edge (search edge 3) 4 [191] 191 root40Tail25 winner = winner :=
  (root40Step25).trans root40Scan26
theorem root40Scan24 : scan edge (search edge 3) 4 [191] 191 root40Tail24 winner = winner :=
  (root40Step24).trans root40Scan25
theorem root40Scan23 : scan edge (search edge 3) 4 [191] 191 root40Tail23 winner = winner :=
  (root40Step23).trans root40Scan24
theorem root40Scan22 : scan edge (search edge 3) 4 [191] 191 root40Tail22 winner = winner :=
  (root40Step22).trans root40Scan23
theorem root40Scan21 : scan edge (search edge 3) 4 [191] 191 root40Tail21 winner = winner :=
  (root40Step21).trans root40Scan22
theorem root40Scan20 : scan edge (search edge 3) 4 [191] 191 root40Tail20 winner = winner :=
  (root40Step20).trans root40Scan21
theorem root40Scan19 : scan edge (search edge 3) 4 [191] 191 root40Tail19 winner = winner :=
  (root40Step19).trans root40Scan20
theorem root40Scan18 : scan edge (search edge 3) 4 [191] 191 root40Tail18 winner = winner :=
  (root40Step18).trans root40Scan19
theorem root40Scan17 : scan edge (search edge 3) 4 [191] 191 root40Tail17 winner = winner :=
  (root40Step17).trans root40Scan18
theorem root40Scan16 : scan edge (search edge 3) 4 [191] 191 root40Tail16 winner = winner :=
  (root40Step16).trans root40Scan17
theorem root40Scan15 : scan edge (search edge 3) 4 [191] 191 root40Tail15 winner = winner :=
  (root40Step15).trans root40Scan16
theorem root40Scan14 : scan edge (search edge 3) 4 [191] 191 root40Tail14 winner = winner :=
  (root40Step14).trans root40Scan15
theorem root40Scan13 : scan edge (search edge 3) 4 [191] 191 root40Tail13 winner = winner :=
  (root40Step13).trans root40Scan14
theorem root40Scan12 : scan edge (search edge 3) 4 [191] 191 root40Tail12 winner = winner :=
  (root40Step12).trans root40Scan13
theorem root40Scan11 : scan edge (search edge 3) 4 [191] 191 root40Tail11 winner = winner :=
  (root40Step11).trans root40Scan12
theorem root40Scan10 : scan edge (search edge 3) 4 [191] 191 root40Tail10 winner = winner :=
  (root40Step10).trans root40Scan11
theorem root40Scan9 : scan edge (search edge 3) 4 [191] 191 root40Tail9 winner = winner :=
  (root40Step9).trans root40Scan10
theorem root40Scan8 : scan edge (search edge 3) 4 [191] 191 root40Tail8 winner = winner :=
  (root40Step8).trans root40Scan9
theorem root40Scan7 : scan edge (search edge 3) 4 [191] 191 root40Tail7 winner = winner :=
  (root40Step7).trans root40Scan8
theorem root40Scan6 : scan edge (search edge 3) 4 [191] 191 root40Tail6 winner = winner :=
  (root40Step6).trans root40Scan7
theorem root40Scan5 : scan edge (search edge 3) 4 [191] 191 root40Tail5 winner = winner :=
  (root40Step5).trans root40Scan6
theorem root40Scan4 : scan edge (search edge 3) 4 [191] 191 root40Tail4 winner = winner :=
  (root40Step4).trans root40Scan5
theorem root40Scan3 : scan edge (search edge 3) 4 [191] 191 root40Tail3 winner = winner :=
  (root40Step3).trans root40Scan4
theorem root40Scan2 : scan edge (search edge 3) 4 [191] 191 root40Tail2 winner = winner :=
  (root40Step2).trans root40Scan3
theorem root40Scan1 : scan edge (search edge 3) 4 [191] 191 root40Tail1 winner = winner :=
  (root40Step1).trans root40Scan2
theorem root40Scan0 : scan edge (search edge 3) 4 [191] 191 root40Tail0 winner = winner :=
  (root40Step0).trans root40Scan1
set_option maxRecDepth 4096 in
theorem root40Branch : search edge 4 [191] root40Candidates 191 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [191] 191 root40Tail0 winner = winner
  exact root40Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
