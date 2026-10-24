import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root57Candidates : List Nat := Data.row281
set_option maxRecDepth 8192 in
theorem root57Candidates_checked : candidates edge 5 0 281 tail58 winner.bound = root57Candidates := by
  apply candidates_eq_row edge 5 0 281 tail58 root57Candidates winner.bound tail58_ordered
  · decide
  · decide
  · funext q
    rfl
def root57Tail81 : List Nat := []
def root57Tail80 : List Nat := 25577 :: root57Tail81
def root57Tail79 : List Nat := 24923 :: root57Tail80
def root57Tail78 : List Nat := 24917 :: root57Tail79
def root57Tail77 : List Nat := 24527 :: root57Tail78
def root57Tail76 : List Nat := 24407 :: root57Tail77
def root57Tail75 : List Nat := 23747 :: root57Tail76
def root57Tail74 : List Nat := 23609 :: root57Tail75
def root57Tail73 : List Nat := 23561 :: root57Tail74
def root57Tail72 : List Nat := 23357 :: root57Tail73
def root57Tail71 : List Nat := 22691 :: root57Tail72
def root57Tail70 : List Nat := 22637 :: root57Tail71
def root57Tail69 : List Nat := 22271 :: root57Tail70
def root57Tail68 : List Nat := 22013 :: root57Tail69
def root57Tail67 : List Nat := 21803 :: root57Tail68
def root57Tail66 : List Nat := 21647 :: root57Tail67
def root57Tail65 : List Nat := 21143 :: root57Tail66
def root57Tail64 : List Nat := 20759 :: root57Tail65
def root57Tail63 : List Nat := 19913 :: root57Tail64
def root57Tail62 : List Nat := 19709 :: root57Tail63
def root57Tail61 : List Nat := 19541 :: root57Tail62
def root57Tail60 : List Nat := 19319 :: root57Tail61
def root57Tail59 : List Nat := 18773 :: root57Tail60
def root57Tail58 : List Nat := 18287 :: root57Tail59
def root57Tail57 : List Nat := 17489 :: root57Tail58
def root57Tail56 : List Nat := 17387 :: root57Tail57
def root57Tail55 : List Nat := 17351 :: root57Tail56
def root57Tail54 : List Nat := 17027 :: root57Tail55
def root57Tail53 : List Nat := 16229 :: root57Tail54
def root57Tail52 : List Nat := 16061 :: root57Tail53
def root57Tail51 : List Nat := 15809 :: root57Tail52
def root57Tail50 : List Nat := 15749 :: root57Tail51
def root57Tail49 : List Nat := 14969 :: root57Tail50
def root57Tail48 : List Nat := 14423 :: root57Tail49
def root57Tail47 : List Nat := 13523 :: root57Tail48
def root57Tail46 : List Nat := 12809 :: root57Tail47
def root57Tail45 : List Nat := 12659 :: root57Tail46
def root57Tail44 : List Nat := 12653 :: root57Tail45
def root57Tail43 : List Nat := 12569 :: root57Tail44
def root57Tail42 : List Nat := 12473 :: root57Tail43
def root57Tail41 : List Nat := 12011 :: root57Tail42
def root57Tail40 : List Nat := 11399 :: root57Tail41
def root57Tail39 : List Nat := 11093 :: root57Tail40
def root57Tail38 : List Nat := 10631 :: root57Tail39
def root57Tail37 : List Nat := 10259 :: root57Tail38
def root57Tail36 : List Nat := 9749 :: root57Tail37
def root57Tail35 : List Nat := 9491 :: root57Tail36
def root57Tail34 : List Nat := 9281 :: root57Tail35
def root57Tail33 : List Nat := 8609 :: root57Tail34
def root57Tail32 : List Nat := 8369 :: root57Tail33
def root57Tail31 : List Nat := 8111 :: root57Tail32
def root57Tail30 : List Nat := 7487 :: root57Tail31
def root57Tail29 : List Nat := 7481 :: root57Tail30
def root57Tail28 : List Nat := 7229 :: root57Tail29
def root57Tail27 : List Nat := 6899 :: root57Tail28
def root57Tail26 : List Nat := 6263 :: root57Tail27
def root57Tail25 : List Nat := 6029 :: root57Tail26
def root57Tail24 : List Nat := 5843 :: root57Tail25
def root57Tail23 : List Nat := 5399 :: root57Tail24
def root57Tail22 : List Nat := 5009 :: root57Tail23
def root57Tail21 : List Nat := 4787 :: root57Tail22
def root57Tail20 : List Nat := 3917 :: root57Tail21
def root57Tail19 : List Nat := 3677 :: root57Tail20
def root57Tail18 : List Nat := 3329 :: root57Tail19
def root57Tail17 : List Nat := 3191 :: root57Tail18
def root57Tail16 : List Nat := 2819 :: root57Tail17
def root57Tail15 : List Nat := 2687 :: root57Tail16
def root57Tail14 : List Nat := 2267 :: root57Tail15
def root57Tail13 : List Nat := 2207 :: root57Tail14
def root57Tail12 : List Nat := 1973 :: root57Tail13
def root57Tail11 : List Nat := 1619 :: root57Tail12
def root57Tail10 : List Nat := 1559 :: root57Tail11
def root57Tail9 : List Nat := 1427 :: root57Tail10
def root57Tail8 : List Nat := 1181 :: root57Tail9
def root57Tail7 : List Nat := 971 :: root57Tail8
def root57Tail6 : List Nat := 857 :: root57Tail7
def root57Tail5 : List Nat := 797 :: root57Tail6
def root57Tail4 : List Nat := 719 :: root57Tail5
def root57Tail3 : List Nat := 653 :: root57Tail4
def root57Tail2 : List Nat := 557 :: root57Tail3
def root57Tail1 : List Nat := 509 :: root57Tail2
def root57Tail0 : List Nat := 419 :: root57Tail1
def root57Candidates0 : List Nat := [8111, 17027, 22691]
set_option maxRecDepth 8192 in
theorem root57Candidates0_checked : candidates edge 4 281 419 root57Tail1 winner.bound = root57Candidates0 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child0 : search edge 3 [281, 419] root57Candidates0 700 winner = winner := by decide
theorem root57Step0 : scan edge (search edge 3) 4 [281] 281 root57Tail0 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail1 winner := by
  rw [root57Tail0, scan, if_neg (by decide), root57Candidates0_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail1
    (if 3 ≤ root57Candidates0.length then search edge 3 [281, 419] root57Candidates0 700 winner else winner) = _
  rw [if_pos (by decide), root57Child0]
def root57Candidates1 : List Nat := [797, 1973, 8369, 12011, 20759]
set_option maxRecDepth 8192 in
theorem root57Candidates1_checked : candidates edge 4 281 509 root57Tail2 winner.bound = root57Candidates1 := by
  unfold candidates
  have row : edge 509 = (SortedNatTree.contains FastRows.row509) := by
    change Data.row509.contains = (SortedNatTree.contains FastRows.row509)
    exact FastRows.row509_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child1 : search edge 3 [281, 509] root57Candidates1 790 winner = winner := by decide
theorem root57Step1 : scan edge (search edge 3) 4 [281] 281 root57Tail1 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail2 winner := by
  rw [root57Tail1, scan, if_neg (by decide), root57Candidates1_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail2
    (if 3 ≤ root57Candidates1.length then search edge 3 [281, 509] root57Candidates1 790 winner else winner) = _
  rw [if_pos (by decide), root57Child1]
def root57Candidates2 : List Nat := [2687, 12659, 16229]
set_option maxRecDepth 8192 in
theorem root57Candidates2_checked : candidates edge 4 281 557 root57Tail3 winner.bound = root57Candidates2 := by
  unfold candidates
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child2 : search edge 3 [281, 557] root57Candidates2 838 winner = winner := by decide
theorem root57Step2 : scan edge (search edge 3) 4 [281] 281 root57Tail2 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail3 winner := by
  rw [root57Tail2, scan, if_neg (by decide), root57Candidates2_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail3
    (if 3 ≤ root57Candidates2.length then search edge 3 [281, 557] root57Candidates2 838 winner else winner) = _
  rw [if_pos (by decide), root57Child2]
def root57Candidates3 : List Nat := [17489, 21647, 24923]
set_option maxRecDepth 8192 in
theorem root57Candidates3_checked : candidates edge 4 281 653 root57Tail4 winner.bound = root57Candidates3 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child3 : search edge 3 [281, 653] root57Candidates3 934 winner = winner := by decide
theorem root57Step3 : scan edge (search edge 3) 4 [281] 281 root57Tail3 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail4 winner := by
  rw [root57Tail3, scan, if_neg (by decide), root57Candidates3_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail4
    (if 3 ≤ root57Candidates3.length then search edge 3 [281, 653] root57Candidates3 934 winner else winner) = _
  rw [if_pos (by decide), root57Child3]
def root57Candidates4 : List Nat := [12569, 19709, 20759]
set_option maxRecDepth 8192 in
theorem root57Candidates4_checked : candidates edge 4 281 719 root57Tail5 winner.bound = root57Candidates4 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child4 : search edge 3 [281, 719] root57Candidates4 1000 winner = winner := by decide
theorem root57Step4 : scan edge (search edge 3) 4 [281] 281 root57Tail4 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail5 winner := by
  rw [root57Tail4, scan, if_neg (by decide), root57Candidates4_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail5
    (if 3 ≤ root57Candidates4.length then search edge 3 [281, 719] root57Candidates4 1000 winner else winner) = _
  rw [if_pos (by decide), root57Child4]
def root57Candidates5 : List Nat := [20759]
set_option maxRecDepth 8192 in
theorem root57Candidates5_checked : candidates edge 4 281 797 root57Tail6 winner.bound = root57Candidates5 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
theorem root57Step5 : scan edge (search edge 3) 4 [281] 281 root57Tail5 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail6 winner := by
  rw [root57Tail5, scan, if_neg (by decide), root57Candidates5_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail6
    (if 3 ≤ root57Candidates5.length then search edge 3 [281, 797] root57Candidates5 1078 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates6 : List Nat := [1973, 9491, 12569]
set_option maxRecDepth 8192 in
theorem root57Candidates6_checked : candidates edge 4 281 857 root57Tail7 winner.bound = root57Candidates6 := by
  unfold candidates
  have row : edge 857 = (SortedNatTree.contains FastRows.row857) := by
    change Data.row857.contains = (SortedNatTree.contains FastRows.row857)
    exact FastRows.row857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child6 : search edge 3 [281, 857] root57Candidates6 1138 winner = winner := by decide
theorem root57Step6 : scan edge (search edge 3) 4 [281] 281 root57Tail6 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail7 winner := by
  rw [root57Tail6, scan, if_neg (by decide), root57Candidates6_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail7
    (if 3 ≤ root57Candidates6.length then search edge 3 [281, 857] root57Candidates6 1138 winner else winner) = _
  rw [if_pos (by decide), root57Child6]
def root57Candidates7 : List Nat := [5009]
set_option maxRecDepth 8192 in
theorem root57Candidates7_checked : candidates edge 4 281 971 root57Tail8 winner.bound = root57Candidates7 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root57Step7 : scan edge (search edge 3) 4 [281] 281 root57Tail7 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail8 winner := by
  rw [root57Tail7, scan, if_neg (by decide), root57Candidates7_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail8
    (if 3 ≤ root57Candidates7.length then search edge 3 [281, 971] root57Candidates7 1252 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates8 : List Nat := [1427, 2687, 24407]
set_option maxRecDepth 8192 in
theorem root57Candidates8_checked : candidates edge 4 281 1181 root57Tail9 winner.bound = root57Candidates8 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child8 : search edge 3 [281, 1181] root57Candidates8 1462 winner = winner := by decide
theorem root57Step8 : scan edge (search edge 3) 4 [281] 281 root57Tail8 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail9 winner := by
  rw [root57Tail8, scan, if_neg (by decide), root57Candidates8_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail9
    (if 3 ≤ root57Candidates8.length then search edge 3 [281, 1181] root57Candidates8 1462 winner else winner) = _
  rw [if_pos (by decide), root57Child8]
def root57Candidates9 : List Nat := [5843, 8111, 12659]
set_option maxRecDepth 8192 in
theorem root57Candidates9_checked : candidates edge 4 281 1427 root57Tail10 winner.bound = root57Candidates9 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child9 : search edge 3 [281, 1427] root57Candidates9 1708 winner = winner := by decide
theorem root57Step9 : scan edge (search edge 3) 4 [281] 281 root57Tail9 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail10 winner := by
  rw [root57Tail9, scan, if_neg (by decide), root57Candidates9_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail10
    (if 3 ≤ root57Candidates9.length then search edge 3 [281, 1427] root57Candidates9 1708 winner else winner) = _
  rw [if_pos (by decide), root57Child9]
def root57Candidates10 : List Nat := [3191, 9491, 12473]
set_option maxRecDepth 8192 in
theorem root57Candidates10_checked : candidates edge 4 281 1559 root57Tail11 winner.bound = root57Candidates10 := by
  unfold candidates
  have row : edge 1559 = (SortedNatTree.contains FastRows.row1559) := by
    change Data.row1559.contains = (SortedNatTree.contains FastRows.row1559)
    exact FastRows.row1559_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child10 : search edge 3 [281, 1559] root57Candidates10 1840 winner = winner := by decide
theorem root57Step10 : scan edge (search edge 3) 4 [281] 281 root57Tail10 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail11 winner := by
  rw [root57Tail10, scan, if_neg (by decide), root57Candidates10_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail11
    (if 3 ≤ root57Candidates10.length then search edge 3 [281, 1559] root57Candidates10 1840 winner else winner) = _
  rw [if_pos (by decide), root57Child10]
def root57Candidates11 : List Nat := [3677, 9749]
set_option maxRecDepth 8192 in
theorem root57Candidates11_checked : candidates edge 4 281 1619 root57Tail12 winner.bound = root57Candidates11 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
theorem root57Step11 : scan edge (search edge 3) 4 [281] 281 root57Tail11 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail12 winner := by
  rw [root57Tail11, scan, if_neg (by decide), root57Candidates11_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail12
    (if 3 ≤ root57Candidates11.length then search edge 3 [281, 1619] root57Candidates11 1900 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates12 : List Nat := [12809, 20759, 21647]
set_option maxRecDepth 8192 in
theorem root57Candidates12_checked : candidates edge 4 281 1973 root57Tail13 winner.bound = root57Candidates12 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child12 : search edge 3 [281, 1973] root57Candidates12 2254 winner = winner := by decide
theorem root57Step12 : scan edge (search edge 3) 4 [281] 281 root57Tail12 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail13 winner := by
  rw [root57Tail12, scan, if_neg (by decide), root57Candidates12_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail13
    (if 3 ≤ root57Candidates12.length then search edge 3 [281, 1973] root57Candidates12 2254 winner else winner) = _
  rw [if_pos (by decide), root57Child12]
def root57Candidates13 : List Nat := [2819, 22013, 22637]
set_option maxRecDepth 8192 in
theorem root57Candidates13_checked : candidates edge 4 281 2207 root57Tail14 winner.bound = root57Candidates13 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child13 : search edge 3 [281, 2207] root57Candidates13 2488 winner = winner := by decide
theorem root57Step13 : scan edge (search edge 3) 4 [281] 281 root57Tail13 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail14 winner := by
  rw [root57Tail13, scan, if_neg (by decide), root57Candidates13_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail14
    (if 3 ≤ root57Candidates13.length then search edge 3 [281, 2207] root57Candidates13 2488 winner else winner) = _
  rw [if_pos (by decide), root57Child13]
def root57Candidates14 : List Nat := [5843, 12659]
set_option maxRecDepth 8192 in
theorem root57Candidates14_checked : candidates edge 4 281 2267 root57Tail15 winner.bound = root57Candidates14 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root57Step14 : scan edge (search edge 3) 4 [281] 281 root57Tail14 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail15 winner := by
  rw [root57Tail14, scan, if_neg (by decide), root57Candidates14_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail15
    (if 3 ≤ root57Candidates14.length then search edge 3 [281, 2267] root57Candidates14 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root57Candidates15_checked : candidates edge 4 281 2687 root57Tail16 winner.bound = root57Candidates15 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root57Step15 : scan edge (search edge 3) 4 [281] 281 root57Tail15 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail16 winner := by
  rw [root57Tail15, scan, if_neg (by decide), root57Candidates15_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail16
    (if 3 ≤ root57Candidates15.length then search edge 3 [281, 2687] root57Candidates15 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates16 : List Nat := [4787]
set_option maxRecDepth 8192 in
theorem root57Candidates16_checked : candidates edge 4 281 2819 root57Tail17 winner.bound = root57Candidates16 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root57Step16 : scan edge (search edge 3) 4 [281] 281 root57Tail16 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail17 winner := by
  rw [root57Tail16, scan, if_neg (by decide), root57Candidates16_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail17
    (if 3 ≤ root57Candidates16.length then search edge 3 [281, 2819] root57Candidates16 3100 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates17 : List Nat := [8111, 14423]
set_option maxRecDepth 8192 in
theorem root57Candidates17_checked : candidates edge 4 281 3191 root57Tail18 winner.bound = root57Candidates17 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root57Step17 : scan edge (search edge 3) 4 [281] 281 root57Tail17 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail18 winner := by
  rw [root57Tail17, scan, if_neg (by decide), root57Candidates17_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail18
    (if 3 ≤ root57Candidates17.length then search edge 3 [281, 3191] root57Candidates17 3472 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root57Candidates18_checked : candidates edge 4 281 3329 root57Tail19 winner.bound = root57Candidates18 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root57Step18 : scan edge (search edge 3) 4 [281] 281 root57Tail18 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail19 winner := by
  rw [root57Tail18, scan, if_neg (by decide), root57Candidates18_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail19
    (if 3 ≤ root57Candidates18.length then search edge 3 [281, 3329] root57Candidates18 3610 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates19 : List Nat := [7229]
set_option maxRecDepth 8192 in
theorem root57Candidates19_checked : candidates edge 4 281 3677 root57Tail20 winner.bound = root57Candidates19 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root57Step19 : scan edge (search edge 3) 4 [281] 281 root57Tail19 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail20 winner := by
  rw [root57Tail19, scan, if_neg (by decide), root57Candidates19_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail20
    (if 3 ≤ root57Candidates19.length then search edge 3 [281, 3677] root57Candidates19 3958 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates20 : List Nat := [5399, 6899]
set_option maxRecDepth 8192 in
theorem root57Candidates20_checked : candidates edge 4 281 3917 root57Tail21 winner.bound = root57Candidates20 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root57Step20 : scan edge (search edge 3) 4 [281] 281 root57Tail20 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail21 winner := by
  rw [root57Tail20, scan, if_neg (by decide), root57Candidates20_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail21
    (if 3 ≤ root57Candidates20.length then search edge 3 [281, 3917] root57Candidates20 4198 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates21 : List Nat := [7229, 17387]
set_option maxRecDepth 8192 in
theorem root57Candidates21_checked : candidates edge 4 281 4787 root57Tail22 winner.bound = root57Candidates21 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root57Step21 : scan edge (search edge 3) 4 [281] 281 root57Tail21 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail22 winner := by
  rw [root57Tail21, scan, if_neg (by decide), root57Candidates21_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail22
    (if 3 ≤ root57Candidates21.length then search edge 3 [281, 4787] root57Candidates21 5068 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates22 : List Nat := [6029, 11399, 16061]
set_option maxRecDepth 8192 in
theorem root57Candidates22_checked : candidates edge 4 281 5009 root57Tail23 winner.bound = root57Candidates22 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child22 : search edge 3 [281, 5009] root57Candidates22 5290 winner = winner := by decide
theorem root57Step22 : scan edge (search edge 3) 4 [281] 281 root57Tail22 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail23 winner := by
  rw [root57Tail22, scan, if_neg (by decide), root57Candidates22_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail23
    (if 3 ≤ root57Candidates22.length then search edge 3 [281, 5009] root57Candidates22 5290 winner else winner) = _
  rw [if_pos (by decide), root57Child22]
def root57Candidates23 : List Nat := [8111, 9749, 15749]
set_option maxRecDepth 8192 in
theorem root57Candidates23_checked : candidates edge 4 281 5399 root57Tail24 winner.bound = root57Candidates23 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root57Child23 : search edge 3 [281, 5399] root57Candidates23 5680 winner = winner := by decide
theorem root57Step23 : scan edge (search edge 3) 4 [281] 281 root57Tail23 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail24 winner := by
  rw [root57Tail23, scan, if_neg (by decide), root57Candidates23_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail24
    (if 3 ≤ root57Candidates23.length then search edge 3 [281, 5399] root57Candidates23 5680 winner else winner) = _
  rw [if_pos (by decide), root57Child23]
def root57Candidates24 : List Nat := [9749, 13523]
set_option maxRecDepth 8192 in
theorem root57Candidates24_checked : candidates edge 4 281 5843 root57Tail25 winner.bound = root57Candidates24 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root57Step24 : scan edge (search edge 3) 4 [281] 281 root57Tail24 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail25 winner := by
  rw [root57Tail24, scan, if_neg (by decide), root57Candidates24_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail25
    (if 3 ≤ root57Candidates24.length then search edge 3 [281, 5843] root57Candidates24 6124 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root57Candidates25_checked : candidates edge 4 281 6029 root57Tail26 winner.bound = root57Candidates25 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root57Step25 : scan edge (search edge 3) 4 [281] 281 root57Tail25 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail26 winner := by
  rw [root57Tail25, scan, if_neg (by decide), root57Candidates25_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail26
    (if 3 ≤ root57Candidates25.length then search edge 3 [281, 6029] root57Candidates25 6310 winner else winner) = _
  rw [if_neg (by decide)]
def root57Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root57Candidates26_checked : candidates edge 4 281 6263 root57Tail27 winner.bound = root57Candidates26 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root57Step26 : scan edge (search edge 3) 4 [281] 281 root57Tail26 winner =
    scan edge (search edge 3) 4 [281] 281 root57Tail27 winner := by
  rw [root57Tail26, scan, if_neg (by decide), root57Candidates26_checked]
  change scan edge (search edge 3) 4 [281] 281 root57Tail27
    (if 3 ≤ root57Candidates26.length then search edge 3 [281, 6263] root57Candidates26 6544 winner else winner) = _
  rw [if_neg (by decide)]
theorem root57Scan27 : scan edge (search edge 3) 4 [281] 281 root57Tail27 winner = winner := by rw [root57Tail27, scan, if_pos (by decide)]
theorem root57Scan26 : scan edge (search edge 3) 4 [281] 281 root57Tail26 winner = winner :=
  (root57Step26).trans root57Scan27
theorem root57Scan25 : scan edge (search edge 3) 4 [281] 281 root57Tail25 winner = winner :=
  (root57Step25).trans root57Scan26
theorem root57Scan24 : scan edge (search edge 3) 4 [281] 281 root57Tail24 winner = winner :=
  (root57Step24).trans root57Scan25
theorem root57Scan23 : scan edge (search edge 3) 4 [281] 281 root57Tail23 winner = winner :=
  (root57Step23).trans root57Scan24
theorem root57Scan22 : scan edge (search edge 3) 4 [281] 281 root57Tail22 winner = winner :=
  (root57Step22).trans root57Scan23
theorem root57Scan21 : scan edge (search edge 3) 4 [281] 281 root57Tail21 winner = winner :=
  (root57Step21).trans root57Scan22
theorem root57Scan20 : scan edge (search edge 3) 4 [281] 281 root57Tail20 winner = winner :=
  (root57Step20).trans root57Scan21
theorem root57Scan19 : scan edge (search edge 3) 4 [281] 281 root57Tail19 winner = winner :=
  (root57Step19).trans root57Scan20
theorem root57Scan18 : scan edge (search edge 3) 4 [281] 281 root57Tail18 winner = winner :=
  (root57Step18).trans root57Scan19
theorem root57Scan17 : scan edge (search edge 3) 4 [281] 281 root57Tail17 winner = winner :=
  (root57Step17).trans root57Scan18
theorem root57Scan16 : scan edge (search edge 3) 4 [281] 281 root57Tail16 winner = winner :=
  (root57Step16).trans root57Scan17
theorem root57Scan15 : scan edge (search edge 3) 4 [281] 281 root57Tail15 winner = winner :=
  (root57Step15).trans root57Scan16
theorem root57Scan14 : scan edge (search edge 3) 4 [281] 281 root57Tail14 winner = winner :=
  (root57Step14).trans root57Scan15
theorem root57Scan13 : scan edge (search edge 3) 4 [281] 281 root57Tail13 winner = winner :=
  (root57Step13).trans root57Scan14
theorem root57Scan12 : scan edge (search edge 3) 4 [281] 281 root57Tail12 winner = winner :=
  (root57Step12).trans root57Scan13
theorem root57Scan11 : scan edge (search edge 3) 4 [281] 281 root57Tail11 winner = winner :=
  (root57Step11).trans root57Scan12
theorem root57Scan10 : scan edge (search edge 3) 4 [281] 281 root57Tail10 winner = winner :=
  (root57Step10).trans root57Scan11
theorem root57Scan9 : scan edge (search edge 3) 4 [281] 281 root57Tail9 winner = winner :=
  (root57Step9).trans root57Scan10
theorem root57Scan8 : scan edge (search edge 3) 4 [281] 281 root57Tail8 winner = winner :=
  (root57Step8).trans root57Scan9
theorem root57Scan7 : scan edge (search edge 3) 4 [281] 281 root57Tail7 winner = winner :=
  (root57Step7).trans root57Scan8
theorem root57Scan6 : scan edge (search edge 3) 4 [281] 281 root57Tail6 winner = winner :=
  (root57Step6).trans root57Scan7
theorem root57Scan5 : scan edge (search edge 3) 4 [281] 281 root57Tail5 winner = winner :=
  (root57Step5).trans root57Scan6
theorem root57Scan4 : scan edge (search edge 3) 4 [281] 281 root57Tail4 winner = winner :=
  (root57Step4).trans root57Scan5
theorem root57Scan3 : scan edge (search edge 3) 4 [281] 281 root57Tail3 winner = winner :=
  (root57Step3).trans root57Scan4
theorem root57Scan2 : scan edge (search edge 3) 4 [281] 281 root57Tail2 winner = winner :=
  (root57Step2).trans root57Scan3
theorem root57Scan1 : scan edge (search edge 3) 4 [281] 281 root57Tail1 winner = winner :=
  (root57Step1).trans root57Scan2
theorem root57Scan0 : scan edge (search edge 3) 4 [281] 281 root57Tail0 winner = winner :=
  (root57Step0).trans root57Scan1
set_option maxRecDepth 4096 in
theorem root57Branch : search edge 4 [281] root57Candidates 281 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [281] 281 root57Tail0 winner = winner
  exact root57Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
