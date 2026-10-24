import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root163Candidates : List Nat := Data.row983
set_option maxRecDepth 8192 in
theorem root163Candidates_checked : candidates edge 5 0 983 tail164 winner.bound = root163Candidates := by
  apply candidates_eq_row edge 5 0 983 tail164 root163Candidates winner.bound tail164_ordered
  · decide
  · decide
  · funext q
    rfl
def root163Tail56 : List Nat := []
def root163Tail55 : List Nat := 24767 :: root163Tail56
def root163Tail54 : List Nat := 23741 :: root163Tail55
def root163Tail53 : List Nat := 23531 :: root163Tail54
def root163Tail52 : List Nat := 23417 :: root163Tail53
def root163Tail51 : List Nat := 23297 :: root163Tail52
def root163Tail50 : List Nat := 22877 :: root163Tail51
def root163Tail49 : List Nat := 22409 :: root163Tail50
def root163Tail48 : List Nat := 21467 :: root163Tail49
def root163Tail47 : List Nat := 21089 :: root163Tail48
def root163Tail46 : List Nat := 20939 :: root163Tail47
def root163Tail45 : List Nat := 20897 :: root163Tail46
def root163Tail44 : List Nat := 20627 :: root163Tail45
def root163Tail43 : List Nat := 20441 :: root163Tail44
def root163Tail42 : List Nat := 19301 :: root163Tail43
def root163Tail41 : List Nat := 18797 :: root163Tail42
def root163Tail40 : List Nat := 18617 :: root163Tail41
def root163Tail39 : List Nat := 16979 :: root163Tail40
def root163Tail38 : List Nat := 15923 :: root163Tail39
def root163Tail37 : List Nat := 15083 :: root163Tail38
def root163Tail36 : List Nat := 14753 :: root163Tail37
def root163Tail35 : List Nat := 14747 :: root163Tail36
def root163Tail34 : List Nat := 14207 :: root163Tail35
def root163Tail33 : List Nat := 13763 :: root163Tail34
def root163Tail32 : List Nat := 13469 :: root163Tail33
def root163Tail31 : List Nat := 13043 :: root163Tail32
def root163Tail30 : List Nat := 12923 :: root163Tail31
def root163Tail29 : List Nat := 12689 :: root163Tail30
def root163Tail28 : List Nat := 12413 :: root163Tail29
def root163Tail27 : List Nat := 11789 :: root163Tail28
def root163Tail26 : List Nat := 11489 :: root163Tail27
def root163Tail25 : List Nat := 10781 :: root163Tail26
def root163Tail24 : List Nat := 10613 :: root163Tail25
def root163Tail23 : List Nat := 10529 :: root163Tail24
def root163Tail22 : List Nat := 10007 :: root163Tail23
def root163Tail21 : List Nat := 9857 :: root163Tail22
def root163Tail20 : List Nat := 9227 :: root163Tail21
def root163Tail19 : List Nat := 8831 :: root163Tail20
def root163Tail18 : List Nat := 8693 :: root163Tail19
def root163Tail17 : List Nat := 7481 :: root163Tail18
def root163Tail16 : List Nat := 7151 :: root163Tail17
def root163Tail15 : List Nat := 7109 :: root163Tail16
def root163Tail14 : List Nat := 6917 :: root163Tail15
def root163Tail13 : List Nat := 6719 :: root163Tail14
def root163Tail12 : List Nat := 6389 :: root163Tail13
def root163Tail11 : List Nat := 5171 :: root163Tail12
def root163Tail10 : List Nat := 4871 :: root163Tail11
def root163Tail9 : List Nat := 4073 :: root163Tail10
def root163Tail8 : List Nat := 3617 :: root163Tail9
def root163Tail7 : List Nat := 3347 :: root163Tail8
def root163Tail6 : List Nat := 3221 :: root163Tail7
def root163Tail5 : List Nat := 2399 :: root163Tail6
def root163Tail4 : List Nat := 2267 :: root163Tail5
def root163Tail3 : List Nat := 2129 :: root163Tail4
def root163Tail2 : List Nat := 1637 :: root163Tail3
def root163Tail1 : List Nat := 1259 :: root163Tail2
def root163Tail0 : List Nat := 1091 :: root163Tail1
def root163Candidates0 : List Nat := [2267]
set_option maxRecDepth 8192 in
theorem root163Candidates0_checked : candidates edge 4 983 1091 root163Tail1 winner.bound = root163Candidates0 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root163Step0 : scan edge (search edge 3) 4 [983] 983 root163Tail0 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail1 winner := by
  rw [root163Tail0, scan, if_neg (by decide), root163Candidates0_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail1
    (if 3 ≤ root163Candidates0.length then search edge 3 [983, 1091] root163Candidates0 2074 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates1 : List Nat := [14753]
set_option maxRecDepth 8192 in
theorem root163Candidates1_checked : candidates edge 4 983 1259 root163Tail2 winner.bound = root163Candidates1 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
theorem root163Step1 : scan edge (search edge 3) 4 [983] 983 root163Tail1 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail2 winner := by
  rw [root163Tail1, scan, if_neg (by decide), root163Candidates1_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail2
    (if 3 ≤ root163Candidates1.length then search edge 3 [983, 1259] root163Candidates1 2242 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates2 : List Nat := [4073, 22877, 23297]
set_option maxRecDepth 8192 in
theorem root163Candidates2_checked : candidates edge 4 983 1637 root163Tail3 winner.bound = root163Candidates2 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root163Child2 : search edge 3 [983, 1637] root163Candidates2 2620 winner = winner := by decide
theorem root163Step2 : scan edge (search edge 3) 4 [983] 983 root163Tail2 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail3 winner := by
  rw [root163Tail2, scan, if_neg (by decide), root163Candidates2_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail3
    (if 3 ≤ root163Candidates2.length then search edge 3 [983, 1637] root163Candidates2 2620 winner else winner) = _
  rw [if_pos (by decide), root163Child2]
def root163Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root163Candidates3_checked : candidates edge 4 983 2129 root163Tail4 winner.bound = root163Candidates3 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root163Step3 : scan edge (search edge 3) 4 [983] 983 root163Tail3 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail4 winner := by
  rw [root163Tail3, scan, if_neg (by decide), root163Candidates3_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail4
    (if 3 ≤ root163Candidates3.length then search edge 3 [983, 2129] root163Candidates3 3112 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates4 : List Nat := [6389, 6917, 9227]
set_option maxRecDepth 8192 in
theorem root163Candidates4_checked : candidates edge 4 983 2267 root163Tail5 winner.bound = root163Candidates4 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root163Child4 : search edge 3 [983, 2267] root163Candidates4 3250 winner = winner := by decide
theorem root163Step4 : scan edge (search edge 3) 4 [983] 983 root163Tail4 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail5 winner := by
  rw [root163Tail4, scan, if_neg (by decide), root163Candidates4_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail5
    (if 3 ≤ root163Candidates4.length then search edge 3 [983, 2267] root163Candidates4 3250 winner else winner) = _
  rw [if_pos (by decide), root163Child4]
def root163Candidates5 : List Nat := [20939]
set_option maxRecDepth 8192 in
theorem root163Candidates5_checked : candidates edge 4 983 2399 root163Tail6 winner.bound = root163Candidates5 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root163Step5 : scan edge (search edge 3) 4 [983] 983 root163Tail5 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail6 winner := by
  rw [root163Tail5, scan, if_neg (by decide), root163Candidates5_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail6
    (if 3 ≤ root163Candidates5.length then search edge 3 [983, 2399] root163Candidates5 3382 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root163Candidates6_checked : candidates edge 4 983 3221 root163Tail7 winner.bound = root163Candidates6 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root163Step6 : scan edge (search edge 3) 4 [983] 983 root163Tail6 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail7 winner := by
  rw [root163Tail6, scan, if_neg (by decide), root163Candidates6_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail7
    (if 3 ≤ root163Candidates6.length then search edge 3 [983, 3221] root163Candidates6 4204 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates7 : List Nat := [8693, 13469]
set_option maxRecDepth 8192 in
theorem root163Candidates7_checked : candidates edge 4 983 3347 root163Tail8 winner.bound = root163Candidates7 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root163Step7 : scan edge (search edge 3) 4 [983] 983 root163Tail7 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail8 winner := by
  rw [root163Tail7, scan, if_neg (by decide), root163Candidates7_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail8
    (if 3 ≤ root163Candidates7.length then search edge 3 [983, 3347] root163Candidates7 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates8 : List Nat := [13043]
set_option maxRecDepth 8192 in
theorem root163Candidates8_checked : candidates edge 4 983 3617 root163Tail9 winner.bound = root163Candidates8 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root163Step8 : scan edge (search edge 3) 4 [983] 983 root163Tail8 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail9 winner := by
  rw [root163Tail8, scan, if_neg (by decide), root163Candidates8_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail9
    (if 3 ≤ root163Candidates8.length then search edge 3 [983, 3617] root163Candidates8 4600 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates9 : List Nat := [6389, 13763]
set_option maxRecDepth 8192 in
theorem root163Candidates9_checked : candidates edge 4 983 4073 root163Tail10 winner.bound = root163Candidates9 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root163Step9 : scan edge (search edge 3) 4 [983] 983 root163Tail9 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail10 winner := by
  rw [root163Tail9, scan, if_neg (by decide), root163Candidates9_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail10
    (if 3 ≤ root163Candidates9.length then search edge 3 [983, 4073] root163Candidates9 5056 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates10 : List Nat := [6389, 6719]
set_option maxRecDepth 8192 in
theorem root163Candidates10_checked : candidates edge 4 983 4871 root163Tail11 winner.bound = root163Candidates10 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root163Step10 : scan edge (search edge 3) 4 [983] 983 root163Tail10 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail11 winner := by
  rw [root163Tail10, scan, if_neg (by decide), root163Candidates10_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail11
    (if 3 ≤ root163Candidates10.length then search edge 3 [983, 4871] root163Candidates10 5854 winner else winner) = _
  rw [if_neg (by decide)]
def root163Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root163Candidates11_checked : candidates edge 4 983 5171 root163Tail12 winner.bound = root163Candidates11 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root163Step11 : scan edge (search edge 3) 4 [983] 983 root163Tail11 winner =
    scan edge (search edge 3) 4 [983] 983 root163Tail12 winner := by
  rw [root163Tail11, scan, if_neg (by decide), root163Candidates11_checked]
  change scan edge (search edge 3) 4 [983] 983 root163Tail12
    (if 3 ≤ root163Candidates11.length then search edge 3 [983, 5171] root163Candidates11 6154 winner else winner) = _
  rw [if_neg (by decide)]
theorem root163Scan12 : scan edge (search edge 3) 4 [983] 983 root163Tail12 winner = winner := by rw [root163Tail12, scan, if_pos (by decide)]
theorem root163Scan11 : scan edge (search edge 3) 4 [983] 983 root163Tail11 winner = winner :=
  (root163Step11).trans root163Scan12
theorem root163Scan10 : scan edge (search edge 3) 4 [983] 983 root163Tail10 winner = winner :=
  (root163Step10).trans root163Scan11
theorem root163Scan9 : scan edge (search edge 3) 4 [983] 983 root163Tail9 winner = winner :=
  (root163Step9).trans root163Scan10
theorem root163Scan8 : scan edge (search edge 3) 4 [983] 983 root163Tail8 winner = winner :=
  (root163Step8).trans root163Scan9
theorem root163Scan7 : scan edge (search edge 3) 4 [983] 983 root163Tail7 winner = winner :=
  (root163Step7).trans root163Scan8
theorem root163Scan6 : scan edge (search edge 3) 4 [983] 983 root163Tail6 winner = winner :=
  (root163Step6).trans root163Scan7
theorem root163Scan5 : scan edge (search edge 3) 4 [983] 983 root163Tail5 winner = winner :=
  (root163Step5).trans root163Scan6
theorem root163Scan4 : scan edge (search edge 3) 4 [983] 983 root163Tail4 winner = winner :=
  (root163Step4).trans root163Scan5
theorem root163Scan3 : scan edge (search edge 3) 4 [983] 983 root163Tail3 winner = winner :=
  (root163Step3).trans root163Scan4
theorem root163Scan2 : scan edge (search edge 3) 4 [983] 983 root163Tail2 winner = winner :=
  (root163Step2).trans root163Scan3
theorem root163Scan1 : scan edge (search edge 3) 4 [983] 983 root163Tail1 winner = winner :=
  (root163Step1).trans root163Scan2
theorem root163Scan0 : scan edge (search edge 3) 4 [983] 983 root163Tail0 winner = winner :=
  (root163Step0).trans root163Scan1
set_option maxRecDepth 4096 in
theorem root163Branch : search edge 4 [983] root163Candidates 983 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [983] 983 root163Tail0 winner = winner
  exact root163Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
