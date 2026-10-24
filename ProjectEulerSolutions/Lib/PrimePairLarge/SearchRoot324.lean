import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root324Candidates : List Nat := Data.row2179
set_option maxRecDepth 8192 in
theorem root324Candidates_checked : candidates edge 5 0 2179 tail325 winner.bound = root324Candidates := by
  apply candidates_eq_row edge 5 0 2179 tail325 root324Candidates winner.bound tail325_ordered
  · decide
  · decide
  · funext q
    rfl
def root324Tail41 : List Nat := []
def root324Tail40 : List Nat := 23053 :: root324Tail41
def root324Tail39 : List Nat := 22051 :: root324Tail40
def root324Tail38 : List Nat := 21817 :: root324Tail39
def root324Tail37 : List Nat := 21673 :: root324Tail38
def root324Tail36 : List Nat := 21061 :: root324Tail37
def root324Tail35 : List Nat := 20719 :: root324Tail36
def root324Tail34 : List Nat := 19927 :: root324Tail35
def root324Tail33 : List Nat := 19423 :: root324Tail34
def root324Tail32 : List Nat := 18517 :: root324Tail33
def root324Tail31 : List Nat := 18289 :: root324Tail32
def root324Tail30 : List Nat := 17683 :: root324Tail31
def root324Tail29 : List Nat := 17053 :: root324Tail30
def root324Tail28 : List Nat := 16921 :: root324Tail29
def root324Tail27 : List Nat := 16903 :: root324Tail28
def root324Tail26 : List Nat := 16843 :: root324Tail27
def root324Tail25 : List Nat := 15559 :: root324Tail26
def root324Tail24 : List Nat := 14929 :: root324Tail25
def root324Tail23 : List Nat := 14503 :: root324Tail24
def root324Tail22 : List Nat := 13789 :: root324Tail23
def root324Tail21 : List Nat := 11731 :: root324Tail22
def root324Tail20 : List Nat := 11503 :: root324Tail21
def root324Tail19 : List Nat := 10453 :: root324Tail20
def root324Tail18 : List Nat := 9871 :: root324Tail19
def root324Tail17 : List Nat := 9781 :: root324Tail18
def root324Tail16 : List Nat := 9241 :: root324Tail17
def root324Tail15 : List Nat := 9109 :: root324Tail16
def root324Tail14 : List Nat := 8467 :: root324Tail15
def root324Tail13 : List Nat := 8419 :: root324Tail14
def root324Tail12 : List Nat := 6529 :: root324Tail13
def root324Tail11 : List Nat := 6343 :: root324Tail12
def root324Tail10 : List Nat := 5857 :: root324Tail11
def root324Tail9 : List Nat := 5647 :: root324Tail10
def root324Tail8 : List Nat := 5227 :: root324Tail9
def root324Tail7 : List Nat := 5119 :: root324Tail8
def root324Tail6 : List Nat := 4261 :: root324Tail7
def root324Tail5 : List Nat := 3631 :: root324Tail6
def root324Tail4 : List Nat := 3469 :: root324Tail5
def root324Tail3 : List Nat := 3253 :: root324Tail4
def root324Tail2 : List Nat := 3169 :: root324Tail3
def root324Tail1 : List Nat := 2917 :: root324Tail2
def root324Tail0 : List Nat := 2503 :: root324Tail1
def root324Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root324Candidates0_checked : candidates edge 4 2179 2503 root324Tail1 winner.bound = root324Candidates0 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
theorem root324Step0 : scan edge (search edge 3) 4 [2179] 2179 root324Tail0 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail1 winner := by
  rw [root324Tail0, scan, if_neg (by decide), root324Candidates0_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail1
    (if 3 ≤ root324Candidates0.length then search edge 3 [2179, 2503] root324Candidates0 4682 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates1 : List Nat := [3169, 3631, 5227]
set_option maxRecDepth 8192 in
theorem root324Candidates1_checked : candidates edge 4 2179 2917 root324Tail2 winner.bound = root324Candidates1 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root324Child1 : search edge 3 [2179, 2917] root324Candidates1 5096 winner = winner := by decide
theorem root324Step1 : scan edge (search edge 3) 4 [2179] 2179 root324Tail1 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail2 winner := by
  rw [root324Tail1, scan, if_neg (by decide), root324Candidates1_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail2
    (if 3 ≤ root324Candidates1.length then search edge 3 [2179, 2917] root324Candidates1 5096 winner else winner) = _
  rw [if_pos (by decide), root324Child1]
def root324Candidates2 : List Nat := [5119, 8419, 14503, 15559, 17683]
set_option maxRecDepth 8192 in
theorem root324Candidates2_checked : candidates edge 4 2179 3169 root324Tail3 winner.bound = root324Candidates2 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root324Child2 : search edge 3 [2179, 3169] root324Candidates2 5348 winner = winner := by decide
theorem root324Step2 : scan edge (search edge 3) 4 [2179] 2179 root324Tail2 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail3 winner := by
  rw [root324Tail2, scan, if_neg (by decide), root324Candidates2_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail3
    (if 3 ≤ root324Candidates2.length then search edge 3 [2179, 3169] root324Candidates2 5348 winner else winner) = _
  rw [if_pos (by decide), root324Child2]
def root324Candidates3 : List Nat := [14503, 18289]
set_option maxRecDepth 8192 in
theorem root324Candidates3_checked : candidates edge 4 2179 3253 root324Tail4 winner.bound = root324Candidates3 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root324Step3 : scan edge (search edge 3) 4 [2179] 2179 root324Tail3 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail4 winner := by
  rw [root324Tail3, scan, if_neg (by decide), root324Candidates3_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail4
    (if 3 ≤ root324Candidates3.length then search edge 3 [2179, 3253] root324Candidates3 5432 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates4 : List Nat := [10453]
set_option maxRecDepth 8192 in
theorem root324Candidates4_checked : candidates edge 4 2179 3469 root324Tail5 winner.bound = root324Candidates4 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root324Step4 : scan edge (search edge 3) 4 [2179] 2179 root324Tail4 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail5 winner := by
  rw [root324Tail4, scan, if_neg (by decide), root324Candidates4_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail5
    (if 3 ≤ root324Candidates4.length then search edge 3 [2179, 3469] root324Candidates4 5648 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates5 : List Nat := [5119, 19927]
set_option maxRecDepth 8192 in
theorem root324Candidates5_checked : candidates edge 4 2179 3631 root324Tail6 winner.bound = root324Candidates5 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root324Step5 : scan edge (search edge 3) 4 [2179] 2179 root324Tail5 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail6 winner := by
  rw [root324Tail5, scan, if_neg (by decide), root324Candidates5_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail6
    (if 3 ≤ root324Candidates5.length then search edge 3 [2179, 3631] root324Candidates5 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates6 : List Nat := [16843]
set_option maxRecDepth 8192 in
theorem root324Candidates6_checked : candidates edge 4 2179 4261 root324Tail7 winner.bound = root324Candidates6 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root324Step6 : scan edge (search edge 3) 4 [2179] 2179 root324Tail6 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail7 winner := by
  rw [root324Tail6, scan, if_neg (by decide), root324Candidates6_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail7
    (if 3 ≤ root324Candidates6.length then search edge 3 [2179, 4261] root324Candidates6 6440 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates7 : List Nat := [6529, 9781, 11731]
set_option maxRecDepth 8192 in
theorem root324Candidates7_checked : candidates edge 4 2179 5119 root324Tail8 winner.bound = root324Candidates7 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root324Child7 : search edge 3 [2179, 5119] root324Candidates7 7298 winner = winner := by decide
theorem root324Step7 : scan edge (search edge 3) 4 [2179] 2179 root324Tail7 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail8 winner := by
  rw [root324Tail7, scan, if_neg (by decide), root324Candidates7_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail8
    (if 3 ≤ root324Candidates7.length then search edge 3 [2179, 5119] root324Candidates7 7298 winner else winner) = _
  rw [if_pos (by decide), root324Child7]
def root324Candidates8 : List Nat := [16921]
set_option maxRecDepth 8192 in
theorem root324Candidates8_checked : candidates edge 4 2179 5227 root324Tail9 winner.bound = root324Candidates8 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root324Step8 : scan edge (search edge 3) 4 [2179] 2179 root324Tail8 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail9 winner := by
  rw [root324Tail8, scan, if_neg (by decide), root324Candidates8_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail9
    (if 3 ≤ root324Candidates8.length then search edge 3 [2179, 5227] root324Candidates8 7406 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root324Candidates9_checked : candidates edge 4 2179 5647 root324Tail10 winner.bound = root324Candidates9 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root324Step9 : scan edge (search edge 3) 4 [2179] 2179 root324Tail9 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail10 winner := by
  rw [root324Tail9, scan, if_neg (by decide), root324Candidates9_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail10
    (if 3 ≤ root324Candidates9.length then search edge 3 [2179, 5647] root324Candidates9 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root324Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root324Candidates10_checked : candidates edge 4 2179 5857 root324Tail11 winner.bound = root324Candidates10 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root324Step10 : scan edge (search edge 3) 4 [2179] 2179 root324Tail10 winner =
    scan edge (search edge 3) 4 [2179] 2179 root324Tail11 winner := by
  rw [root324Tail10, scan, if_neg (by decide), root324Candidates10_checked]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail11
    (if 3 ≤ root324Candidates10.length then search edge 3 [2179, 5857] root324Candidates10 8036 winner else winner) = _
  rw [if_neg (by decide)]
theorem root324Scan11 : scan edge (search edge 3) 4 [2179] 2179 root324Tail11 winner = winner := by rw [root324Tail11, scan, if_pos (by decide)]
theorem root324Scan10 : scan edge (search edge 3) 4 [2179] 2179 root324Tail10 winner = winner :=
  (root324Step10).trans root324Scan11
theorem root324Scan9 : scan edge (search edge 3) 4 [2179] 2179 root324Tail9 winner = winner :=
  (root324Step9).trans root324Scan10
theorem root324Scan8 : scan edge (search edge 3) 4 [2179] 2179 root324Tail8 winner = winner :=
  (root324Step8).trans root324Scan9
theorem root324Scan7 : scan edge (search edge 3) 4 [2179] 2179 root324Tail7 winner = winner :=
  (root324Step7).trans root324Scan8
theorem root324Scan6 : scan edge (search edge 3) 4 [2179] 2179 root324Tail6 winner = winner :=
  (root324Step6).trans root324Scan7
theorem root324Scan5 : scan edge (search edge 3) 4 [2179] 2179 root324Tail5 winner = winner :=
  (root324Step5).trans root324Scan6
theorem root324Scan4 : scan edge (search edge 3) 4 [2179] 2179 root324Tail4 winner = winner :=
  (root324Step4).trans root324Scan5
theorem root324Scan3 : scan edge (search edge 3) 4 [2179] 2179 root324Tail3 winner = winner :=
  (root324Step3).trans root324Scan4
theorem root324Scan2 : scan edge (search edge 3) 4 [2179] 2179 root324Tail2 winner = winner :=
  (root324Step2).trans root324Scan3
theorem root324Scan1 : scan edge (search edge 3) 4 [2179] 2179 root324Tail1 winner = winner :=
  (root324Step1).trans root324Scan2
theorem root324Scan0 : scan edge (search edge 3) 4 [2179] 2179 root324Tail0 winner = winner :=
  (root324Step0).trans root324Scan1
set_option maxRecDepth 4096 in
theorem root324Branch : search edge 4 [2179] root324Candidates 2179 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2179] 2179 root324Tail0 winner = winner
  exact root324Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
