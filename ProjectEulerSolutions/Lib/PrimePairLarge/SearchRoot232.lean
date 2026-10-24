import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root232Candidates : List Nat := Data.row1483
set_option maxRecDepth 8192 in
theorem root232Candidates_checked : candidates edge 5 0 1483 tail233 winner.bound = root232Candidates := by
  apply candidates_eq_row edge 5 0 1483 tail233 root232Candidates winner.bound tail233_ordered
  · decide
  · decide
  · funext q
    rfl
def root232Tail45 : List Nat := []
def root232Tail44 : List Nat := 20443 :: root232Tail45
def root232Tail43 : List Nat := 20149 :: root232Tail44
def root232Tail42 : List Nat := 19603 :: root232Tail43
def root232Tail41 : List Nat := 19009 :: root232Tail42
def root232Tail40 : List Nat := 18169 :: root232Tail41
def root232Tail39 : List Nat := 17599 :: root232Tail40
def root232Tail38 : List Nat := 17137 :: root232Tail39
def root232Tail37 : List Nat := 16747 :: root232Tail38
def root232Tail36 : List Nat := 15373 :: root232Tail37
def root232Tail35 : List Nat := 15313 :: root232Tail36
def root232Tail34 : List Nat := 15091 :: root232Tail35
def root232Tail33 : List Nat := 12979 :: root232Tail34
def root232Tail32 : List Nat := 11311 :: root232Tail33
def root232Tail31 : List Nat := 10993 :: root232Tail32
def root232Tail30 : List Nat := 10891 :: root232Tail31
def root232Tail29 : List Nat := 10861 :: root232Tail30
def root232Tail28 : List Nat := 10501 :: root232Tail29
def root232Tail27 : List Nat := 10429 :: root232Tail28
def root232Tail26 : List Nat := 10273 :: root232Tail27
def root232Tail25 : List Nat := 9973 :: root232Tail26
def root232Tail24 : List Nat := 9631 :: root232Tail25
def root232Tail23 : List Nat := 9547 :: root232Tail24
def root232Tail22 : List Nat := 9511 :: root232Tail23
def root232Tail21 : List Nat := 9337 :: root232Tail22
def root232Tail20 : List Nat := 9127 :: root232Tail21
def root232Tail19 : List Nat := 8209 :: root232Tail20
def root232Tail18 : List Nat := 8191 :: root232Tail19
def root232Tail17 : List Nat := 7873 :: root232Tail18
def root232Tail16 : List Nat := 7789 :: root232Tail17
def root232Tail15 : List Nat := 7591 :: root232Tail16
def root232Tail14 : List Nat := 7393 :: root232Tail15
def root232Tail13 : List Nat := 7159 :: root232Tail14
def root232Tail12 : List Nat := 7027 :: root232Tail13
def root232Tail11 : List Nat := 6961 :: root232Tail12
def root232Tail10 : List Nat := 6163 :: root232Tail11
def root232Tail9 : List Nat := 5869 :: root232Tail10
def root232Tail8 : List Nat := 5683 :: root232Tail9
def root232Tail7 : List Nat := 4357 :: root232Tail8
def root232Tail6 : List Nat := 3457 :: root232Tail7
def root232Tail5 : List Nat := 2791 :: root232Tail6
def root232Tail4 : List Nat := 2593 :: root232Tail5
def root232Tail3 : List Nat := 2341 :: root232Tail4
def root232Tail2 : List Nat := 2281 :: root232Tail3
def root232Tail1 : List Nat := 2269 :: root232Tail2
def root232Tail0 : List Nat := 1867 :: root232Tail1
def root232Candidates0 : List Nat := [7873, 15091]
set_option maxRecDepth 8192 in
theorem root232Candidates0_checked : candidates edge 4 1483 1867 root232Tail1 winner.bound = root232Candidates0 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
theorem root232Step0 : scan edge (search edge 3) 4 [1483] 1483 root232Tail0 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail1 winner := by
  rw [root232Tail0, scan, if_neg (by decide), root232Candidates0_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail1
    (if 3 ≤ root232Candidates0.length then search edge 3 [1483, 1867] root232Candidates0 3350 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates1 : List Nat := [2281, 7027, 9337, 9631, 10429, 15373]
set_option maxRecDepth 8192 in
theorem root232Candidates1_checked : candidates edge 4 1483 2269 root232Tail2 winner.bound = root232Candidates1 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root232Child1 : search edge 3 [1483, 2269] root232Candidates1 3752 winner = winner := by decide
theorem root232Step1 : scan edge (search edge 3) 4 [1483] 1483 root232Tail1 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail2 winner := by
  rw [root232Tail1, scan, if_neg (by decide), root232Candidates1_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail2
    (if 3 ≤ root232Candidates1.length then search edge 3 [1483, 2269] root232Candidates1 3752 winner else winner) = _
  rw [if_pos (by decide), root232Child1]
def root232Candidates2 : List Nat := [9127, 18169]
set_option maxRecDepth 8192 in
theorem root232Candidates2_checked : candidates edge 4 1483 2281 root232Tail3 winner.bound = root232Candidates2 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root232Step2 : scan edge (search edge 3) 4 [1483] 1483 root232Tail2 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail3 winner := by
  rw [root232Tail2, scan, if_neg (by decide), root232Candidates2_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail3
    (if 3 ≤ root232Candidates2.length then search edge 3 [1483, 2281] root232Candidates2 3764 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates3 : List Nat := [9511]
set_option maxRecDepth 8192 in
theorem root232Candidates3_checked : candidates edge 4 1483 2341 root232Tail4 winner.bound = root232Candidates3 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root232Step3 : scan edge (search edge 3) 4 [1483] 1483 root232Tail3 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail4 winner := by
  rw [root232Tail3, scan, if_neg (by decide), root232Candidates3_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail4
    (if 3 ≤ root232Candidates3.length then search edge 3 [1483, 2341] root232Candidates3 3824 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates4 : List Nat := [5683, 7027, 9547]
set_option maxRecDepth 8192 in
theorem root232Candidates4_checked : candidates edge 4 1483 2593 root232Tail5 winner.bound = root232Candidates4 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root232Child4 : search edge 3 [1483, 2593] root232Candidates4 4076 winner = winner := by decide
theorem root232Step4 : scan edge (search edge 3) 4 [1483] 1483 root232Tail4 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail5 winner := by
  rw [root232Tail4, scan, if_neg (by decide), root232Candidates4_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail5
    (if 3 ≤ root232Candidates4.length then search edge 3 [1483, 2593] root232Candidates4 4076 winner else winner) = _
  rw [if_pos (by decide), root232Child4]
def root232Candidates5 : List Nat := [6961, 9511]
set_option maxRecDepth 8192 in
theorem root232Candidates5_checked : candidates edge 4 1483 2791 root232Tail6 winner.bound = root232Candidates5 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root232Step5 : scan edge (search edge 3) 4 [1483] 1483 root232Tail5 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail6 winner := by
  rw [root232Tail5, scan, if_neg (by decide), root232Candidates5_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail6
    (if 3 ≤ root232Candidates5.length then search edge 3 [1483, 2791] root232Candidates5 4274 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates6 : List Nat := [5869, 8209]
set_option maxRecDepth 8192 in
theorem root232Candidates6_checked : candidates edge 4 1483 3457 root232Tail7 winner.bound = root232Candidates6 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root232Step6 : scan edge (search edge 3) 4 [1483] 1483 root232Tail6 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail7 winner := by
  rw [root232Tail6, scan, if_neg (by decide), root232Candidates6_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail7
    (if 3 ≤ root232Candidates6.length then search edge 3 [1483, 3457] root232Candidates6 4940 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates7 : List Nat := [8209]
set_option maxRecDepth 8192 in
theorem root232Candidates7_checked : candidates edge 4 1483 4357 root232Tail8 winner.bound = root232Candidates7 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root232Step7 : scan edge (search edge 3) 4 [1483] 1483 root232Tail7 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail8 winner := by
  rw [root232Tail7, scan, if_neg (by decide), root232Candidates7_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail8
    (if 3 ≤ root232Candidates7.length then search edge 3 [1483, 4357] root232Candidates7 5840 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates8 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root232Candidates8_checked : candidates edge 4 1483 5683 root232Tail9 winner.bound = root232Candidates8 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root232Step8 : scan edge (search edge 3) 4 [1483] 1483 root232Tail8 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail9 winner := by
  rw [root232Tail8, scan, if_neg (by decide), root232Candidates8_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail9
    (if 3 ≤ root232Candidates8.length then search edge 3 [1483, 5683] root232Candidates8 7166 winner else winner) = _
  rw [if_neg (by decide)]
def root232Candidates9 : List Nat := [6163, 6961, 8209, 9547, 10993, 15313]
set_option maxRecDepth 8192 in
theorem root232Candidates9_checked : candidates edge 4 1483 5869 root232Tail10 winner.bound = root232Candidates9 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root232Child9 : search edge 3 [1483, 5869] root232Candidates9 7352 winner = winner := by decide
theorem root232Step9 : scan edge (search edge 3) 4 [1483] 1483 root232Tail9 winner =
    scan edge (search edge 3) 4 [1483] 1483 root232Tail10 winner := by
  rw [root232Tail9, scan, if_neg (by decide), root232Candidates9_checked]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail10
    (if 3 ≤ root232Candidates9.length then search edge 3 [1483, 5869] root232Candidates9 7352 winner else winner) = _
  rw [if_pos (by decide), root232Child9]
theorem root232Scan10 : scan edge (search edge 3) 4 [1483] 1483 root232Tail10 winner = winner := by rw [root232Tail10, scan, if_pos (by decide)]
theorem root232Scan9 : scan edge (search edge 3) 4 [1483] 1483 root232Tail9 winner = winner :=
  (root232Step9).trans root232Scan10
theorem root232Scan8 : scan edge (search edge 3) 4 [1483] 1483 root232Tail8 winner = winner :=
  (root232Step8).trans root232Scan9
theorem root232Scan7 : scan edge (search edge 3) 4 [1483] 1483 root232Tail7 winner = winner :=
  (root232Step7).trans root232Scan8
theorem root232Scan6 : scan edge (search edge 3) 4 [1483] 1483 root232Tail6 winner = winner :=
  (root232Step6).trans root232Scan7
theorem root232Scan5 : scan edge (search edge 3) 4 [1483] 1483 root232Tail5 winner = winner :=
  (root232Step5).trans root232Scan6
theorem root232Scan4 : scan edge (search edge 3) 4 [1483] 1483 root232Tail4 winner = winner :=
  (root232Step4).trans root232Scan5
theorem root232Scan3 : scan edge (search edge 3) 4 [1483] 1483 root232Tail3 winner = winner :=
  (root232Step3).trans root232Scan4
theorem root232Scan2 : scan edge (search edge 3) 4 [1483] 1483 root232Tail2 winner = winner :=
  (root232Step2).trans root232Scan3
theorem root232Scan1 : scan edge (search edge 3) 4 [1483] 1483 root232Tail1 winner = winner :=
  (root232Step1).trans root232Scan2
theorem root232Scan0 : scan edge (search edge 3) 4 [1483] 1483 root232Tail0 winner = winner :=
  (root232Step0).trans root232Scan1
set_option maxRecDepth 4096 in
theorem root232Branch : search edge 4 [1483] root232Candidates 1483 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1483] 1483 root232Tail0 winner = winner
  exact root232Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
