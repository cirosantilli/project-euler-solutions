import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root293Candidates : List Nat := Data.row1949
set_option maxRecDepth 8192 in
theorem root293Candidates_checked : candidates edge 5 0 1949 tail294 winner.bound = root293Candidates := by
  apply candidates_eq_row edge 5 0 1949 tail294 root293Candidates winner.bound tail294_ordered
  · decide
  · decide
  · funext q
    rfl
def root293Tail42 : List Nat := []
def root293Tail41 : List Nat := 24071 :: root293Tail42
def root293Tail40 : List Nat := 23789 :: root293Tail41
def root293Tail39 : List Nat := 23741 :: root293Tail40
def root293Tail38 : List Nat := 23369 :: root293Tail39
def root293Tail37 : List Nat := 21773 :: root293Tail38
def root293Tail36 : List Nat := 21317 :: root293Tail37
def root293Tail35 : List Nat := 20261 :: root293Tail36
def root293Tail34 : List Nat := 19997 :: root293Tail35
def root293Tail33 : List Nat := 19973 :: root293Tail34
def root293Tail32 : List Nat := 19961 :: root293Tail33
def root293Tail31 : List Nat := 19427 :: root293Tail32
def root293Tail30 : List Nat := 18461 :: root293Tail31
def root293Tail29 : List Nat := 18233 :: root293Tail30
def root293Tail28 : List Nat := 17747 :: root293Tail29
def root293Tail27 : List Nat := 17159 :: root293Tail28
def root293Tail26 : List Nat := 15473 :: root293Tail27
def root293Tail25 : List Nat := 15263 :: root293Tail26
def root293Tail24 : List Nat := 14051 :: root293Tail25
def root293Tail23 : List Nat := 11939 :: root293Tail24
def root293Tail22 : List Nat := 11447 :: root293Tail23
def root293Tail21 : List Nat := 10889 :: root293Tail22
def root293Tail20 : List Nat := 10847 :: root293Tail21
def root293Tail19 : List Nat := 10589 :: root293Tail20
def root293Tail18 : List Nat := 10259 :: root293Tail19
def root293Tail17 : List Nat := 10007 :: root293Tail18
def root293Tail16 : List Nat := 9749 :: root293Tail17
def root293Tail15 : List Nat := 9041 :: root293Tail16
def root293Tail14 : List Nat := 7901 :: root293Tail15
def root293Tail13 : List Nat := 7727 :: root293Tail14
def root293Tail12 : List Nat := 6863 :: root293Tail13
def root293Tail11 : List Nat := 5669 :: root293Tail12
def root293Tail10 : List Nat := 4337 :: root293Tail11
def root293Tail9 : List Nat := 3989 :: root293Tail10
def root293Tail8 : List Nat := 3923 :: root293Tail9
def root293Tail7 : List Nat := 3911 :: root293Tail8
def root293Tail6 : List Nat := 2903 :: root293Tail7
def root293Tail5 : List Nat := 2819 :: root293Tail6
def root293Tail4 : List Nat := 2789 :: root293Tail5
def root293Tail3 : List Nat := 2663 :: root293Tail4
def root293Tail2 : List Nat := 2657 :: root293Tail3
def root293Tail1 : List Nat := 2621 :: root293Tail2
def root293Tail0 : List Nat := 2027 :: root293Tail1
def root293Candidates0 : List Nat := [10889, 15263, 15473]
set_option maxRecDepth 8192 in
theorem root293Candidates0_checked : candidates edge 4 1949 2027 root293Tail1 winner.bound = root293Candidates0 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root293Child0 : search edge 3 [1949, 2027] root293Candidates0 3976 winner = winner := by decide
theorem root293Step0 : scan edge (search edge 3) 4 [1949] 1949 root293Tail0 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail1 winner := by
  rw [root293Tail0, scan, if_neg (by decide), root293Candidates0_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail1
    (if 3 ≤ root293Candidates0.length then search edge 3 [1949, 2027] root293Candidates0 3976 winner else winner) = _
  rw [if_pos (by decide), root293Child0]
def root293Candidates1 : List Nat := [2789]
set_option maxRecDepth 8192 in
theorem root293Candidates1_checked : candidates edge 4 1949 2621 root293Tail2 winner.bound = root293Candidates1 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root293Step1 : scan edge (search edge 3) 4 [1949] 1949 root293Tail1 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail2 winner := by
  rw [root293Tail1, scan, if_neg (by decide), root293Candidates1_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail2
    (if 3 ≤ root293Candidates1.length then search edge 3 [1949, 2621] root293Candidates1 4570 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates2 : List Nat := [3923]
set_option maxRecDepth 8192 in
theorem root293Candidates2_checked : candidates edge 4 1949 2657 root293Tail3 winner.bound = root293Candidates2 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root293Step2 : scan edge (search edge 3) 4 [1949] 1949 root293Tail2 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail3 winner := by
  rw [root293Tail2, scan, if_neg (by decide), root293Candidates2_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail3
    (if 3 ≤ root293Candidates2.length then search edge 3 [1949, 2657] root293Candidates2 4606 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates3 : List Nat := [5669, 7901, 10007, 10259]
set_option maxRecDepth 8192 in
theorem root293Candidates3_checked : candidates edge 4 1949 2663 root293Tail4 winner.bound = root293Candidates3 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root293Child3 : search edge 3 [1949, 2663] root293Candidates3 4612 winner = winner := by decide
theorem root293Step3 : scan edge (search edge 3) 4 [1949] 1949 root293Tail3 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail4 winner := by
  rw [root293Tail3, scan, if_neg (by decide), root293Candidates3_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail4
    (if 3 ≤ root293Candidates3.length then search edge 3 [1949, 2663] root293Candidates3 4612 winner else winner) = _
  rw [if_pos (by decide), root293Child3]
def root293Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root293Candidates4_checked : candidates edge 4 1949 2789 root293Tail5 winner.bound = root293Candidates4 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root293Step4 : scan edge (search edge 3) 4 [1949] 1949 root293Tail4 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail5 winner := by
  rw [root293Tail4, scan, if_neg (by decide), root293Candidates4_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail5
    (if 3 ≤ root293Candidates4.length then search edge 3 [1949, 2789] root293Candidates4 4738 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root293Candidates5_checked : candidates edge 4 1949 2819 root293Tail6 winner.bound = root293Candidates5 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root293Step5 : scan edge (search edge 3) 4 [1949] 1949 root293Tail5 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail6 winner := by
  rw [root293Tail5, scan, if_neg (by decide), root293Candidates5_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail6
    (if 3 ≤ root293Candidates5.length then search edge 3 [1949, 2819] root293Candidates5 4768 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates6 : List Nat := [14051, 17747]
set_option maxRecDepth 8192 in
theorem root293Candidates6_checked : candidates edge 4 1949 2903 root293Tail7 winner.bound = root293Candidates6 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root293Step6 : scan edge (search edge 3) 4 [1949] 1949 root293Tail6 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail7 winner := by
  rw [root293Tail6, scan, if_neg (by decide), root293Candidates6_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail7
    (if 3 ≤ root293Candidates6.length then search edge 3 [1949, 2903] root293Candidates6 4852 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates7 : List Nat := [18461]
set_option maxRecDepth 8192 in
theorem root293Candidates7_checked : candidates edge 4 1949 3911 root293Tail8 winner.bound = root293Candidates7 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root293Step7 : scan edge (search edge 3) 4 [1949] 1949 root293Tail7 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail8 winner := by
  rw [root293Tail7, scan, if_neg (by decide), root293Candidates7_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail8
    (if 3 ≤ root293Candidates7.length then search edge 3 [1949, 3911] root293Candidates7 5860 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates8 : List Nat := [3989, 9041, 18233, 19427, 19997]
set_option maxRecDepth 8192 in
theorem root293Candidates8_checked : candidates edge 4 1949 3923 root293Tail9 winner.bound = root293Candidates8 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root293Child8 : search edge 3 [1949, 3923] root293Candidates8 5872 winner = winner := by decide
theorem root293Step8 : scan edge (search edge 3) 4 [1949] 1949 root293Tail8 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail9 winner := by
  rw [root293Tail8, scan, if_neg (by decide), root293Candidates8_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail9
    (if 3 ≤ root293Candidates8.length then search edge 3 [1949, 3923] root293Candidates8 5872 winner else winner) = _
  rw [if_pos (by decide), root293Child8]
def root293Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root293Candidates9_checked : candidates edge 4 1949 3989 root293Tail10 winner.bound = root293Candidates9 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root293Step9 : scan edge (search edge 3) 4 [1949] 1949 root293Tail9 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail10 winner := by
  rw [root293Tail9, scan, if_neg (by decide), root293Candidates9_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail10
    (if 3 ≤ root293Candidates9.length then search edge 3 [1949, 3989] root293Candidates9 5938 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates10 : List Nat := [15473]
set_option maxRecDepth 8192 in
theorem root293Candidates10_checked : candidates edge 4 1949 4337 root293Tail11 winner.bound = root293Candidates10 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root293Step10 : scan edge (search edge 3) 4 [1949] 1949 root293Tail10 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail11 winner := by
  rw [root293Tail10, scan, if_neg (by decide), root293Candidates10_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail11
    (if 3 ≤ root293Candidates10.length then search edge 3 [1949, 4337] root293Candidates10 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root293Candidates11 : List Nat := [15263]
set_option maxRecDepth 8192 in
theorem root293Candidates11_checked : candidates edge 4 1949 5669 root293Tail12 winner.bound = root293Candidates11 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root293Step11 : scan edge (search edge 3) 4 [1949] 1949 root293Tail11 winner =
    scan edge (search edge 3) 4 [1949] 1949 root293Tail12 winner := by
  rw [root293Tail11, scan, if_neg (by decide), root293Candidates11_checked]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail12
    (if 3 ≤ root293Candidates11.length then search edge 3 [1949, 5669] root293Candidates11 7618 winner else winner) = _
  rw [if_neg (by decide)]
theorem root293Scan12 : scan edge (search edge 3) 4 [1949] 1949 root293Tail12 winner = winner := by rw [root293Tail12, scan, if_pos (by decide)]
theorem root293Scan11 : scan edge (search edge 3) 4 [1949] 1949 root293Tail11 winner = winner :=
  (root293Step11).trans root293Scan12
theorem root293Scan10 : scan edge (search edge 3) 4 [1949] 1949 root293Tail10 winner = winner :=
  (root293Step10).trans root293Scan11
theorem root293Scan9 : scan edge (search edge 3) 4 [1949] 1949 root293Tail9 winner = winner :=
  (root293Step9).trans root293Scan10
theorem root293Scan8 : scan edge (search edge 3) 4 [1949] 1949 root293Tail8 winner = winner :=
  (root293Step8).trans root293Scan9
theorem root293Scan7 : scan edge (search edge 3) 4 [1949] 1949 root293Tail7 winner = winner :=
  (root293Step7).trans root293Scan8
theorem root293Scan6 : scan edge (search edge 3) 4 [1949] 1949 root293Tail6 winner = winner :=
  (root293Step6).trans root293Scan7
theorem root293Scan5 : scan edge (search edge 3) 4 [1949] 1949 root293Tail5 winner = winner :=
  (root293Step5).trans root293Scan6
theorem root293Scan4 : scan edge (search edge 3) 4 [1949] 1949 root293Tail4 winner = winner :=
  (root293Step4).trans root293Scan5
theorem root293Scan3 : scan edge (search edge 3) 4 [1949] 1949 root293Tail3 winner = winner :=
  (root293Step3).trans root293Scan4
theorem root293Scan2 : scan edge (search edge 3) 4 [1949] 1949 root293Tail2 winner = winner :=
  (root293Step2).trans root293Scan3
theorem root293Scan1 : scan edge (search edge 3) 4 [1949] 1949 root293Tail1 winner = winner :=
  (root293Step1).trans root293Scan2
theorem root293Scan0 : scan edge (search edge 3) 4 [1949] 1949 root293Tail0 winner = winner :=
  (root293Step0).trans root293Scan1
set_option maxRecDepth 4096 in
theorem root293Branch : search edge 4 [1949] root293Candidates 1949 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1949] 1949 root293Tail0 winner = winner
  exact root293Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
