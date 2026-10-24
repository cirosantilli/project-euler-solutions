import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root330Candidates : List Nat := Data.row2239
set_option maxRecDepth 8192 in
theorem root330Candidates_checked : candidates edge 5 0 2239 tail331 winner.bound = root330Candidates := by
  apply candidates_eq_row edge 5 0 2239 tail331 root330Candidates winner.bound tail331_ordered
  · decide
  · decide
  · funext q
    rfl
def root330Tail50 : List Nat := []
def root330Tail49 : List Nat := 23071 :: root330Tail50
def root330Tail48 : List Nat := 23041 :: root330Tail49
def root330Tail47 : List Nat := 21937 :: root330Tail48
def root330Tail46 : List Nat := 21589 :: root330Tail47
def root330Tail45 : List Nat := 21487 :: root330Tail46
def root330Tail44 : List Nat := 21391 :: root330Tail45
def root330Tail43 : List Nat := 20563 :: root330Tail44
def root330Tail42 : List Nat := 19477 :: root330Tail43
def root330Tail41 : List Nat := 19249 :: root330Tail42
def root330Tail40 : List Nat := 18523 :: root330Tail41
def root330Tail39 : List Nat := 18379 :: root330Tail40
def root330Tail38 : List Nat := 17959 :: root330Tail39
def root330Tail37 : List Nat := 17713 :: root330Tail38
def root330Tail36 : List Nat := 17389 :: root330Tail37
def root330Tail35 : List Nat := 17191 :: root330Tail36
def root330Tail34 : List Nat := 15193 :: root330Tail35
def root330Tail33 : List Nat := 15187 :: root330Tail34
def root330Tail32 : List Nat := 12433 :: root330Tail33
def root330Tail31 : List Nat := 12211 :: root330Tail32
def root330Tail30 : List Nat := 11953 :: root330Tail31
def root330Tail29 : List Nat := 11719 :: root330Tail30
def root330Tail28 : List Nat := 11689 :: root330Tail29
def root330Tail27 : List Nat := 11329 :: root330Tail28
def root330Tail26 : List Nat := 10069 :: root330Tail27
def root330Tail25 : List Nat := 9439 :: root330Tail26
def root330Tail24 : List Nat := 9283 :: root330Tail25
def root330Tail23 : List Nat := 8923 :: root330Tail24
def root330Tail22 : List Nat := 8419 :: root330Tail23
def root330Tail21 : List Nat := 8329 :: root330Tail22
def root330Tail20 : List Nat := 8221 :: root330Tail21
def root330Tail19 : List Nat := 7951 :: root330Tail20
def root330Tail18 : List Nat := 7411 :: root330Tail19
def root330Tail17 : List Nat := 7393 :: root330Tail18
def root330Tail16 : List Nat := 7351 :: root330Tail17
def root330Tail15 : List Nat := 7237 :: root330Tail16
def root330Tail14 : List Nat := 7027 :: root330Tail15
def root330Tail13 : List Nat := 6823 :: root330Tail14
def root330Tail12 : List Nat := 6481 :: root330Tail13
def root330Tail11 : List Nat := 6007 :: root330Tail12
def root330Tail10 : List Nat := 5683 :: root330Tail11
def root330Tail9 : List Nat := 5557 :: root330Tail10
def root330Tail8 : List Nat := 5209 :: root330Tail9
def root330Tail7 : List Nat := 5179 :: root330Tail8
def root330Tail6 : List Nat := 4987 :: root330Tail7
def root330Tail5 : List Nat := 4441 :: root330Tail6
def root330Tail4 : List Nat := 4261 :: root330Tail5
def root330Tail3 : List Nat := 4129 :: root330Tail4
def root330Tail2 : List Nat := 3919 :: root330Tail3
def root330Tail1 : List Nat := 3853 :: root330Tail2
def root330Tail0 : List Nat := 2857 :: root330Tail1
def root330Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root330Candidates0_checked : candidates edge 4 2239 2857 root330Tail1 winner.bound = root330Candidates0 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
theorem root330Step0 : scan edge (search edge 3) 4 [2239] 2239 root330Tail0 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail1 winner := by
  rw [root330Tail0, scan, if_neg (by decide), root330Candidates0_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail1
    (if 3 ≤ root330Candidates0.length then search edge 3 [2239, 2857] root330Candidates0 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates1 : List Nat := [5683, 7237, 8923]
set_option maxRecDepth 8192 in
theorem root330Candidates1_checked : candidates edge 4 2239 3853 root330Tail2 winner.bound = root330Candidates1 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root330Child1 : search edge 3 [2239, 3853] root330Candidates1 6092 winner = winner := by decide
theorem root330Step1 : scan edge (search edge 3) 4 [2239] 2239 root330Tail1 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail2 winner := by
  rw [root330Tail1, scan, if_neg (by decide), root330Candidates1_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail2
    (if 3 ≤ root330Candidates1.length then search edge 3 [2239, 3853] root330Candidates1 6092 winner else winner) = _
  rw [if_pos (by decide), root330Child1]
def root330Candidates2 : List Nat := [5179, 6007]
set_option maxRecDepth 8192 in
theorem root330Candidates2_checked : candidates edge 4 2239 3919 root330Tail3 winner.bound = root330Candidates2 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root330Step2 : scan edge (search edge 3) 4 [2239] 2239 root330Tail2 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail3 winner := by
  rw [root330Tail2, scan, if_neg (by decide), root330Candidates2_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail3
    (if 3 ≤ root330Candidates2.length then search edge 3 [2239, 3919] root330Candidates2 6158 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates3 : List Nat := [8221, 17389]
set_option maxRecDepth 8192 in
theorem root330Candidates3_checked : candidates edge 4 2239 4129 root330Tail4 winner.bound = root330Candidates3 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root330Step3 : scan edge (search edge 3) 4 [2239] 2239 root330Tail3 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail4 winner := by
  rw [root330Tail3, scan, if_neg (by decide), root330Candidates3_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail4
    (if 3 ≤ root330Candidates3.length then search edge 3 [2239, 4129] root330Candidates3 6368 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates4 : List Nat := [7951, 12433]
set_option maxRecDepth 8192 in
theorem root330Candidates4_checked : candidates edge 4 2239 4261 root330Tail5 winner.bound = root330Candidates4 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root330Step4 : scan edge (search edge 3) 4 [2239] 2239 root330Tail4 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail5 winner := by
  rw [root330Tail4, scan, if_neg (by decide), root330Candidates4_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail5
    (if 3 ≤ root330Candidates4.length then search edge 3 [2239, 4261] root330Candidates4 6500 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates5 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root330Candidates5_checked : candidates edge 4 2239 4441 root330Tail6 winner.bound = root330Candidates5 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root330Step5 : scan edge (search edge 3) 4 [2239] 2239 root330Tail5 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail6 winner := by
  rw [root330Tail5, scan, if_neg (by decide), root330Candidates5_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail6
    (if 3 ≤ root330Candidates5.length then search edge 3 [2239, 4441] root330Candidates5 6680 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root330Candidates6_checked : candidates edge 4 2239 4987 root330Tail7 winner.bound = root330Candidates6 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root330Step6 : scan edge (search edge 3) 4 [2239] 2239 root330Tail6 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail7 winner := by
  rw [root330Tail6, scan, if_neg (by decide), root330Candidates6_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail7
    (if 3 ≤ root330Candidates6.length then search edge 3 [2239, 4987] root330Candidates6 7226 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates7 : List Nat := [5683, 9439]
set_option maxRecDepth 8192 in
theorem root330Candidates7_checked : candidates edge 4 2239 5179 root330Tail8 winner.bound = root330Candidates7 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root330Step7 : scan edge (search edge 3) 4 [2239] 2239 root330Tail7 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail8 winner := by
  rw [root330Tail7, scan, if_neg (by decide), root330Candidates7_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail8
    (if 3 ≤ root330Candidates7.length then search edge 3 [2239, 5179] root330Candidates7 7418 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates8 : List Nat := [10069, 11329]
set_option maxRecDepth 8192 in
theorem root330Candidates8_checked : candidates edge 4 2239 5209 root330Tail9 winner.bound = root330Candidates8 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root330Step8 : scan edge (search edge 3) 4 [2239] 2239 root330Tail8 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail9 winner := by
  rw [root330Tail8, scan, if_neg (by decide), root330Candidates8_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail9
    (if 3 ≤ root330Candidates8.length then search edge 3 [2239, 5209] root330Candidates8 7448 winner else winner) = _
  rw [if_neg (by decide)]
def root330Candidates9 : List Nat := [7393, 10069, 17959]
set_option maxRecDepth 8192 in
theorem root330Candidates9_checked : candidates edge 4 2239 5557 root330Tail10 winner.bound = root330Candidates9 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root330Child9 : search edge 3 [2239, 5557] root330Candidates9 7796 winner = winner := by decide
theorem root330Step9 : scan edge (search edge 3) 4 [2239] 2239 root330Tail9 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail10 winner := by
  rw [root330Tail9, scan, if_neg (by decide), root330Candidates9_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail10
    (if 3 ≤ root330Candidates9.length then search edge 3 [2239, 5557] root330Candidates9 7796 winner else winner) = _
  rw [if_pos (by decide), root330Child9]
def root330Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root330Candidates10_checked : candidates edge 4 2239 5683 root330Tail11 winner.bound = root330Candidates10 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root330Step10 : scan edge (search edge 3) 4 [2239] 2239 root330Tail10 winner =
    scan edge (search edge 3) 4 [2239] 2239 root330Tail11 winner := by
  rw [root330Tail10, scan, if_neg (by decide), root330Candidates10_checked]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail11
    (if 3 ≤ root330Candidates10.length then search edge 3 [2239, 5683] root330Candidates10 7922 winner else winner) = _
  rw [if_neg (by decide)]
theorem root330Scan11 : scan edge (search edge 3) 4 [2239] 2239 root330Tail11 winner = winner := by rw [root330Tail11, scan, if_pos (by decide)]
theorem root330Scan10 : scan edge (search edge 3) 4 [2239] 2239 root330Tail10 winner = winner :=
  (root330Step10).trans root330Scan11
theorem root330Scan9 : scan edge (search edge 3) 4 [2239] 2239 root330Tail9 winner = winner :=
  (root330Step9).trans root330Scan10
theorem root330Scan8 : scan edge (search edge 3) 4 [2239] 2239 root330Tail8 winner = winner :=
  (root330Step8).trans root330Scan9
theorem root330Scan7 : scan edge (search edge 3) 4 [2239] 2239 root330Tail7 winner = winner :=
  (root330Step7).trans root330Scan8
theorem root330Scan6 : scan edge (search edge 3) 4 [2239] 2239 root330Tail6 winner = winner :=
  (root330Step6).trans root330Scan7
theorem root330Scan5 : scan edge (search edge 3) 4 [2239] 2239 root330Tail5 winner = winner :=
  (root330Step5).trans root330Scan6
theorem root330Scan4 : scan edge (search edge 3) 4 [2239] 2239 root330Tail4 winner = winner :=
  (root330Step4).trans root330Scan5
theorem root330Scan3 : scan edge (search edge 3) 4 [2239] 2239 root330Tail3 winner = winner :=
  (root330Step3).trans root330Scan4
theorem root330Scan2 : scan edge (search edge 3) 4 [2239] 2239 root330Tail2 winner = winner :=
  (root330Step2).trans root330Scan3
theorem root330Scan1 : scan edge (search edge 3) 4 [2239] 2239 root330Tail1 winner = winner :=
  (root330Step1).trans root330Scan2
theorem root330Scan0 : scan edge (search edge 3) 4 [2239] 2239 root330Tail0 winner = winner :=
  (root330Step0).trans root330Scan1
set_option maxRecDepth 4096 in
theorem root330Branch : search edge 4 [2239] root330Candidates 2239 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2239] 2239 root330Tail0 winner = winner
  exact root330Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
