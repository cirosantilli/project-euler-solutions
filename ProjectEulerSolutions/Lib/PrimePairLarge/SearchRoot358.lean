import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root358Candidates : List Nat := Data.row2437
set_option maxRecDepth 8192 in
theorem root358Candidates_checked : candidates edge 5 0 2437 tail359 winner.bound = root358Candidates := by
  apply candidates_eq_row edge 5 0 2437 tail359 root358Candidates winner.bound tail359_ordered
  · decide
  · decide
  · funext q
    rfl
def root358Tail36 : List Nat := []
def root358Tail35 : List Nat := 23251 :: root358Tail36
def root358Tail34 : List Nat := 22669 :: root358Tail35
def root358Tail33 : List Nat := 20929 :: root358Tail34
def root358Tail32 : List Nat := 19309 :: root358Tail33
def root358Tail31 : List Nat := 18451 :: root358Tail32
def root358Tail30 : List Nat := 18169 :: root358Tail31
def root358Tail29 : List Nat := 18127 :: root358Tail30
def root358Tail28 : List Nat := 17863 :: root358Tail29
def root358Tail27 : List Nat := 17599 :: root358Tail28
def root358Tail26 : List Nat := 16741 :: root358Tail27
def root358Tail25 : List Nat := 16519 :: root358Tail26
def root358Tail24 : List Nat := 15733 :: root358Tail25
def root358Tail23 : List Nat := 15319 :: root358Tail24
def root358Tail22 : List Nat := 14011 :: root358Tail23
def root358Tail21 : List Nat := 13597 :: root358Tail22
def root358Tail20 : List Nat := 12379 :: root358Tail21
def root358Tail19 : List Nat := 12277 :: root358Tail20
def root358Tail18 : List Nat := 10501 :: root358Tail19
def root358Tail17 : List Nat := 10429 :: root358Tail18
def root358Tail16 : List Nat := 8353 :: root358Tail17
def root358Tail15 : List Nat := 7039 :: root358Tail16
def root358Tail14 : List Nat := 7027 :: root358Tail15
def root358Tail13 : List Nat := 6967 :: root358Tail14
def root358Tail12 : List Nat := 6607 :: root358Tail13
def root358Tail11 : List Nat := 6337 :: root358Tail12
def root358Tail10 : List Nat := 6301 :: root358Tail11
def root358Tail9 : List Nat := 6073 :: root358Tail10
def root358Tail8 : List Nat := 5077 :: root358Tail9
def root358Tail7 : List Nat := 4969 :: root358Tail8
def root358Tail6 : List Nat := 4621 :: root358Tail7
def root358Tail5 : List Nat := 4111 :: root358Tail6
def root358Tail4 : List Nat := 3931 :: root358Tail5
def root358Tail3 : List Nat := 3889 :: root358Tail4
def root358Tail2 : List Nat := 3541 :: root358Tail3
def root358Tail1 : List Nat := 3361 :: root358Tail2
def root358Tail0 : List Nat := 3301 :: root358Tail1
def root358Candidates0 : List Nat := [6967, 14011]
set_option maxRecDepth 8192 in
theorem root358Candidates0_checked : candidates edge 4 2437 3301 root358Tail1 winner.bound = root358Candidates0 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root358Step0 : scan edge (search edge 3) 4 [2437] 2437 root358Tail0 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail1 winner := by
  rw [root358Tail0, scan, if_neg (by decide), root358Candidates0_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail1
    (if 3 ≤ root358Candidates0.length then search edge 3 [2437, 3301] root358Candidates0 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root358Candidates1_checked : candidates edge 4 2437 3361 root358Tail2 winner.bound = root358Candidates1 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root358Step1 : scan edge (search edge 3) 4 [2437] 2437 root358Tail1 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail2 winner := by
  rw [root358Tail1, scan, if_neg (by decide), root358Candidates1_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail2
    (if 3 ≤ root358Candidates1.length then search edge 3 [2437, 3361] root358Candidates1 5798 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates2 : List Nat := [18169]
set_option maxRecDepth 8192 in
theorem root358Candidates2_checked : candidates edge 4 2437 3541 root358Tail3 winner.bound = root358Candidates2 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root358Step2 : scan edge (search edge 3) 4 [2437] 2437 root358Tail2 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail3 winner := by
  rw [root358Tail2, scan, if_neg (by decide), root358Candidates2_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail3
    (if 3 ≤ root358Candidates2.length then search edge 3 [2437, 3541] root358Candidates2 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates3 : List Nat := [13597, 19309]
set_option maxRecDepth 8192 in
theorem root358Candidates3_checked : candidates edge 4 2437 3889 root358Tail4 winner.bound = root358Candidates3 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root358Step3 : scan edge (search edge 3) 4 [2437] 2437 root358Tail3 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail4 winner := by
  rw [root358Tail3, scan, if_neg (by decide), root358Candidates3_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail4
    (if 3 ≤ root358Candidates3.length then search edge 3 [2437, 3889] root358Candidates3 6326 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root358Candidates4_checked : candidates edge 4 2437 3931 root358Tail5 winner.bound = root358Candidates4 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root358Step4 : scan edge (search edge 3) 4 [2437] 2437 root358Tail4 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail5 winner := by
  rw [root358Tail4, scan, if_neg (by decide), root358Candidates4_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail5
    (if 3 ≤ root358Candidates4.length then search edge 3 [2437, 3931] root358Candidates4 6368 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates5 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root358Candidates5_checked : candidates edge 4 2437 4111 root358Tail6 winner.bound = root358Candidates5 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root358Step5 : scan edge (search edge 3) 4 [2437] 2437 root358Tail5 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail6 winner := by
  rw [root358Tail5, scan, if_neg (by decide), root358Candidates5_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail6
    (if 3 ≤ root358Candidates5.length then search edge 3 [2437, 4111] root358Candidates5 6548 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates6 : List Nat := [15733]
set_option maxRecDepth 8192 in
theorem root358Candidates6_checked : candidates edge 4 2437 4621 root358Tail7 winner.bound = root358Candidates6 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root358Step6 : scan edge (search edge 3) 4 [2437] 2437 root358Tail6 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail7 winner := by
  rw [root358Tail6, scan, if_neg (by decide), root358Candidates6_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail7
    (if 3 ≤ root358Candidates6.length then search edge 3 [2437, 4621] root358Candidates6 7058 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root358Candidates7_checked : candidates edge 4 2437 4969 root358Tail8 winner.bound = root358Candidates7 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root358Step7 : scan edge (search edge 3) 4 [2437] 2437 root358Tail7 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail8 winner := by
  rw [root358Tail7, scan, if_neg (by decide), root358Candidates7_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail8
    (if 3 ≤ root358Candidates7.length then search edge 3 [2437, 4969] root358Candidates7 7406 winner else winner) = _
  rw [if_neg (by decide)]
def root358Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root358Candidates8_checked : candidates edge 4 2437 5077 root358Tail9 winner.bound = root358Candidates8 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root358Step8 : scan edge (search edge 3) 4 [2437] 2437 root358Tail8 winner =
    scan edge (search edge 3) 4 [2437] 2437 root358Tail9 winner := by
  rw [root358Tail8, scan, if_neg (by decide), root358Candidates8_checked]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail9
    (if 3 ≤ root358Candidates8.length then search edge 3 [2437, 5077] root358Candidates8 7514 winner else winner) = _
  rw [if_neg (by decide)]
theorem root358Scan9 : scan edge (search edge 3) 4 [2437] 2437 root358Tail9 winner = winner := by rw [root358Tail9, scan, if_pos (by decide)]
theorem root358Scan8 : scan edge (search edge 3) 4 [2437] 2437 root358Tail8 winner = winner :=
  (root358Step8).trans root358Scan9
theorem root358Scan7 : scan edge (search edge 3) 4 [2437] 2437 root358Tail7 winner = winner :=
  (root358Step7).trans root358Scan8
theorem root358Scan6 : scan edge (search edge 3) 4 [2437] 2437 root358Tail6 winner = winner :=
  (root358Step6).trans root358Scan7
theorem root358Scan5 : scan edge (search edge 3) 4 [2437] 2437 root358Tail5 winner = winner :=
  (root358Step5).trans root358Scan6
theorem root358Scan4 : scan edge (search edge 3) 4 [2437] 2437 root358Tail4 winner = winner :=
  (root358Step4).trans root358Scan5
theorem root358Scan3 : scan edge (search edge 3) 4 [2437] 2437 root358Tail3 winner = winner :=
  (root358Step3).trans root358Scan4
theorem root358Scan2 : scan edge (search edge 3) 4 [2437] 2437 root358Tail2 winner = winner :=
  (root358Step2).trans root358Scan3
theorem root358Scan1 : scan edge (search edge 3) 4 [2437] 2437 root358Tail1 winner = winner :=
  (root358Step1).trans root358Scan2
theorem root358Scan0 : scan edge (search edge 3) 4 [2437] 2437 root358Tail0 winner = winner :=
  (root358Step0).trans root358Scan1
set_option maxRecDepth 4096 in
theorem root358Branch : search edge 4 [2437] root358Candidates 2437 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2437] 2437 root358Tail0 winner = winner
  exact root358Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
