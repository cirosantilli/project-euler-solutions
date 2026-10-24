import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root562Candidates : List Nat := Data.row4099
set_option maxRecDepth 8192 in
theorem root562Candidates_checked : candidates edge 5 0 4099 tail563 winner.bound = root562Candidates := by
  apply candidates_eq_row edge 5 0 4099 tail563 root562Candidates winner.bound tail563_ordered
  · decide
  · decide
  · funext q
    rfl
def root562Tail27 : List Nat := []
def root562Tail26 : List Nat := 21397 :: root562Tail27
def root562Tail25 : List Nat := 19219 :: root562Tail26
def root562Tail24 : List Nat := 19141 :: root562Tail25
def root562Tail23 : List Nat := 18211 :: root562Tail24
def root562Tail22 : List Nat := 18133 :: root562Tail23
def root562Tail21 : List Nat := 16993 :: root562Tail22
def root562Tail20 : List Nat := 16693 :: root562Tail21
def root562Tail19 : List Nat := 15901 :: root562Tail20
def root562Tail18 : List Nat := 15817 :: root562Tail19
def root562Tail17 : List Nat := 15787 :: root562Tail18
def root562Tail16 : List Nat := 15439 :: root562Tail17
def root562Tail15 : List Nat := 15061 :: root562Tail16
def root562Tail14 : List Nat := 14551 :: root562Tail15
def root562Tail13 : List Nat := 14221 :: root562Tail14
def root562Tail12 : List Nat := 13597 :: root562Tail13
def root562Tail11 : List Nat := 13267 :: root562Tail12
def root562Tail10 : List Nat := 13099 :: root562Tail11
def root562Tail9 : List Nat := 12757 :: root562Tail10
def root562Tail8 : List Nat := 12553 :: root562Tail9
def root562Tail7 : List Nat := 12157 :: root562Tail8
def root562Tail6 : List Nat := 10909 :: root562Tail7
def root562Tail5 : List Nat := 8821 :: root562Tail6
def root562Tail4 : List Nat := 6793 :: root562Tail5
def root562Tail3 : List Nat := 5953 :: root562Tail4
def root562Tail2 : List Nat := 4999 :: root562Tail3
def root562Tail1 : List Nat := 4909 :: root562Tail2
def root562Tail0 : List Nat := 4783 :: root562Tail1
def root562Candidates0 : List Nat := [4999]
set_option maxRecDepth 8192 in
theorem root562Candidates0_checked : candidates edge 4 4099 4783 root562Tail1 winner.bound = root562Candidates0 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root562Step0 : scan edge (search edge 3) 4 [4099] 4099 root562Tail0 winner =
    scan edge (search edge 3) 4 [4099] 4099 root562Tail1 winner := by
  rw [root562Tail0, scan, if_neg (by decide), root562Candidates0_checked]
  change scan edge (search edge 3) 4 [4099] 4099 root562Tail1
    (if 3 ≤ root562Candidates0.length then search edge 3 [4099, 4783] root562Candidates0 8882 winner else winner) = _
  rw [if_neg (by decide)]
def root562Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root562Candidates1_checked : candidates edge 4 4099 4909 root562Tail2 winner.bound = root562Candidates1 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root562Step1 : scan edge (search edge 3) 4 [4099] 4099 root562Tail1 winner =
    scan edge (search edge 3) 4 [4099] 4099 root562Tail2 winner := by
  rw [root562Tail1, scan, if_neg (by decide), root562Candidates1_checked]
  change scan edge (search edge 3) 4 [4099] 4099 root562Tail2
    (if 3 ≤ root562Candidates1.length then search edge 3 [4099, 4909] root562Candidates1 9008 winner else winner) = _
  rw [if_neg (by decide)]
def root562Candidates2 : List Nat := [15817]
set_option maxRecDepth 8192 in
theorem root562Candidates2_checked : candidates edge 4 4099 4999 root562Tail3 winner.bound = root562Candidates2 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root562Step2 : scan edge (search edge 3) 4 [4099] 4099 root562Tail2 winner =
    scan edge (search edge 3) 4 [4099] 4099 root562Tail3 winner := by
  rw [root562Tail2, scan, if_neg (by decide), root562Candidates2_checked]
  change scan edge (search edge 3) 4 [4099] 4099 root562Tail3
    (if 3 ≤ root562Candidates2.length then search edge 3 [4099, 4999] root562Candidates2 9098 winner else winner) = _
  rw [if_neg (by decide)]
theorem root562Scan3 : scan edge (search edge 3) 4 [4099] 4099 root562Tail3 winner = winner := by rw [root562Tail3, scan, if_pos (by decide)]
theorem root562Scan2 : scan edge (search edge 3) 4 [4099] 4099 root562Tail2 winner = winner :=
  (root562Step2).trans root562Scan3
theorem root562Scan1 : scan edge (search edge 3) 4 [4099] 4099 root562Tail1 winner = winner :=
  (root562Step1).trans root562Scan2
theorem root562Scan0 : scan edge (search edge 3) 4 [4099] 4099 root562Tail0 winner = winner :=
  (root562Step0).trans root562Scan1
set_option maxRecDepth 4096 in
theorem root562Branch : search edge 4 [4099] root562Candidates 4099 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4099] 4099 root562Tail0 winner = winner
  exact root562Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
