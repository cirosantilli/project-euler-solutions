import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root556Candidates : List Nat := Data.row4051
set_option maxRecDepth 8192 in
theorem root556Candidates_checked : candidates edge 5 0 4051 tail557 winner.bound = root556Candidates := by
  apply candidates_eq_row edge 5 0 4051 tail557 root556Candidates winner.bound tail557_ordered
  · decide
  · decide
  · funext q
    rfl
def root556Tail40 : List Nat := []
def root556Tail39 : List Nat := 21481 :: root556Tail40
def root556Tail38 : List Nat := 21397 :: root556Tail39
def root556Tail37 : List Nat := 21187 :: root556Tail38
def root556Tail36 : List Nat := 20887 :: root556Tail37
def root556Tail35 : List Nat := 20161 :: root556Tail36
def root556Tail34 : List Nat := 20113 :: root556Tail35
def root556Tail33 : List Nat := 19609 :: root556Tail34
def root556Tail32 : List Nat := 18457 :: root556Tail33
def root556Tail31 : List Nat := 18289 :: root556Tail32
def root556Tail30 : List Nat := 17431 :: root556Tail31
def root556Tail29 : List Nat := 16141 :: root556Tail30
def root556Tail28 : List Nat := 15913 :: root556Tail29
def root556Tail27 : List Nat := 15259 :: root556Tail28
def root556Tail26 : List Nat := 15013 :: root556Tail27
def root556Tail25 : List Nat := 14173 :: root556Tail26
def root556Tail24 : List Nat := 13873 :: root556Tail25
def root556Tail23 : List Nat := 12613 :: root556Tail24
def root556Tail22 : List Nat := 12451 :: root556Tail23
def root556Tail21 : List Nat := 12157 :: root556Tail22
def root556Tail20 : List Nat := 10957 :: root556Tail21
def root556Tail19 : List Nat := 10789 :: root556Tail20
def root556Tail18 : List Nat := 10729 :: root556Tail19
def root556Tail17 : List Nat := 9133 :: root556Tail18
def root556Tail16 : List Nat := 8689 :: root556Tail17
def root556Tail15 : List Nat := 8563 :: root556Tail16
def root556Tail14 : List Nat := 8311 :: root556Tail15
def root556Tail13 : List Nat := 8263 :: root556Tail14
def root556Tail12 : List Nat := 7951 :: root556Tail13
def root556Tail11 : List Nat := 7591 :: root556Tail12
def root556Tail10 : List Nat := 7549 :: root556Tail11
def root556Tail9 : List Nat := 7219 :: root556Tail10
def root556Tail8 : List Nat := 7027 :: root556Tail9
def root556Tail7 : List Nat := 6841 :: root556Tail8
def root556Tail6 : List Nat := 6211 :: root556Tail7
def root556Tail5 : List Nat := 6121 :: root556Tail6
def root556Tail4 : List Nat := 5701 :: root556Tail5
def root556Tail3 : List Nat := 5521 :: root556Tail4
def root556Tail2 : List Nat := 5077 :: root556Tail3
def root556Tail1 : List Nat := 4657 :: root556Tail2
def root556Tail0 : List Nat := 4129 :: root556Tail1
def root556Candidates0 : List Nat := [5521]
set_option maxRecDepth 8192 in
theorem root556Candidates0_checked : candidates edge 4 4051 4129 root556Tail1 winner.bound = root556Candidates0 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root556Step0 : scan edge (search edge 3) 4 [4051] 4051 root556Tail0 winner =
    scan edge (search edge 3) 4 [4051] 4051 root556Tail1 winner := by
  rw [root556Tail0, scan, if_neg (by decide), root556Candidates0_checked]
  change scan edge (search edge 3) 4 [4051] 4051 root556Tail1
    (if 3 ≤ root556Candidates0.length then search edge 3 [4051, 4129] root556Candidates0 8180 winner else winner) = _
  rw [if_neg (by decide)]
def root556Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root556Candidates1_checked : candidates edge 4 4051 4657 root556Tail2 winner.bound = root556Candidates1 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root556Step1 : scan edge (search edge 3) 4 [4051] 4051 root556Tail1 winner =
    scan edge (search edge 3) 4 [4051] 4051 root556Tail2 winner := by
  rw [root556Tail1, scan, if_neg (by decide), root556Candidates1_checked]
  change scan edge (search edge 3) 4 [4051] 4051 root556Tail2
    (if 3 ≤ root556Candidates1.length then search edge 3 [4051, 4657] root556Candidates1 8708 winner else winner) = _
  rw [if_neg (by decide)]
def root556Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root556Candidates2_checked : candidates edge 4 4051 5077 root556Tail3 winner.bound = root556Candidates2 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root556Step2 : scan edge (search edge 3) 4 [4051] 4051 root556Tail2 winner =
    scan edge (search edge 3) 4 [4051] 4051 root556Tail3 winner := by
  rw [root556Tail2, scan, if_neg (by decide), root556Candidates2_checked]
  change scan edge (search edge 3) 4 [4051] 4051 root556Tail3
    (if 3 ≤ root556Candidates2.length then search edge 3 [4051, 5077] root556Candidates2 9128 winner else winner) = _
  rw [if_neg (by decide)]
theorem root556Scan3 : scan edge (search edge 3) 4 [4051] 4051 root556Tail3 winner = winner := by rw [root556Tail3, scan, if_pos (by decide)]
theorem root556Scan2 : scan edge (search edge 3) 4 [4051] 4051 root556Tail2 winner = winner :=
  (root556Step2).trans root556Scan3
theorem root556Scan1 : scan edge (search edge 3) 4 [4051] 4051 root556Tail1 winner = winner :=
  (root556Step1).trans root556Scan2
theorem root556Scan0 : scan edge (search edge 3) 4 [4051] 4051 root556Tail0 winner = winner :=
  (root556Step0).trans root556Scan1
set_option maxRecDepth 4096 in
theorem root556Branch : search edge 4 [4051] root556Candidates 4051 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4051] 4051 root556Tail0 winner = winner
  exact root556Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
