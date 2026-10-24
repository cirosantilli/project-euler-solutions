import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root635Candidates : List Nat := Data.row4729
set_option maxRecDepth 8192 in
theorem root635Candidates_checked : candidates edge 5 0 4729 tail636 winner.bound = root635Candidates := by
  apply candidates_eq_row edge 5 0 4729 tail636 root635Candidates winner.bound tail636_ordered
  · decide
  · decide
  · funext q
    rfl
def root635Tail44 : List Nat := []
def root635Tail43 : List Nat := 20947 :: root635Tail44
def root635Tail42 : List Nat := 20743 :: root635Tail43
def root635Tail41 : List Nat := 20269 :: root635Tail42
def root635Tail40 : List Nat := 20143 :: root635Tail41
def root635Tail39 : List Nat := 19927 :: root635Tail40
def root635Tail38 : List Nat := 19597 :: root635Tail39
def root635Tail37 : List Nat := 19483 :: root635Tail38
def root635Tail36 : List Nat := 19441 :: root635Tail37
def root635Tail35 : List Nat := 19219 :: root635Tail36
def root635Tail34 : List Nat := 18637 :: root635Tail35
def root635Tail33 : List Nat := 18289 :: root635Tail34
def root635Tail32 : List Nat := 18181 :: root635Tail33
def root635Tail31 : List Nat := 16921 :: root635Tail32
def root635Tail30 : List Nat := 16879 :: root635Tail31
def root635Tail29 : List Nat := 16651 :: root635Tail30
def root635Tail28 : List Nat := 15559 :: root635Tail29
def root635Tail27 : List Nat := 15439 :: root635Tail28
def root635Tail26 : List Nat := 15349 :: root635Tail27
def root635Tail25 : List Nat := 15277 :: root635Tail26
def root635Tail24 : List Nat := 14437 :: root635Tail25
def root635Tail23 : List Nat := 12829 :: root635Tail24
def root635Tail22 : List Nat := 12619 :: root635Tail23
def root635Tail21 : List Nat := 12457 :: root635Tail22
def root635Tail20 : List Nat := 12301 :: root635Tail21
def root635Tail19 : List Nat := 12289 :: root635Tail20
def root635Tail18 : List Nat := 11251 :: root635Tail19
def root635Tail17 : List Nat := 11239 :: root635Tail18
def root635Tail16 : List Nat := 10831 :: root635Tail17
def root635Tail15 : List Nat := 9859 :: root635Tail16
def root635Tail14 : List Nat := 9601 :: root635Tail15
def root635Tail13 : List Nat := 9181 :: root635Tail14
def root635Tail12 : List Nat := 8887 :: root635Tail13
def root635Tail11 : List Nat := 8863 :: root635Tail12
def root635Tail10 : List Nat := 8431 :: root635Tail11
def root635Tail9 : List Nat := 7963 :: root635Tail10
def root635Tail8 : List Nat := 7603 :: root635Tail9
def root635Tail7 : List Nat := 7207 :: root635Tail8
def root635Tail6 : List Nat := 6961 :: root635Tail7
def root635Tail5 : List Nat := 5791 :: root635Tail6
def root635Tail4 : List Nat := 5749 :: root635Tail5
def root635Tail3 : List Nat := 5407 :: root635Tail4
def root635Tail2 : List Nat := 5077 :: root635Tail3
def root635Tail1 : List Nat := 4999 :: root635Tail2
def root635Tail0 : List Nat := 4909 :: root635Tail1
def root635Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root635Candidates0_checked : candidates edge 4 4729 4909 root635Tail1 winner.bound = root635Candidates0 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root635Step0 : scan edge (search edge 3) 4 [4729] 4729 root635Tail0 winner =
    scan edge (search edge 3) 4 [4729] 4729 root635Tail1 winner := by
  rw [root635Tail0, scan, if_neg (by decide), root635Candidates0_checked]
  change scan edge (search edge 3) 4 [4729] 4729 root635Tail1
    (if 3 ≤ root635Candidates0.length then search edge 3 [4729, 4909] root635Candidates0 9638 winner else winner) = _
  rw [if_neg (by decide)]
def root635Candidates1 : List Nat := [5791]
set_option maxRecDepth 8192 in
theorem root635Candidates1_checked : candidates edge 4 4729 4999 root635Tail2 winner.bound = root635Candidates1 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root635Step1 : scan edge (search edge 3) 4 [4729] 4729 root635Tail1 winner =
    scan edge (search edge 3) 4 [4729] 4729 root635Tail2 winner := by
  rw [root635Tail1, scan, if_neg (by decide), root635Candidates1_checked]
  change scan edge (search edge 3) 4 [4729] 4729 root635Tail2
    (if 3 ≤ root635Candidates1.length then search edge 3 [4729, 4999] root635Candidates1 9728 winner else winner) = _
  rw [if_neg (by decide)]
def root635Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root635Candidates2_checked : candidates edge 4 4729 5077 root635Tail3 winner.bound = root635Candidates2 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root635Step2 : scan edge (search edge 3) 4 [4729] 4729 root635Tail2 winner =
    scan edge (search edge 3) 4 [4729] 4729 root635Tail3 winner := by
  rw [root635Tail2, scan, if_neg (by decide), root635Candidates2_checked]
  change scan edge (search edge 3) 4 [4729] 4729 root635Tail3
    (if 3 ≤ root635Candidates2.length then search edge 3 [4729, 5077] root635Candidates2 9806 winner else winner) = _
  rw [if_neg (by decide)]
theorem root635Scan3 : scan edge (search edge 3) 4 [4729] 4729 root635Tail3 winner = winner := by rw [root635Tail3, scan, if_pos (by decide)]
theorem root635Scan2 : scan edge (search edge 3) 4 [4729] 4729 root635Tail2 winner = winner :=
  (root635Step2).trans root635Scan3
theorem root635Scan1 : scan edge (search edge 3) 4 [4729] 4729 root635Tail1 winner = winner :=
  (root635Step1).trans root635Scan2
theorem root635Scan0 : scan edge (search edge 3) 4 [4729] 4729 root635Tail0 winner = winner :=
  (root635Step0).trans root635Scan1
set_option maxRecDepth 4096 in
theorem root635Branch : search edge 4 [4729] root635Candidates 4729 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4729] 4729 root635Tail0 winner = winner
  exact root635Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
