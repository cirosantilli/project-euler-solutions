import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root594Candidates : List Nat := Data.row4373
set_option maxRecDepth 8192 in
theorem root594Candidates_checked : candidates edge 5 0 4373 tail595 winner.bound = root594Candidates := by
  apply candidates_eq_row edge 5 0 4373 tail595 root594Candidates winner.bound tail595_ordered
  · decide
  · decide
  · funext q
    rfl
def root594Tail37 : List Nat := []
def root594Tail36 : List Nat := 20981 :: root594Tail37
def root594Tail35 : List Nat := 20249 :: root594Tail36
def root594Tail34 : List Nat := 19853 :: root594Tail35
def root594Tail33 : List Nat := 18899 :: root594Tail34
def root594Tail32 : List Nat := 17159 :: root594Tail33
def root594Tail31 : List Nat := 16883 :: root594Tail32
def root594Tail30 : List Nat := 16631 :: root594Tail31
def root594Tail29 : List Nat := 15287 :: root594Tail30
def root594Tail28 : List Nat := 14747 :: root594Tail29
def root594Tail27 : List Nat := 14321 :: root594Tail28
def root594Tail26 : List Nat := 12983 :: root594Tail27
def root594Tail25 : List Nat := 12791 :: root594Tail26
def root594Tail24 : List Nat := 12497 :: root594Tail25
def root594Tail23 : List Nat := 12377 :: root594Tail24
def root594Tail22 : List Nat := 11807 :: root594Tail23
def root594Tail21 : List Nat := 11783 :: root594Tail22
def root594Tail20 : List Nat := 11471 :: root594Tail21
def root594Tail19 : List Nat := 10739 :: root594Tail20
def root594Tail18 : List Nat := 10487 :: root594Tail19
def root594Tail17 : List Nat := 10463 :: root594Tail18
def root594Tail16 : List Nat := 10163 :: root594Tail17
def root594Tail15 : List Nat := 9551 :: root594Tail16
def root594Tail14 : List Nat := 9431 :: root594Tail15
def root594Tail13 : List Nat := 8609 :: root594Tail14
def root594Tail12 : List Nat := 8423 :: root594Tail13
def root594Tail11 : List Nat := 8237 :: root594Tail12
def root594Tail10 : List Nat := 8147 :: root594Tail11
def root594Tail9 : List Nat := 8039 :: root594Tail10
def root594Tail8 : List Nat := 7919 :: root594Tail9
def root594Tail7 : List Nat := 7541 :: root594Tail8
def root594Tail6 : List Nat := 7121 :: root594Tail7
def root594Tail5 : List Nat := 7019 :: root594Tail6
def root594Tail4 : List Nat := 5927 :: root594Tail5
def root594Tail3 : List Nat := 5801 :: root594Tail4
def root594Tail2 : List Nat := 5693 :: root594Tail3
def root594Tail1 : List Nat := 5297 :: root594Tail2
def root594Tail0 : List Nat := 5189 :: root594Tail1
def root594Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root594Candidates0_checked : candidates edge 4 4373 5189 root594Tail1 winner.bound = root594Candidates0 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root594Step0 : scan edge (search edge 3) 4 [4373] 4373 root594Tail0 winner =
    scan edge (search edge 3) 4 [4373] 4373 root594Tail1 winner := by
  rw [root594Tail0, scan, if_neg (by decide), root594Candidates0_checked]
  change scan edge (search edge 3) 4 [4373] 4373 root594Tail1
    (if 3 ≤ root594Candidates0.length then search edge 3 [4373, 5189] root594Candidates0 9562 winner else winner) = _
  rw [if_neg (by decide)]
def root594Candidates1 : List Nat := [7541, 11783]
set_option maxRecDepth 8192 in
theorem root594Candidates1_checked : candidates edge 4 4373 5297 root594Tail2 winner.bound = root594Candidates1 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root594Step1 : scan edge (search edge 3) 4 [4373] 4373 root594Tail1 winner =
    scan edge (search edge 3) 4 [4373] 4373 root594Tail2 winner := by
  rw [root594Tail1, scan, if_neg (by decide), root594Candidates1_checked]
  change scan edge (search edge 3) 4 [4373] 4373 root594Tail2
    (if 3 ≤ root594Candidates1.length then search edge 3 [4373, 5297] root594Candidates1 9670 winner else winner) = _
  rw [if_neg (by decide)]
theorem root594Scan2 : scan edge (search edge 3) 4 [4373] 4373 root594Tail2 winner = winner := by rw [root594Tail2, scan, if_pos (by decide)]
theorem root594Scan1 : scan edge (search edge 3) 4 [4373] 4373 root594Tail1 winner = winner :=
  (root594Step1).trans root594Scan2
theorem root594Scan0 : scan edge (search edge 3) 4 [4373] 4373 root594Tail0 winner = winner :=
  (root594Step0).trans root594Scan1
set_option maxRecDepth 4096 in
theorem root594Branch : search edge 4 [4373] root594Candidates 4373 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4373] 4373 root594Tail0 winner = winner
  exact root594Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
