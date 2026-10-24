import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root530Candidates : List Nat := Data.row3847
set_option maxRecDepth 8192 in
theorem root530Candidates_checked : candidates edge 5 0 3847 tail531 winner.bound = root530Candidates := by
  apply candidates_eq_row edge 5 0 3847 tail531 root530Candidates winner.bound tail531_ordered
  · decide
  · decide
  · funext q
    rfl
def root530Tail41 : List Nat := []
def root530Tail40 : List Nat := 21751 :: root530Tail41
def root530Tail39 : List Nat := 21613 :: root530Tail40
def root530Tail38 : List Nat := 21319 :: root530Tail39
def root530Tail37 : List Nat := 21157 :: root530Tail38
def root530Tail36 : List Nat := 21061 :: root530Tail37
def root530Tail35 : List Nat := 19681 :: root530Tail36
def root530Tail34 : List Nat := 19507 :: root530Tail35
def root530Tail33 : List Nat := 19273 :: root530Tail34
def root530Tail32 : List Nat := 17239 :: root530Tail33
def root530Tail31 : List Nat := 17041 :: root530Tail32
def root530Tail30 : List Nat := 16567 :: root530Tail31
def root530Tail29 : List Nat := 16369 :: root530Tail30
def root530Tail28 : List Nat := 15277 :: root530Tail29
def root530Tail27 : List Nat := 13267 :: root530Tail28
def root530Tail26 : List Nat := 13183 :: root530Tail27
def root530Tail25 : List Nat := 12967 :: root530Tail26
def root530Tail24 : List Nat := 12451 :: root530Tail25
def root530Tail23 : List Nat := 12049 :: root530Tail24
def root530Tail22 : List Nat := 11863 :: root530Tail23
def root530Tail21 : List Nat := 10789 :: root530Tail22
def root530Tail20 : List Nat := 10141 :: root530Tail21
def root530Tail19 : List Nat := 10069 :: root530Tail20
def root530Tail18 : List Nat := 9949 :: root530Tail19
def root530Tail17 : List Nat := 9901 :: root530Tail18
def root530Tail16 : List Nat := 9817 :: root530Tail17
def root530Tail15 : List Nat := 8929 :: root530Tail16
def root530Tail14 : List Nat := 8887 :: root530Tail15
def root530Tail13 : List Nat := 8821 :: root530Tail14
def root530Tail12 : List Nat := 8233 :: root530Tail13
def root530Tail11 : List Nat := 8221 :: root530Tail12
def root530Tail10 : List Nat := 6883 :: root530Tail11
def root530Tail9 : List Nat := 6553 :: root530Tail10
def root530Tail8 : List Nat := 6367 :: root530Tail9
def root530Tail7 : List Nat := 6211 :: root530Tail8
def root530Tail6 : List Nat := 5737 :: root530Tail7
def root530Tail5 : List Nat := 5659 :: root530Tail6
def root530Tail4 : List Nat := 5197 :: root530Tail5
def root530Tail3 : List Nat := 5077 :: root530Tail4
def root530Tail2 : List Nat := 4099 :: root530Tail3
def root530Tail1 : List Nat := 4021 :: root530Tail2
def root530Tail0 : List Nat := 3931 :: root530Tail1
def root530Candidates0 : List Nat := [6553, 8233]
set_option maxRecDepth 8192 in
theorem root530Candidates0_checked : candidates edge 4 3847 3931 root530Tail1 winner.bound = root530Candidates0 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root530Step0 : scan edge (search edge 3) 4 [3847] 3847 root530Tail0 winner =
    scan edge (search edge 3) 4 [3847] 3847 root530Tail1 winner := by
  rw [root530Tail0, scan, if_neg (by decide), root530Candidates0_checked]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail1
    (if 3 ≤ root530Candidates0.length then search edge 3 [3847, 3931] root530Candidates0 7778 winner else winner) = _
  rw [if_neg (by decide)]
def root530Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root530Candidates1_checked : candidates edge 4 3847 4021 root530Tail2 winner.bound = root530Candidates1 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root530Step1 : scan edge (search edge 3) 4 [3847] 3847 root530Tail1 winner =
    scan edge (search edge 3) 4 [3847] 3847 root530Tail2 winner := by
  rw [root530Tail1, scan, if_neg (by decide), root530Candidates1_checked]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail2
    (if 3 ≤ root530Candidates1.length then search edge 3 [3847, 4021] root530Candidates1 7868 winner else winner) = _
  rw [if_neg (by decide)]
def root530Candidates2 : List Nat := [8821, 13267]
set_option maxRecDepth 8192 in
theorem root530Candidates2_checked : candidates edge 4 3847 4099 root530Tail3 winner.bound = root530Candidates2 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root530Step2 : scan edge (search edge 3) 4 [3847] 3847 root530Tail2 winner =
    scan edge (search edge 3) 4 [3847] 3847 root530Tail3 winner := by
  rw [root530Tail2, scan, if_neg (by decide), root530Candidates2_checked]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail3
    (if 3 ≤ root530Candidates2.length then search edge 3 [3847, 4099] root530Candidates2 7946 winner else winner) = _
  rw [if_neg (by decide)]
def root530Candidates3 : List Nat := [6367, 10141, 13267]
set_option maxRecDepth 8192 in
theorem root530Candidates3_checked : candidates edge 4 3847 5077 root530Tail4 winner.bound = root530Candidates3 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root530Child3 : search edge 3 [3847, 5077] root530Candidates3 8924 winner = winner := by decide
theorem root530Step3 : scan edge (search edge 3) 4 [3847] 3847 root530Tail3 winner =
    scan edge (search edge 3) 4 [3847] 3847 root530Tail4 winner := by
  rw [root530Tail3, scan, if_neg (by decide), root530Candidates3_checked]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail4
    (if 3 ≤ root530Candidates3.length then search edge 3 [3847, 5077] root530Candidates3 8924 winner else winner) = _
  rw [if_pos (by decide), root530Child3]
def root530Candidates4 : List Nat := [8221, 8929, 15277]
set_option maxRecDepth 8192 in
theorem root530Candidates4_checked : candidates edge 4 3847 5197 root530Tail5 winner.bound = root530Candidates4 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root530Child4 : search edge 3 [3847, 5197] root530Candidates4 9044 winner = winner := by decide
theorem root530Step4 : scan edge (search edge 3) 4 [3847] 3847 root530Tail4 winner =
    scan edge (search edge 3) 4 [3847] 3847 root530Tail5 winner := by
  rw [root530Tail4, scan, if_neg (by decide), root530Candidates4_checked]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail5
    (if 3 ≤ root530Candidates4.length then search edge 3 [3847, 5197] root530Candidates4 9044 winner else winner) = _
  rw [if_pos (by decide), root530Child4]
theorem root530Scan5 : scan edge (search edge 3) 4 [3847] 3847 root530Tail5 winner = winner := by rw [root530Tail5, scan, if_pos (by decide)]
theorem root530Scan4 : scan edge (search edge 3) 4 [3847] 3847 root530Tail4 winner = winner :=
  (root530Step4).trans root530Scan5
theorem root530Scan3 : scan edge (search edge 3) 4 [3847] 3847 root530Tail3 winner = winner :=
  (root530Step3).trans root530Scan4
theorem root530Scan2 : scan edge (search edge 3) 4 [3847] 3847 root530Tail2 winner = winner :=
  (root530Step2).trans root530Scan3
theorem root530Scan1 : scan edge (search edge 3) 4 [3847] 3847 root530Tail1 winner = winner :=
  (root530Step1).trans root530Scan2
theorem root530Scan0 : scan edge (search edge 3) 4 [3847] 3847 root530Tail0 winner = winner :=
  (root530Step0).trans root530Scan1
set_option maxRecDepth 4096 in
theorem root530Branch : search edge 4 [3847] root530Candidates 3847 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3847] 3847 root530Tail0 winner = winner
  exact root530Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
