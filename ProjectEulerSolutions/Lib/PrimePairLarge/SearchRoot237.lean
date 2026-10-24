import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root237Candidates : List Nat := Data.row1511
set_option maxRecDepth 8192 in
theorem root237Candidates_checked : candidates edge 5 0 1511 tail238 winner.bound = root237Candidates := by
  apply candidates_eq_row edge 5 0 1511 tail238 root237Candidates winner.bound tail238_ordered
  · decide
  · decide
  · funext q
    rfl
def root237Tail46 : List Nat := []
def root237Tail45 : List Nat := 24137 :: root237Tail46
def root237Tail44 : List Nat := 24107 :: root237Tail45
def root237Tail43 : List Nat := 23447 :: root237Tail44
def root237Tail42 : List Nat := 22739 :: root237Tail43
def root237Tail41 : List Nat := 22709 :: root237Tail42
def root237Tail40 : List Nat := 20639 :: root237Tail41
def root237Tail39 : List Nat := 20549 :: root237Tail40
def root237Tail38 : List Nat := 20117 :: root237Tail39
def root237Tail37 : List Nat := 19853 :: root237Tail38
def root237Tail36 : List Nat := 18917 :: root237Tail37
def root237Tail35 : List Nat := 18089 :: root237Tail36
def root237Tail34 : List Nat := 17921 :: root237Tail35
def root237Tail33 : List Nat := 17903 :: root237Tail34
def root237Tail32 : List Nat := 17807 :: root237Tail33
def root237Tail31 : List Nat := 17207 :: root237Tail32
def root237Tail30 : List Nat := 16649 :: root237Tail31
def root237Tail29 : List Nat := 15467 :: root237Tail30
def root237Tail28 : List Nat := 15149 :: root237Tail29
def root237Tail27 : List Nat := 14591 :: root237Tail28
def root237Tail26 : List Nat := 14519 :: root237Tail27
def root237Tail25 : List Nat := 13397 :: root237Tail26
def root237Tail24 : List Nat := 13049 :: root237Tail25
def root237Tail23 : List Nat := 12743 :: root237Tail24
def root237Tail22 : List Nat := 11423 :: root237Tail23
def root237Tail21 : List Nat := 11273 :: root237Tail22
def root237Tail20 : List Nat := 10559 :: root237Tail21
def root237Tail19 : List Nat := 10181 :: root237Tail20
def root237Tail18 : List Nat := 9803 :: root237Tail19
def root237Tail17 : List Nat := 9371 :: root237Tail18
def root237Tail16 : List Nat := 9311 :: root237Tail17
def root237Tail15 : List Nat := 9161 :: root237Tail16
def root237Tail14 : List Nat := 8963 :: root237Tail15
def root237Tail13 : List Nat := 8513 :: root237Tail14
def root237Tail12 : List Nat := 7853 :: root237Tail13
def root237Tail11 : List Nat := 7043 :: root237Tail12
def root237Tail10 : List Nat := 5981 :: root237Tail11
def root237Tail9 : List Nat := 5843 :: root237Tail10
def root237Tail8 : List Nat := 5099 :: root237Tail9
def root237Tail7 : List Nat := 5039 :: root237Tail8
def root237Tail6 : List Nat := 4889 :: root237Tail7
def root237Tail5 : List Nat := 4373 :: root237Tail6
def root237Tail4 : List Nat := 3389 :: root237Tail5
def root237Tail3 : List Nat := 3011 :: root237Tail4
def root237Tail2 : List Nat := 2267 :: root237Tail3
def root237Tail1 : List Nat := 2129 :: root237Tail2
def root237Tail0 : List Nat := 1571 :: root237Tail1
def root237Candidates0 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root237Candidates0_checked : candidates edge 4 1511 1571 root237Tail1 winner.bound = root237Candidates0 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
theorem root237Step0 : scan edge (search edge 3) 4 [1511] 1511 root237Tail0 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail1 winner := by
  rw [root237Tail0, scan, if_neg (by decide), root237Candidates0_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail1
    (if 3 ≤ root237Candidates0.length then search edge 3 [1511, 1571] root237Candidates0 3082 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates1_checked : candidates edge 4 1511 2129 root237Tail2 winner.bound = root237Candidates1 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root237Step1 : scan edge (search edge 3) 4 [1511] 1511 root237Tail1 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail2 winner := by
  rw [root237Tail1, scan, if_neg (by decide), root237Candidates1_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail2
    (if 3 ≤ root237Candidates1.length then search edge 3 [1511, 2129] root237Candidates1 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates2 : List Nat := [5843]
set_option maxRecDepth 8192 in
theorem root237Candidates2_checked : candidates edge 4 1511 2267 root237Tail3 winner.bound = root237Candidates2 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root237Step2 : scan edge (search edge 3) 4 [1511] 1511 root237Tail2 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail3 winner := by
  rw [root237Tail2, scan, if_neg (by decide), root237Candidates2_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail3
    (if 3 ≤ root237Candidates2.length then search edge 3 [1511, 2267] root237Candidates2 3778 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates3_checked : candidates edge 4 1511 3011 root237Tail4 winner.bound = root237Candidates3 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root237Step3 : scan edge (search edge 3) 4 [1511] 1511 root237Tail3 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail4 winner := by
  rw [root237Tail3, scan, if_neg (by decide), root237Candidates3_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail4
    (if 3 ≤ root237Candidates3.length then search edge 3 [1511, 3011] root237Candidates3 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates4_checked : candidates edge 4 1511 3389 root237Tail5 winner.bound = root237Candidates4 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root237Step4 : scan edge (search edge 3) 4 [1511] 1511 root237Tail4 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail5 winner := by
  rw [root237Tail4, scan, if_neg (by decide), root237Candidates4_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail5
    (if 3 ≤ root237Candidates4.length then search edge 3 [1511, 3389] root237Candidates4 4900 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates5 : List Nat := [19853]
set_option maxRecDepth 8192 in
theorem root237Candidates5_checked : candidates edge 4 1511 4373 root237Tail6 winner.bound = root237Candidates5 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root237Step5 : scan edge (search edge 3) 4 [1511] 1511 root237Tail5 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail6 winner := by
  rw [root237Tail5, scan, if_neg (by decide), root237Candidates5_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail6
    (if 3 ≤ root237Candidates5.length then search edge 3 [1511, 4373] root237Candidates5 5884 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates6_checked : candidates edge 4 1511 4889 root237Tail7 winner.bound = root237Candidates6 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root237Step6 : scan edge (search edge 3) 4 [1511] 1511 root237Tail6 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail7 winner := by
  rw [root237Tail6, scan, if_neg (by decide), root237Candidates6_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail7
    (if 3 ≤ root237Candidates6.length then search edge 3 [1511, 4889] root237Candidates6 6400 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates7_checked : candidates edge 4 1511 5039 root237Tail8 winner.bound = root237Candidates7 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root237Step7 : scan edge (search edge 3) 4 [1511] 1511 root237Tail7 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail8 winner := by
  rw [root237Tail7, scan, if_neg (by decide), root237Candidates7_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail8
    (if 3 ≤ root237Candidates7.length then search edge 3 [1511, 5039] root237Candidates7 6550 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates8 : List Nat := [7853]
set_option maxRecDepth 8192 in
theorem root237Candidates8_checked : candidates edge 4 1511 5099 root237Tail9 winner.bound = root237Candidates8 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root237Step8 : scan edge (search edge 3) 4 [1511] 1511 root237Tail8 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail9 winner := by
  rw [root237Tail8, scan, if_neg (by decide), root237Candidates8_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail9
    (if 3 ≤ root237Candidates8.length then search edge 3 [1511, 5099] root237Candidates8 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root237Candidates9_checked : candidates edge 4 1511 5843 root237Tail10 winner.bound = root237Candidates9 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root237Step9 : scan edge (search edge 3) 4 [1511] 1511 root237Tail9 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail10 winner := by
  rw [root237Tail9, scan, if_neg (by decide), root237Candidates9_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail10
    (if 3 ≤ root237Candidates9.length then search edge 3 [1511, 5843] root237Candidates9 7354 winner else winner) = _
  rw [if_neg (by decide)]
def root237Candidates10 : List Nat := [17903]
set_option maxRecDepth 8192 in
theorem root237Candidates10_checked : candidates edge 4 1511 5981 root237Tail11 winner.bound = root237Candidates10 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root237Step10 : scan edge (search edge 3) 4 [1511] 1511 root237Tail10 winner =
    scan edge (search edge 3) 4 [1511] 1511 root237Tail11 winner := by
  rw [root237Tail10, scan, if_neg (by decide), root237Candidates10_checked]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail11
    (if 3 ≤ root237Candidates10.length then search edge 3 [1511, 5981] root237Candidates10 7492 winner else winner) = _
  rw [if_neg (by decide)]
theorem root237Scan11 : scan edge (search edge 3) 4 [1511] 1511 root237Tail11 winner = winner := by rw [root237Tail11, scan, if_pos (by decide)]
theorem root237Scan10 : scan edge (search edge 3) 4 [1511] 1511 root237Tail10 winner = winner :=
  (root237Step10).trans root237Scan11
theorem root237Scan9 : scan edge (search edge 3) 4 [1511] 1511 root237Tail9 winner = winner :=
  (root237Step9).trans root237Scan10
theorem root237Scan8 : scan edge (search edge 3) 4 [1511] 1511 root237Tail8 winner = winner :=
  (root237Step8).trans root237Scan9
theorem root237Scan7 : scan edge (search edge 3) 4 [1511] 1511 root237Tail7 winner = winner :=
  (root237Step7).trans root237Scan8
theorem root237Scan6 : scan edge (search edge 3) 4 [1511] 1511 root237Tail6 winner = winner :=
  (root237Step6).trans root237Scan7
theorem root237Scan5 : scan edge (search edge 3) 4 [1511] 1511 root237Tail5 winner = winner :=
  (root237Step5).trans root237Scan6
theorem root237Scan4 : scan edge (search edge 3) 4 [1511] 1511 root237Tail4 winner = winner :=
  (root237Step4).trans root237Scan5
theorem root237Scan3 : scan edge (search edge 3) 4 [1511] 1511 root237Tail3 winner = winner :=
  (root237Step3).trans root237Scan4
theorem root237Scan2 : scan edge (search edge 3) 4 [1511] 1511 root237Tail2 winner = winner :=
  (root237Step2).trans root237Scan3
theorem root237Scan1 : scan edge (search edge 3) 4 [1511] 1511 root237Tail1 winner = winner :=
  (root237Step1).trans root237Scan2
theorem root237Scan0 : scan edge (search edge 3) 4 [1511] 1511 root237Tail0 winner = winner :=
  (root237Step0).trans root237Scan1
set_option maxRecDepth 4096 in
theorem root237Branch : search edge 4 [1511] root237Candidates 1511 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1511] 1511 root237Tail0 winner = winner
  exact root237Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
