import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root328Candidates : List Nat := Data.row2221
set_option maxRecDepth 8192 in
theorem root328Candidates_checked : candidates edge 5 0 2221 tail329 winner.bound = root328Candidates := by
  apply candidates_eq_row edge 5 0 2221 tail329 root328Candidates winner.bound tail329_ordered
  · decide
  · decide
  · funext q
    rfl
def root328Tail40 : List Nat := []
def root328Tail39 : List Nat := 22279 :: root328Tail40
def root328Tail38 : List Nat := 22189 :: root328Tail39
def root328Tail37 : List Nat := 20641 :: root328Tail38
def root328Tail36 : List Nat := 20341 :: root328Tail37
def root328Tail35 : List Nat := 20161 :: root328Tail36
def root328Tail34 : List Nat := 20047 :: root328Tail35
def root328Tail33 : List Nat := 20029 :: root328Tail34
def root328Tail32 : List Nat := 19699 :: root328Tail33
def root328Tail31 : List Nat := 15919 :: root328Tail32
def root328Tail30 : List Nat := 15601 :: root328Tail31
def root328Tail29 : List Nat := 14713 :: root328Tail30
def root328Tail28 : List Nat := 14083 :: root328Tail29
def root328Tail27 : List Nat := 13999 :: root328Tail28
def root328Tail26 : List Nat := 13711 :: root328Tail27
def root328Tail25 : List Nat := 13033 :: root328Tail26
def root328Tail24 : List Nat := 12829 :: root328Tail25
def root328Tail23 : List Nat := 10459 :: root328Tail24
def root328Tail22 : List Nat := 9973 :: root328Tail23
def root328Tail21 : List Nat := 9907 :: root328Tail22
def root328Tail20 : List Nat := 9283 :: root328Tail21
def root328Tail19 : List Nat := 8059 :: root328Tail20
def root328Tail18 : List Nat := 7873 :: root328Tail19
def root328Tail17 : List Nat := 7681 :: root328Tail18
def root328Tail16 : List Nat := 7639 :: root328Tail17
def root328Tail15 : List Nat := 6673 :: root328Tail16
def root328Tail14 : List Nat := 6451 :: root328Tail15
def root328Tail13 : List Nat := 6043 :: root328Tail14
def root328Tail12 : List Nat := 5839 :: root328Tail13
def root328Tail11 : List Nat := 5737 :: root328Tail12
def root328Tail10 : List Nat := 5581 :: root328Tail11
def root328Tail9 : List Nat := 5527 :: root328Tail10
def root328Tail8 : List Nat := 4909 :: root328Tail9
def root328Tail7 : List Nat := 4663 :: root328Tail8
def root328Tail6 : List Nat := 4519 :: root328Tail7
def root328Tail5 : List Nat := 4513 :: root328Tail6
def root328Tail4 : List Nat := 4201 :: root328Tail5
def root328Tail3 : List Nat := 4111 :: root328Tail4
def root328Tail2 : List Nat := 3847 :: root328Tail3
def root328Tail1 : List Nat := 3049 :: root328Tail2
def root328Tail0 : List Nat := 2389 :: root328Tail1
def root328Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root328Candidates0_checked : candidates edge 4 2221 2389 root328Tail1 winner.bound = root328Candidates0 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root328Step0 : scan edge (search edge 3) 4 [2221] 2221 root328Tail0 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail1 winner := by
  rw [root328Tail0, scan, if_neg (by decide), root328Candidates0_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail1
    (if 3 ≤ root328Candidates0.length then search edge 3 [2221, 2389] root328Candidates0 4610 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates1 : List Nat := [5581]
set_option maxRecDepth 8192 in
theorem root328Candidates1_checked : candidates edge 4 2221 3049 root328Tail2 winner.bound = root328Candidates1 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root328Step1 : scan edge (search edge 3) 4 [2221] 2221 root328Tail1 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail2 winner := by
  rw [root328Tail1, scan, if_neg (by decide), root328Candidates1_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail2
    (if 3 ≤ root328Candidates1.length then search edge 3 [2221, 3049] root328Candidates1 5270 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates2 : List Nat := [5737]
set_option maxRecDepth 8192 in
theorem root328Candidates2_checked : candidates edge 4 2221 3847 root328Tail3 winner.bound = root328Candidates2 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root328Step2 : scan edge (search edge 3) 4 [2221] 2221 root328Tail2 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail3 winner := by
  rw [root328Tail2, scan, if_neg (by decide), root328Candidates2_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail3
    (if 3 ≤ root328Candidates2.length then search edge 3 [2221, 3847] root328Candidates2 6068 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates3 : List Nat := [5527, 19699]
set_option maxRecDepth 8192 in
theorem root328Candidates3_checked : candidates edge 4 2221 4111 root328Tail4 winner.bound = root328Candidates3 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root328Step3 : scan edge (search edge 3) 4 [2221] 2221 root328Tail3 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail4 winner := by
  rw [root328Tail3, scan, if_neg (by decide), root328Candidates3_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail4
    (if 3 ≤ root328Candidates3.length then search edge 3 [2221, 4111] root328Candidates3 6332 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates4 : List Nat := [5527]
set_option maxRecDepth 8192 in
theorem root328Candidates4_checked : candidates edge 4 2221 4201 root328Tail5 winner.bound = root328Candidates4 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root328Step4 : scan edge (search edge 3) 4 [2221] 2221 root328Tail4 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail5 winner := by
  rw [root328Tail4, scan, if_neg (by decide), root328Candidates4_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail5
    (if 3 ≤ root328Candidates4.length then search edge 3 [2221, 4201] root328Candidates4 6422 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates5 : List Nat := [5581]
set_option maxRecDepth 8192 in
theorem root328Candidates5_checked : candidates edge 4 2221 4513 root328Tail6 winner.bound = root328Candidates5 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root328Step5 : scan edge (search edge 3) 4 [2221] 2221 root328Tail5 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail6 winner := by
  rw [root328Tail5, scan, if_neg (by decide), root328Candidates5_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail6
    (if 3 ≤ root328Candidates5.length then search edge 3 [2221, 4513] root328Candidates5 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates6 : List Nat := [6043, 9907]
set_option maxRecDepth 8192 in
theorem root328Candidates6_checked : candidates edge 4 2221 4519 root328Tail7 winner.bound = root328Candidates6 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root328Step6 : scan edge (search edge 3) 4 [2221] 2221 root328Tail6 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail7 winner := by
  rw [root328Tail6, scan, if_neg (by decide), root328Candidates6_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail7
    (if 3 ≤ root328Candidates6.length then search edge 3 [2221, 4519] root328Candidates6 6740 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates7 : List Nat := [5839]
set_option maxRecDepth 8192 in
theorem root328Candidates7_checked : candidates edge 4 2221 4663 root328Tail8 winner.bound = root328Candidates7 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root328Step7 : scan edge (search edge 3) 4 [2221] 2221 root328Tail7 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail8 winner := by
  rw [root328Tail7, scan, if_neg (by decide), root328Candidates7_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail8
    (if 3 ≤ root328Candidates7.length then search edge 3 [2221, 4663] root328Candidates7 6884 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates8 : List Nat := [9907]
set_option maxRecDepth 8192 in
theorem root328Candidates8_checked : candidates edge 4 2221 4909 root328Tail9 winner.bound = root328Candidates8 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root328Step8 : scan edge (search edge 3) 4 [2221] 2221 root328Tail8 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail9 winner := by
  rw [root328Tail8, scan, if_neg (by decide), root328Candidates8_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail9
    (if 3 ≤ root328Candidates8.length then search edge 3 [2221, 4909] root328Candidates8 7130 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root328Candidates9_checked : candidates edge 4 2221 5527 root328Tail10 winner.bound = root328Candidates9 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root328Step9 : scan edge (search edge 3) 4 [2221] 2221 root328Tail9 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail10 winner := by
  rw [root328Tail9, scan, if_neg (by decide), root328Candidates9_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail10
    (if 3 ≤ root328Candidates9.length then search edge 3 [2221, 5527] root328Candidates9 7748 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates10 : List Nat := [5737, 5839]
set_option maxRecDepth 8192 in
theorem root328Candidates10_checked : candidates edge 4 2221 5581 root328Tail11 winner.bound = root328Candidates10 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root328Step10 : scan edge (search edge 3) 4 [2221] 2221 root328Tail10 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail11 winner := by
  rw [root328Tail10, scan, if_neg (by decide), root328Candidates10_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail11
    (if 3 ≤ root328Candidates10.length then search edge 3 [2221, 5581] root328Candidates10 7802 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root328Candidates11_checked : candidates edge 4 2221 5737 root328Tail12 winner.bound = root328Candidates11 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root328Step11 : scan edge (search edge 3) 4 [2221] 2221 root328Tail11 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail12 winner := by
  rw [root328Tail11, scan, if_neg (by decide), root328Candidates11_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail12
    (if 3 ≤ root328Candidates11.length then search edge 3 [2221, 5737] root328Candidates11 7958 winner else winner) = _
  rw [if_neg (by decide)]
def root328Candidates12 : List Nat := [7681, 9907, 13999]
set_option maxRecDepth 8192 in
theorem root328Candidates12_checked : candidates edge 4 2221 5839 root328Tail13 winner.bound = root328Candidates12 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root328Child12 : search edge 3 [2221, 5839] root328Candidates12 8060 winner = winner := by decide
theorem root328Step12 : scan edge (search edge 3) 4 [2221] 2221 root328Tail12 winner =
    scan edge (search edge 3) 4 [2221] 2221 root328Tail13 winner := by
  rw [root328Tail12, scan, if_neg (by decide), root328Candidates12_checked]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail13
    (if 3 ≤ root328Candidates12.length then search edge 3 [2221, 5839] root328Candidates12 8060 winner else winner) = _
  rw [if_pos (by decide), root328Child12]
theorem root328Scan13 : scan edge (search edge 3) 4 [2221] 2221 root328Tail13 winner = winner := by rw [root328Tail13, scan, if_pos (by decide)]
theorem root328Scan12 : scan edge (search edge 3) 4 [2221] 2221 root328Tail12 winner = winner :=
  (root328Step12).trans root328Scan13
theorem root328Scan11 : scan edge (search edge 3) 4 [2221] 2221 root328Tail11 winner = winner :=
  (root328Step11).trans root328Scan12
theorem root328Scan10 : scan edge (search edge 3) 4 [2221] 2221 root328Tail10 winner = winner :=
  (root328Step10).trans root328Scan11
theorem root328Scan9 : scan edge (search edge 3) 4 [2221] 2221 root328Tail9 winner = winner :=
  (root328Step9).trans root328Scan10
theorem root328Scan8 : scan edge (search edge 3) 4 [2221] 2221 root328Tail8 winner = winner :=
  (root328Step8).trans root328Scan9
theorem root328Scan7 : scan edge (search edge 3) 4 [2221] 2221 root328Tail7 winner = winner :=
  (root328Step7).trans root328Scan8
theorem root328Scan6 : scan edge (search edge 3) 4 [2221] 2221 root328Tail6 winner = winner :=
  (root328Step6).trans root328Scan7
theorem root328Scan5 : scan edge (search edge 3) 4 [2221] 2221 root328Tail5 winner = winner :=
  (root328Step5).trans root328Scan6
theorem root328Scan4 : scan edge (search edge 3) 4 [2221] 2221 root328Tail4 winner = winner :=
  (root328Step4).trans root328Scan5
theorem root328Scan3 : scan edge (search edge 3) 4 [2221] 2221 root328Tail3 winner = winner :=
  (root328Step3).trans root328Scan4
theorem root328Scan2 : scan edge (search edge 3) 4 [2221] 2221 root328Tail2 winner = winner :=
  (root328Step2).trans root328Scan3
theorem root328Scan1 : scan edge (search edge 3) 4 [2221] 2221 root328Tail1 winner = winner :=
  (root328Step1).trans root328Scan2
theorem root328Scan0 : scan edge (search edge 3) 4 [2221] 2221 root328Tail0 winner = winner :=
  (root328Step0).trans root328Scan1
set_option maxRecDepth 4096 in
theorem root328Branch : search edge 4 [2221] root328Candidates 2221 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2221] 2221 root328Tail0 winner = winner
  exact root328Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
