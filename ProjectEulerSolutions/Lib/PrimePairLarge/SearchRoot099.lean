import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root99Candidates : List Nat := Data.row557
set_option maxRecDepth 8192 in
theorem root99Candidates_checked : candidates edge 5 0 557 tail100 winner.bound = root99Candidates := by
  apply candidates_eq_row edge 5 0 557 tail100 root99Candidates winner.bound tail100_ordered
  · decide
  · decide
  · funext q
    rfl
def root99Tail61 : List Nat := []
def root99Tail60 : List Nat := 24107 :: root99Tail61
def root99Tail59 : List Nat := 23753 :: root99Tail60
def root99Tail58 : List Nat := 23039 :: root99Tail59
def root99Tail57 : List Nat := 21881 :: root99Tail58
def root99Tail56 : List Nat := 21863 :: root99Tail57
def root99Tail55 : List Nat := 21401 :: root99Tail56
def root99Tail54 : List Nat := 21191 :: root99Tail55
def root99Tail53 : List Nat := 21059 :: root99Tail54
def root99Tail52 : List Nat := 20627 :: root99Tail53
def root99Tail51 : List Nat := 20183 :: root99Tail52
def root99Tail50 : List Nat := 20177 :: root99Tail51
def root99Tail49 : List Nat := 20117 :: root99Tail50
def root99Tail48 : List Nat := 18671 :: root99Tail49
def root99Tail47 : List Nat := 17117 :: root99Tail48
def root99Tail46 : List Nat := 16901 :: root99Tail47
def root99Tail45 : List Nat := 16763 :: root99Tail46
def root99Tail44 : List Nat := 16481 :: root99Tail45
def root99Tail43 : List Nat := 16319 :: root99Tail44
def root99Tail42 : List Nat := 16229 :: root99Tail43
def root99Tail41 : List Nat := 15263 :: root99Tail42
def root99Tail40 : List Nat := 15227 :: root99Tail41
def root99Tail39 : List Nat := 15053 :: root99Tail40
def root99Tail38 : List Nat := 14759 :: root99Tail39
def root99Tail37 : List Nat := 14159 :: root99Tail38
def root99Tail36 : List Nat := 13883 :: root99Tail37
def root99Tail35 : List Nat := 13781 :: root99Tail36
def root99Tail34 : List Nat := 13331 :: root99Tail35
def root99Tail33 : List Nat := 13127 :: root99Tail34
def root99Tail32 : List Nat := 13109 :: root99Tail33
def root99Tail31 : List Nat := 12953 :: root99Tail32
def root99Tail30 : List Nat := 12659 :: root99Tail31
def root99Tail29 : List Nat := 12347 :: root99Tail30
def root99Tail28 : List Nat := 12227 :: root99Tail29
def root99Tail27 : List Nat := 12197 :: root99Tail28
def root99Tail26 : List Nat := 12119 :: root99Tail27
def root99Tail25 : List Nat := 12113 :: root99Tail26
def root99Tail24 : List Nat := 11969 :: root99Tail25
def root99Tail23 : List Nat := 11069 :: root99Tail24
def root99Tail22 : List Nat := 10799 :: root99Tail23
def root99Tail21 : List Nat := 10211 :: root99Tail22
def root99Tail20 : List Nat := 8861 :: root99Tail21
def root99Tail19 : List Nat := 8753 :: root99Tail20
def root99Tail18 : List Nat := 8387 :: root99Tail19
def root99Tail17 : List Nat := 8291 :: root99Tail18
def root99Tail16 : List Nat := 8093 :: root99Tail17
def root99Tail15 : List Nat := 8087 :: root99Tail16
def root99Tail14 : List Nat := 7079 :: root99Tail15
def root99Tail13 : List Nat := 6491 :: root99Tail14
def root99Tail12 : List Nat := 6257 :: root99Tail13
def root99Tail11 : List Nat := 6047 :: root99Tail12
def root99Tail10 : List Nat := 5879 :: root99Tail11
def root99Tail9 : List Nat := 5651 :: root99Tail10
def root99Tail8 : List Nat := 4703 :: root99Tail9
def root99Tail7 : List Nat := 3947 :: root99Tail8
def root99Tail6 : List Nat := 3929 :: root99Tail7
def root99Tail5 : List Nat := 3389 :: root99Tail6
def root99Tail4 : List Nat := 3221 :: root99Tail5
def root99Tail3 : List Nat := 2729 :: root99Tail4
def root99Tail2 : List Nat := 2687 :: root99Tail3
def root99Tail1 : List Nat := 1721 :: root99Tail2
def root99Tail0 : List Nat := 1487 :: root99Tail1
def root99Candidates0 : List Nat := [2729, 10799]
set_option maxRecDepth 8192 in
theorem root99Candidates0_checked : candidates edge 4 557 1487 root99Tail1 winner.bound = root99Candidates0 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
theorem root99Step0 : scan edge (search edge 3) 4 [557] 557 root99Tail0 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail1 winner := by
  rw [root99Tail0, scan, if_neg (by decide), root99Candidates0_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail1
    (if 3 ≤ root99Candidates0.length then search edge 3 [557, 1487] root99Candidates0 2044 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates1 : List Nat := [8387, 20177]
set_option maxRecDepth 8192 in
theorem root99Candidates1_checked : candidates edge 4 557 1721 root99Tail2 winner.bound = root99Candidates1 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root99Step1 : scan edge (search edge 3) 4 [557] 557 root99Tail1 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail2 winner := by
  rw [root99Tail1, scan, if_neg (by decide), root99Candidates1_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail2
    (if 3 ≤ root99Candidates1.length then search edge 3 [557, 1721] root99Candidates1 2278 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates2 : List Nat := [20117]
set_option maxRecDepth 8192 in
theorem root99Candidates2_checked : candidates edge 4 557 2687 root99Tail3 winner.bound = root99Candidates2 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root99Step2 : scan edge (search edge 3) 4 [557] 557 root99Tail2 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail3 winner := by
  rw [root99Tail2, scan, if_neg (by decide), root99Candidates2_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail3
    (if 3 ≤ root99Candidates2.length then search edge 3 [557, 2687] root99Candidates2 3244 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates3 : List Nat := [12659]
set_option maxRecDepth 8192 in
theorem root99Candidates3_checked : candidates edge 4 557 2729 root99Tail4 winner.bound = root99Candidates3 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root99Step3 : scan edge (search edge 3) 4 [557] 557 root99Tail3 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail4 winner := by
  rw [root99Tail3, scan, if_neg (by decide), root99Candidates3_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail4
    (if 3 ≤ root99Candidates3.length then search edge 3 [557, 2729] root99Candidates3 3286 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root99Candidates4_checked : candidates edge 4 557 3221 root99Tail5 winner.bound = root99Candidates4 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root99Step4 : scan edge (search edge 3) 4 [557] 557 root99Tail4 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail5 winner := by
  rw [root99Tail4, scan, if_neg (by decide), root99Candidates4_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail5
    (if 3 ≤ root99Candidates4.length then search edge 3 [557, 3221] root99Candidates4 3778 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates5 : List Nat := [5879, 15263, 16901]
set_option maxRecDepth 8192 in
theorem root99Candidates5_checked : candidates edge 4 557 3389 root99Tail6 winner.bound = root99Candidates5 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root99Child5 : search edge 3 [557, 3389] root99Candidates5 3946 winner = winner := by decide
theorem root99Step5 : scan edge (search edge 3) 4 [557] 557 root99Tail5 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail6 winner := by
  rw [root99Tail5, scan, if_neg (by decide), root99Candidates5_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail6
    (if 3 ≤ root99Candidates5.length then search edge 3 [557, 3389] root99Candidates5 3946 winner else winner) = _
  rw [if_pos (by decide), root99Child5]
def root99Candidates6 : List Nat := [16319]
set_option maxRecDepth 8192 in
theorem root99Candidates6_checked : candidates edge 4 557 3929 root99Tail7 winner.bound = root99Candidates6 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root99Step6 : scan edge (search edge 3) 4 [557] 557 root99Tail6 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail7 winner := by
  rw [root99Tail6, scan, if_neg (by decide), root99Candidates6_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail7
    (if 3 ≤ root99Candidates6.length then search edge 3 [557, 3929] root99Candidates6 4486 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates7 : List Nat := [8093, 8861, 12347]
set_option maxRecDepth 8192 in
theorem root99Candidates7_checked : candidates edge 4 557 3947 root99Tail8 winner.bound = root99Candidates7 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root99Child7 : search edge 3 [557, 3947] root99Candidates7 4504 winner = winner := by decide
theorem root99Step7 : scan edge (search edge 3) 4 [557] 557 root99Tail7 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail8 winner := by
  rw [root99Tail7, scan, if_neg (by decide), root99Candidates7_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail8
    (if 3 ≤ root99Candidates7.length then search edge 3 [557, 3947] root99Candidates7 4504 winner else winner) = _
  rw [if_pos (by decide), root99Child7]
def root99Candidates8 : List Nat := [6257, 13331]
set_option maxRecDepth 8192 in
theorem root99Candidates8_checked : candidates edge 4 557 4703 root99Tail9 winner.bound = root99Candidates8 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root99Step8 : scan edge (search edge 3) 4 [557] 557 root99Tail8 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail9 winner := by
  rw [root99Tail8, scan, if_neg (by decide), root99Candidates8_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail9
    (if 3 ≤ root99Candidates8.length then search edge 3 [557, 4703] root99Candidates8 5260 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates9 : List Nat := [8093]
set_option maxRecDepth 8192 in
theorem root99Candidates9_checked : candidates edge 4 557 5651 root99Tail10 winner.bound = root99Candidates9 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root99Step9 : scan edge (search edge 3) 4 [557] 557 root99Tail9 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail10 winner := by
  rw [root99Tail9, scan, if_neg (by decide), root99Candidates9_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail10
    (if 3 ≤ root99Candidates9.length then search edge 3 [557, 5651] root99Candidates9 6208 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root99Candidates10_checked : candidates edge 4 557 5879 root99Tail11 winner.bound = root99Candidates10 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root99Step10 : scan edge (search edge 3) 4 [557] 557 root99Tail10 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail11 winner := by
  rw [root99Tail10, scan, if_neg (by decide), root99Candidates10_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail11
    (if 3 ≤ root99Candidates10.length then search edge 3 [557, 5879] root99Candidates10 6436 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root99Candidates11_checked : candidates edge 4 557 6047 root99Tail12 winner.bound = root99Candidates11 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root99Step11 : scan edge (search edge 3) 4 [557] 557 root99Tail11 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail12 winner := by
  rw [root99Tail11, scan, if_neg (by decide), root99Candidates11_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail12
    (if 3 ≤ root99Candidates11.length then search edge 3 [557, 6047] root99Candidates11 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root99Candidates12 : List Nat := [8387]
set_option maxRecDepth 8192 in
theorem root99Candidates12_checked : candidates edge 4 557 6257 root99Tail13 winner.bound = root99Candidates12 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root99Step12 : scan edge (search edge 3) 4 [557] 557 root99Tail12 winner =
    scan edge (search edge 3) 4 [557] 557 root99Tail13 winner := by
  rw [root99Tail12, scan, if_neg (by decide), root99Candidates12_checked]
  change scan edge (search edge 3) 4 [557] 557 root99Tail13
    (if 3 ≤ root99Candidates12.length then search edge 3 [557, 6257] root99Candidates12 6814 winner else winner) = _
  rw [if_neg (by decide)]
theorem root99Scan13 : scan edge (search edge 3) 4 [557] 557 root99Tail13 winner = winner := by rw [root99Tail13, scan, if_pos (by decide)]
theorem root99Scan12 : scan edge (search edge 3) 4 [557] 557 root99Tail12 winner = winner :=
  (root99Step12).trans root99Scan13
theorem root99Scan11 : scan edge (search edge 3) 4 [557] 557 root99Tail11 winner = winner :=
  (root99Step11).trans root99Scan12
theorem root99Scan10 : scan edge (search edge 3) 4 [557] 557 root99Tail10 winner = winner :=
  (root99Step10).trans root99Scan11
theorem root99Scan9 : scan edge (search edge 3) 4 [557] 557 root99Tail9 winner = winner :=
  (root99Step9).trans root99Scan10
theorem root99Scan8 : scan edge (search edge 3) 4 [557] 557 root99Tail8 winner = winner :=
  (root99Step8).trans root99Scan9
theorem root99Scan7 : scan edge (search edge 3) 4 [557] 557 root99Tail7 winner = winner :=
  (root99Step7).trans root99Scan8
theorem root99Scan6 : scan edge (search edge 3) 4 [557] 557 root99Tail6 winner = winner :=
  (root99Step6).trans root99Scan7
theorem root99Scan5 : scan edge (search edge 3) 4 [557] 557 root99Tail5 winner = winner :=
  (root99Step5).trans root99Scan6
theorem root99Scan4 : scan edge (search edge 3) 4 [557] 557 root99Tail4 winner = winner :=
  (root99Step4).trans root99Scan5
theorem root99Scan3 : scan edge (search edge 3) 4 [557] 557 root99Tail3 winner = winner :=
  (root99Step3).trans root99Scan4
theorem root99Scan2 : scan edge (search edge 3) 4 [557] 557 root99Tail2 winner = winner :=
  (root99Step2).trans root99Scan3
theorem root99Scan1 : scan edge (search edge 3) 4 [557] 557 root99Tail1 winner = winner :=
  (root99Step1).trans root99Scan2
theorem root99Scan0 : scan edge (search edge 3) 4 [557] 557 root99Tail0 winner = winner :=
  (root99Step0).trans root99Scan1
set_option maxRecDepth 4096 in
theorem root99Branch : search edge 4 [557] root99Candidates 557 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [557] 557 root99Tail0 winner = winner
  exact root99Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
