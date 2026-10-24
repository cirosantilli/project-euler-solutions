import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root396Candidates : List Nat := Data.row2731
set_option maxRecDepth 8192 in
theorem root396Candidates_checked : candidates edge 5 0 2731 tail397 winner.bound = root396Candidates := by
  apply candidates_eq_row edge 5 0 2731 tail397 root396Candidates winner.bound tail397_ordered
  · decide
  · decide
  · funext q
    rfl
def root396Tail43 : List Nat := []
def root396Tail42 : List Nat := 21157 :: root396Tail43
def root396Tail41 : List Nat := 20773 :: root396Tail42
def root396Tail40 : List Nat := 18691 :: root396Tail41
def root396Tail39 : List Nat := 18553 :: root396Tail40
def root396Tail38 : List Nat := 18229 :: root396Tail39
def root396Tail37 : List Nat := 18049 :: root396Tail38
def root396Tail36 : List Nat := 17881 :: root396Tail37
def root396Tail35 : List Nat := 17569 :: root396Tail36
def root396Tail34 : List Nat := 17539 :: root396Tail35
def root396Tail33 : List Nat := 17137 :: root396Tail34
def root396Tail32 : List Nat := 16447 :: root396Tail33
def root396Tail31 : List Nat := 15439 :: root396Tail32
def root396Tail30 : List Nat := 15121 :: root396Tail31
def root396Tail29 : List Nat := 15091 :: root396Tail30
def root396Tail28 : List Nat := 14221 :: root396Tail29
def root396Tail27 : List Nat := 13723 :: root396Tail28
def root396Tail26 : List Nat := 12841 :: root396Tail27
def root396Tail25 : List Nat := 12547 :: root396Tail26
def root396Tail24 : List Nat := 12391 :: root396Tail25
def root396Tail23 : List Nat := 11329 :: root396Tail24
def root396Tail22 : List Nat := 11287 :: root396Tail23
def root396Tail21 : List Nat := 11161 :: root396Tail22
def root396Tail20 : List Nat := 10723 :: root396Tail21
def root396Tail19 : List Nat := 10639 :: root396Tail20
def root396Tail18 : List Nat := 10597 :: root396Tail19
def root396Tail17 : List Nat := 10531 :: root396Tail18
def root396Tail16 : List Nat := 10333 :: root396Tail17
def root396Tail15 : List Nat := 9661 :: root396Tail16
def root396Tail14 : List Nat := 9619 :: root396Tail15
def root396Tail13 : List Nat := 9157 :: root396Tail14
def root396Tail12 : List Nat := 8737 :: root396Tail13
def root396Tail11 : List Nat := 7993 :: root396Tail12
def root396Tail10 : List Nat := 7867 :: root396Tail11
def root396Tail9 : List Nat := 7753 :: root396Tail10
def root396Tail8 : List Nat := 6451 :: root396Tail9
def root396Tail7 : List Nat := 5641 :: root396Tail8
def root396Tail6 : List Nat := 4759 :: root396Tail7
def root396Tail5 : List Nat := 4657 :: root396Tail6
def root396Tail4 : List Nat := 4423 :: root396Tail5
def root396Tail3 : List Nat := 4129 :: root396Tail4
def root396Tail2 : List Nat := 3613 :: root396Tail3
def root396Tail1 : List Nat := 3571 :: root396Tail2
def root396Tail0 : List Nat := 3301 :: root396Tail1
def root396Candidates0 : List Nat := [10531]
set_option maxRecDepth 8192 in
theorem root396Candidates0_checked : candidates edge 4 2731 3301 root396Tail1 winner.bound = root396Candidates0 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root396Step0 : scan edge (search edge 3) 4 [2731] 2731 root396Tail0 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail1 winner := by
  rw [root396Tail0, scan, if_neg (by decide), root396Candidates0_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail1
    (if 3 ≤ root396Candidates0.length then search edge 3 [2731, 3301] root396Candidates0 6032 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates1 : List Nat := [6451, 11329]
set_option maxRecDepth 8192 in
theorem root396Candidates1_checked : candidates edge 4 2731 3571 root396Tail2 winner.bound = root396Candidates1 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root396Step1 : scan edge (search edge 3) 4 [2731] 2731 root396Tail1 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail2 winner := by
  rw [root396Tail1, scan, if_neg (by decide), root396Candidates1_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail2
    (if 3 ≤ root396Candidates1.length then search edge 3 [2731, 3571] root396Candidates1 6302 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates2 : List Nat := [18229]
set_option maxRecDepth 8192 in
theorem root396Candidates2_checked : candidates edge 4 2731 3613 root396Tail3 winner.bound = root396Candidates2 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root396Step2 : scan edge (search edge 3) 4 [2731] 2731 root396Tail2 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail3 winner := by
  rw [root396Tail2, scan, if_neg (by decide), root396Candidates2_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail3
    (if 3 ≤ root396Candidates2.length then search edge 3 [2731, 3613] root396Candidates2 6344 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates3 : List Nat := [10723]
set_option maxRecDepth 8192 in
theorem root396Candidates3_checked : candidates edge 4 2731 4129 root396Tail4 winner.bound = root396Candidates3 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root396Step3 : scan edge (search edge 3) 4 [2731] 2731 root396Tail3 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail4 winner := by
  rw [root396Tail3, scan, if_neg (by decide), root396Candidates3_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail4
    (if 3 ≤ root396Candidates3.length then search edge 3 [2731, 4129] root396Candidates3 6860 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root396Candidates4_checked : candidates edge 4 2731 4423 root396Tail5 winner.bound = root396Candidates4 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root396Step4 : scan edge (search edge 3) 4 [2731] 2731 root396Tail4 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail5 winner := by
  rw [root396Tail4, scan, if_neg (by decide), root396Candidates4_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail5
    (if 3 ≤ root396Candidates4.length then search edge 3 [2731, 4423] root396Candidates4 7154 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates5 : List Nat := [7753, 8737]
set_option maxRecDepth 8192 in
theorem root396Candidates5_checked : candidates edge 4 2731 4657 root396Tail6 winner.bound = root396Candidates5 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root396Step5 : scan edge (search edge 3) 4 [2731] 2731 root396Tail5 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail6 winner := by
  rw [root396Tail5, scan, if_neg (by decide), root396Candidates5_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail6
    (if 3 ≤ root396Candidates5.length then search edge 3 [2731, 4657] root396Candidates5 7388 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates6 : List Nat := [9661, 17539]
set_option maxRecDepth 8192 in
theorem root396Candidates6_checked : candidates edge 4 2731 4759 root396Tail7 winner.bound = root396Candidates6 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root396Step6 : scan edge (search edge 3) 4 [2731] 2731 root396Tail6 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail7 winner := by
  rw [root396Tail6, scan, if_neg (by decide), root396Candidates6_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail7
    (if 3 ≤ root396Candidates6.length then search edge 3 [2731, 4759] root396Candidates6 7490 winner else winner) = _
  rw [if_neg (by decide)]
def root396Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root396Candidates7_checked : candidates edge 4 2731 5641 root396Tail8 winner.bound = root396Candidates7 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root396Step7 : scan edge (search edge 3) 4 [2731] 2731 root396Tail7 winner =
    scan edge (search edge 3) 4 [2731] 2731 root396Tail8 winner := by
  rw [root396Tail7, scan, if_neg (by decide), root396Candidates7_checked]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail8
    (if 3 ≤ root396Candidates7.length then search edge 3 [2731, 5641] root396Candidates7 8372 winner else winner) = _
  rw [if_neg (by decide)]
theorem root396Scan8 : scan edge (search edge 3) 4 [2731] 2731 root396Tail8 winner = winner := by rw [root396Tail8, scan, if_pos (by decide)]
theorem root396Scan7 : scan edge (search edge 3) 4 [2731] 2731 root396Tail7 winner = winner :=
  (root396Step7).trans root396Scan8
theorem root396Scan6 : scan edge (search edge 3) 4 [2731] 2731 root396Tail6 winner = winner :=
  (root396Step6).trans root396Scan7
theorem root396Scan5 : scan edge (search edge 3) 4 [2731] 2731 root396Tail5 winner = winner :=
  (root396Step5).trans root396Scan6
theorem root396Scan4 : scan edge (search edge 3) 4 [2731] 2731 root396Tail4 winner = winner :=
  (root396Step4).trans root396Scan5
theorem root396Scan3 : scan edge (search edge 3) 4 [2731] 2731 root396Tail3 winner = winner :=
  (root396Step3).trans root396Scan4
theorem root396Scan2 : scan edge (search edge 3) 4 [2731] 2731 root396Tail2 winner = winner :=
  (root396Step2).trans root396Scan3
theorem root396Scan1 : scan edge (search edge 3) 4 [2731] 2731 root396Tail1 winner = winner :=
  (root396Step1).trans root396Scan2
theorem root396Scan0 : scan edge (search edge 3) 4 [2731] 2731 root396Tail0 winner = winner :=
  (root396Step0).trans root396Scan1
set_option maxRecDepth 4096 in
theorem root396Branch : search edge 4 [2731] root396Candidates 2731 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2731] 2731 root396Tail0 winner = winner
  exact root396Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
